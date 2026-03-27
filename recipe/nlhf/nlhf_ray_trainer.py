import os
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from recipe.refplay.refplay_ray_trainer import RayRefPlayTrainer
from recipe.spin.spin_trainer import Role, _timer, compute_onlineDPO_pref, compute_response_mask, reduce_metrics
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.utils.tracking import Tracking


class RayNLHFTrainer(RayRefPlayTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_reference_policy = True

    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        policy_a_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        policy_a_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.policy_a,
            role="actor_rollout",
        )
        self.resource_pool_to_cls[policy_a_pool]["policy_a"] = policy_a_cls

        policy_b_pool = self.resource_pool_manager.get_resource_pool(Role.Actor)
        policy_b_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Actor],
            config=self.config.policy_b,
            role="actor_rollout",
        )
        self.resource_pool_to_cls[policy_b_pool]["policy_b"] = policy_b_cls

        anchor_pool = self.resource_pool_manager.get_resource_pool(Role.Rollout)
        anchor_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Rollout],
            config=self.config.anchor_rollout,
            role="rollout",
        )
        self.resource_pool_to_cls[anchor_pool]["anchor_rollout"] = anchor_cls

        all_wg = {}
        self.wg_dicts = []
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            self.wg_dicts.append(wg_dict)

        self.policy_a_wg = all_wg["policy_a"]
        self.policy_a_wg.init_model()
        self.actor_rollout_wg = self.policy_a_wg

        self.policy_b_wg = all_wg["policy_b"]
        self.policy_b_wg.init_model()

        self.anchor_rollout_wg = all_wg["anchor_rollout"]
        self.anchor_rollout_wg.init_model()
        self.ref_policy_wg = self.anchor_rollout_wg

    def _normalize_reward_result(self, batch: DataProto, reward_result):
        reward_extra_infos_dict = {}

        if isinstance(reward_result, dict):
            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
            if "reward_tensor" in reward_result:
                reward_tensor = reward_result["reward_tensor"]
            else:
                seq_reward = reward_result.get("sequence_reward")
                if seq_reward is None:
                    seq_reward = reward_result.get("sequence_rewards")
                if seq_reward is None:
                    raise KeyError("Reward callable must return either reward_tensor or sequence_reward(s)")
                seq_reward = torch.as_tensor(seq_reward, dtype=torch.float32, device=batch.batch["response_mask"].device)
                response_mask = batch.batch["response_mask"].to(torch.float32)
                reward_tensor = torch.zeros_like(response_mask, dtype=torch.float32)
                last_indices = response_mask.sum(dim=-1).long() - 1
                last_indices = torch.clamp(last_indices, min=0)
                reward_tensor.scatter_(1, last_indices.unsqueeze(1), seq_reward.unsqueeze(1))
        elif torch.is_tensor(reward_result):
            reward_tensor = reward_result
        else:
            raise TypeError(f"Unsupported reward result type: {type(reward_result)}")

        if reward_tensor.ndim == 1:
            response_mask = batch.batch["response_mask"].to(torch.float32)
            out = torch.zeros_like(response_mask, dtype=torch.float32)
            last_indices = response_mask.sum(dim=-1).long() - 1
            last_indices = torch.clamp(last_indices, min=0)
            out.scatter_(1, last_indices.unsqueeze(1), reward_tensor.to(out.device).unsqueeze(1))
            reward_tensor = out

        return reward_tensor, reward_extra_infos_dict

    def _score_batch(self, batch: DataProto, reward_callable=None):
        reward_extra_infos_dict = {}
        reward_callable = self.reward_fn if reward_callable is None else reward_callable

        try:
            reward_result = reward_callable(batch, return_dict=True)
            reward_tensor, reward_extra_infos_dict = self._normalize_reward_result(batch, reward_result)
        except Exception:
            import traceback

            traceback.print_exc()
            reward_tensor = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
            reward_extra_infos_dict = {}

        batch.batch["token_level_scores"] = reward_tensor
        batch.batch["token_level_rewards"] = reward_tensor
        batch.batch["seq_level_rewards"] = reward_tensor.sum(dim=-1)
        if reward_extra_infos_dict:
            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
        return batch, reward_extra_infos_dict

    def _build_train_gen_batch(self, batch: DataProto) -> DataProto:
        gen_batch = self._pop_generation_batch(batch)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        gen_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": pad_token_id,
            "recompute_log_prob": False,
            "do_sample": self.config.policy_a.rollout.do_sample,
            "validate": False,
            "temperature": self.config.policy_a.rollout.temperature,
            "top_p": self.config.policy_a.rollout.top_p,
            "top_k": self.config.policy_a.rollout.top_k,
            "response_length": self.config.policy_a.rollout.response_length,
        }
        return gen_batch

    def _build_eval_gen_batch(self, batch: DataProto) -> DataProto:
        gen_batch = self._pop_generation_batch(batch)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        gen_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": pad_token_id,
            "recompute_log_prob": False,
            "do_sample": self.config.policy_a.rollout.val_kwargs.do_sample,
            "validate": True,
        }
        return gen_batch

    def _save_checkpoint(self):
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir,
            f"global_step_{self.global_steps}",
        )
        policy_a_local_path = os.path.join(local_global_step_folder, "policy_a")
        policy_b_local_path = os.path.join(local_global_step_folder, "policy_b")

        max_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None)
        self.policy_a_wg.save_checkpoint(policy_a_local_path, None, self.global_steps, max_ckpt_to_keep=max_ckpt_to_keep)
        self.policy_b_wg.save_checkpoint(policy_b_local_path, None, self.global_steps, max_ckpt_to_keep=max_ckpt_to_keep)

        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        latest_file = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(latest_file, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)

        if self.config.trainer.resume_mode == "auto":
            from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

            global_step_folder = find_latest_ckpt_path(checkpoint_folder)
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        elif self.config.trainer.resume_mode == "resume_path":
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                global_step_folder = os.path.join(os.getcwd(), global_step_folder)
        else:
            raise NotImplementedError(f"Unsupported resume mode: {self.config.trainer.resume_mode}")

        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        print(f"Load from checkpoint folder: {global_step_folder}")
        print(f"Setting global step to {self.global_steps}")

        policy_a_path = os.path.join(global_step_folder, "policy_a")
        policy_b_path = os.path.join(global_step_folder, "policy_b")
        resume_weights_only = bool(OmegaConf.select(self.config, "trainer.resume_weights_only", default=False))

        if resume_weights_only:
            self.policy_a_wg.load_model_checkpoint_only(policy_a_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
            self.policy_b_wg.load_model_checkpoint_only(policy_b_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        else:
            self.policy_a_wg.load_checkpoint(policy_a_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
            self.policy_b_wg.load_checkpoint(policy_b_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        return self.global_steps

    def _update_targets_for_step(self):
        mode = self.config.algorithm.get("update_mode", "alternating")
        if mode == "both":
            return ["policy_a", "policy_b"]
        if mode == "policy_a_only":
            return ["policy_a"]
        if mode == "policy_b_only":
            return ["policy_b"]
        if mode == "alternating":
            return ["policy_a"] if self.global_steps % 2 == 1 else ["policy_b"]
        raise ValueError(f"Unsupported update_mode: {mode}")

    def _prefix_metrics(self, metrics: dict, prefix: str):
        renamed = {}
        for key, value in metrics.items():
            if key.startswith("actor/"):
                renamed[key.replace("actor/", f"{prefix}/", 1)] = value
            else:
                renamed[f"{prefix}/{key}"] = value
        return renamed

    def _validate_duel(self, max_samples: int | None = None):
        policy_a_scores_all = []
        policy_b_scores_all = []
        example_index = 0

        for batch_idx, test_data in enumerate(self.val_dataloader):
            if max_samples is not None and example_index >= max_samples:
                break

            test_batch = DataProto.from_single_dict(test_data)
            eval_gen_batch = self._build_eval_gen_batch(deepcopy(test_batch))

            batch_padded, pad_size = pad_dataproto_to_divisor(eval_gen_batch, self.policy_a_wg.world_size)
            policy_a_output = unpad_dataproto(self.policy_a_wg.generate_sequences(batch_padded), pad_size=pad_size)

            batch_padded, pad_size = pad_dataproto_to_divisor(deepcopy(eval_gen_batch), self.policy_b_wg.world_size)
            policy_b_output = unpad_dataproto(self.policy_b_wg.generate_sequences(batch_padded), pad_size=pad_size)

            policy_a_batch = deepcopy(test_batch).union(policy_a_output)
            policy_b_batch = deepcopy(test_batch).union(policy_b_output)
            policy_a_batch.batch["response_mask"] = compute_response_mask(policy_a_batch)
            policy_b_batch.batch["response_mask"] = compute_response_mask(policy_b_batch)

            policy_a_batch, _ = self._score_batch(policy_a_batch, reward_callable=self.val_reward_fn)
            policy_b_batch, _ = self._score_batch(policy_b_batch, reward_callable=self.val_reward_fn)

            a_scores = policy_a_batch.batch["token_level_rewards"].sum(dim=-1).cpu().tolist()
            b_scores = policy_b_batch.batch["token_level_rewards"].sum(dim=-1).cpu().tolist()
            if max_samples is not None:
                keep = min(len(a_scores), max_samples - example_index)
                a_scores = a_scores[:keep]
                b_scores = b_scores[:keep]

            policy_a_scores_all.extend(a_scores)
            policy_b_scores_all.extend(b_scores)
            example_index += len(a_scores)
            pprint(f"[nlhf val] processed {example_index} samples after batch {batch_idx}")

        a_np = np.asarray(policy_a_scores_all, dtype=np.float32)
        b_np = np.asarray(policy_b_scores_all, dtype=np.float32)
        return {
            "val/policy_a_reward_mean": float(a_np.mean()) if a_np.size else 0.0,
            "val/policy_b_reward_mean": float(b_np.mean()) if b_np.size else 0.0,
            "val/policy_a_win_rate": float((a_np > b_np).mean()) if a_np.size else 0.0,
            "val/policy_b_win_rate": float((b_np > a_np).mean()) if a_np.size else 0.0,
            "val/tie_rate": float((a_np == b_np).mean()) if a_np.size else 0.0,
        }

    def fit_nlhf(self):
        if self.config.policy_a.rollout.n != 1 or self.config.policy_b.rollout.n != 1:
            raise ValueError("NLHF MVP expects policy_a.rollout.n=1 and policy_b.rollout.n=1")

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True, throw_on_missing=False),
        )

        loaded_step = self._load_checkpoint()
        self.global_steps = loaded_step + 1 if loaded_step is not None and loaded_step > 0 else 1

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", False):
            val_metrics = self._validate_duel(max_samples=self.config.evaluation.get("max_samples"))
            logger.log(data=val_metrics, step=max(0, self.global_steps - 1))
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="NLHF Training Progress")

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if self.global_steps > self.total_training_steps:
                    progress_bar.close()
                    return

                batch = DataProto.from_single_dict(batch_dict)
                gen_batch = self._build_train_gen_batch(deepcopy(batch))
                metrics = {}
                timing_raw = {}
                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    with _timer("gen_policy_a", timing_raw):
                        policy_a_gen = self.policy_a_wg.generate_sequences(gen_batch)
                    with _timer("gen_policy_b", timing_raw):
                        policy_b_gen = self.policy_b_wg.generate_sequences(deepcopy(gen_batch))

                    policy_a_batch = deepcopy(batch).union(policy_a_gen)
                    policy_b_batch = deepcopy(batch).union(policy_b_gen)
                    policy_a_batch.batch["response_mask"] = compute_response_mask(policy_a_batch)
                    policy_b_batch.batch["response_mask"] = compute_response_mask(policy_b_batch)

                    with _timer("reward_policy_a", timing_raw):
                        policy_a_batch, _ = self._score_batch(policy_a_batch)
                    with _timer("reward_policy_b", timing_raw):
                        policy_b_batch, _ = self._score_batch(policy_b_batch)

                    a_rewards = policy_a_batch.batch["token_level_rewards"].sum(dim=-1)
                    b_rewards = policy_b_batch.batch["token_level_rewards"].sum(dim=-1)
                    metrics["game/policy_a_reward_mean"] = a_rewards.mean().item()
                    metrics["game/policy_b_reward_mean"] = b_rewards.mean().item()
                    metrics["game/policy_a_win_rate"] = (a_rewards > b_rewards).float().mean().item()
                    metrics["game/policy_b_win_rate"] = (b_rewards > a_rewards).float().mean().item()
                    metrics["game/tie_rate"] = (a_rewards == b_rewards).float().mean().item()
                    metrics["game/reward_margin_mean"] = (a_rewards - b_rewards).mean().item()

                    pair_batch = self._build_pair_batch(policy_a_batch, policy_b_batch)
                    pair_batch.meta_info["global_token_num"] = torch.sum(pair_batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("anchor_ref_log_prob", timing_raw):
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(pair_batch)
                        pair_batch = pair_batch.union(ref_log_prob)

                    with _timer("preference", timing_raw):
                        pair_batch = compute_onlineDPO_pref(pair_batch)

                    dpo_update_batch = self._prepare_dpo_batch(pair_batch)

                    for target_policy in self._update_targets_for_step():
                        with _timer(f"update_{target_policy}", timing_raw):
                            if target_policy == "policy_a":
                                output = self.policy_a_wg.update_actor_dpo(dpo_update_batch)
                            else:
                                output = self.policy_b_wg.update_actor_dpo(dpo_update_batch)
                            metrics.update(self._prefix_metrics(reduce_metrics(output.meta_info["metrics"]), target_policy))

                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with _timer("testing", timing_raw):
                        metrics.update(self._validate_duel(max_samples=self.config.evaluation.get("max_samples")))

                should_save = False
                if self.config.trainer.save_freq > 0:
                    should_save = (
                        is_last_step
                        or self.global_steps == 1
                        or self.global_steps % self.config.trainer.save_freq == 0
                    )
                if should_save:
                    with _timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                    **{f"time/{name}": value for name, value in timing_raw.items()},
                })
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

