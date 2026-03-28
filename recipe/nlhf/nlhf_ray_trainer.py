from __future__ import annotations

import os
import shutil
from collections import OrderedDict
from copy import deepcopy
from pprint import pprint
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from recipe.refplay.refplay_ray_trainer import RayRefPlayTrainer
from recipe.spin.spin_trainer import Role, _timer, compute_response_mask, reduce_metrics
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.utils.tracking import Tracking


class RayNLHFTrainer(RayRefPlayTrainer):
    def __init__(self, *args, pairwise_judge: Any, **kwargs):
        super().__init__(*args, **kwargs)
        self.pairwise_judge = pairwise_judge
        self.global_steps = 0
        self.alternative_policy_ema_decay = float(self.config.algorithm.get("alternative_policy_ema_decay", 0.95))
        self.alternative_policy_sync_freq = int(self.config.algorithm.get("alternative_policy_sync_freq", 1))
        self.alternative_policy_strategy = str(self.config.algorithm.get("alternative_policy_strategy", "nash_ema_pg"))
        self.alt_ema_state = None
        self.reference_snapshot_dir = None
        self.policy_wg = None
        self.alternative_policy_wg = None
        self.reference_wg = None

    def _to_python_list(self, values: Any, batch_size: int) -> list[Any]:
        if values is None:
            return [None] * batch_size
        if hasattr(values, "tolist") and not isinstance(values, list):
            values = values.tolist()
        if isinstance(values, tuple):
            values = list(values)
        if isinstance(values, list):
            return values
        return [values] * batch_size

    def _extract_raw_prompts(self, batch: DataProto) -> list[Any]:
        raw_prompts = batch.non_tensor_batch.get("raw_prompt")
        if raw_prompts is None:
            raise KeyError("NLHF expects raw_prompt in non_tensor_batch")
        return self._to_python_list(raw_prompts, batch.batch.batch_size[0])

    def _extract_benchmark_ids(self, batch: DataProto) -> list[str]:
        batch_size = batch.batch.batch_size[0]
        field = self.config.algorithm.get("benchmark_field", "data_source")
        values = batch.non_tensor_batch.get(field)
        if values is None and field != "data_source":
            values = batch.non_tensor_batch.get("data_source")
        values = self._to_python_list(values, batch_size)
        benchmark_ids = []
        for value in values:
            if isinstance(value, bytes):
                benchmark_ids.append(value.decode("utf-8"))
            else:
                benchmark_ids.append(str(value) if value is not None else "unknown")
        return benchmark_ids

    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        policy_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        policy_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.main_policy,
            role="actor_rollout",
        )
        self.resource_pool_to_cls[policy_pool]["policy"] = policy_cls

        alt_pool = self.resource_pool_manager.get_resource_pool(Role.Rollout)
        alt_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Rollout],
            config=self.config.alternative_policy,
            role="rollout",
        )
        self.resource_pool_to_cls[alt_pool]["alternative_policy"] = alt_cls

        ref_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
        ref_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.RefPolicy],
            config=self.config.reference_policy,
            role="ref",
        )
        self.resource_pool_to_cls[ref_pool]["reference_policy"] = ref_cls

        all_wg = {}
        self.wg_dicts = []
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, **wg_kwargs)
            all_wg.update(wg_dict.spawn(prefix_set=class_dict.keys()))
            self.wg_dicts.append(wg_dict)

        self.policy_wg = all_wg["policy"]
        self.policy_wg.init_model()
        self.actor_rollout_wg = self.policy_wg

        self.alternative_policy_wg = all_wg["alternative_policy"]
        self.alternative_policy_wg.init_model()

        self.reference_wg = all_wg["reference_policy"]
        self.reference_wg.init_model()
        self.ref_policy_wg = self.reference_wg

    def _extract_reward_entries(self, batch: DataProto, key: str) -> list[Any]:
        values = batch.non_tensor_batch.get(key)
        batch_size = batch.batch.batch_size[0]
        if values is None:
            return [None] * batch_size
        if hasattr(values, "tolist") and not isinstance(values, list):
            values = values.tolist()
        if isinstance(values, tuple):
            values = list(values)
        if isinstance(values, list):
            return values
        return [values] * batch_size

    def _build_train_gen_batch(self, batch: DataProto) -> DataProto:
        gen_batch = self._pop_generation_batch(batch)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        gen_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": pad_token_id,
            "recompute_log_prob": False,
            "do_sample": self.config.main_policy.rollout.do_sample,
            "validate": False,
            "temperature": self.config.main_policy.rollout.temperature,
            "top_p": self.config.main_policy.rollout.top_p,
            "top_k": self.config.main_policy.rollout.top_k,
            "response_length": self.config.main_policy.rollout.response_length,
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
            "do_sample": self.config.main_policy.rollout.val_kwargs.do_sample,
            "validate": True,
        }
        return gen_batch

    def _generate_with_policy(self, worker_group, gen_batch: DataProto) -> DataProto:
        batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, worker_group.world_size)
        output = worker_group.generate_sequences(batch_padded)
        return unpad_dataproto(output, pad_size=pad_size)

    def _decode_responses(self, output_batch: DataProto) -> list[str]:
        responses = output_batch.batch["responses"].cpu()
        response_mask = compute_response_mask(output_batch).cpu()
        texts = []
        for response_ids, mask in zip(responses, response_mask, strict=True):
            valid_len = int(mask.sum().item())
            texts.append(self.tokenizer.decode(response_ids[:valid_len], skip_special_tokens=True))
        return texts

    def _score_pairwise_preferences(
        self,
        *,
        batch: DataProto,
        policy_texts: list[str],
        alternative_texts: list[str],
    ) -> tuple[torch.Tensor, dict[str, list[Any]]]:
        raw_prompts = self._extract_raw_prompts(batch)
        benchmark_ids = self._extract_benchmark_ids(batch)
        reward_model_entries = self._extract_reward_entries(batch, "reward_model")
        extra_info_entries = self._extract_reward_entries(batch, "extra_info")

        result = self.pairwise_judge.compare_texts(
            prompts=raw_prompts,
            left_responses=policy_texts,
            right_responses=alternative_texts,
            benchmark_ids=benchmark_ids,
            reward_model_entries=reward_model_entries,
            extra_info_entries=extra_info_entries,
        )
        probs = torch.tensor(result["preference_probs"], dtype=torch.float32)
        extras = result.get("pairwise_extra_info", {})
        return probs, extras

    def _build_nlhf_update_batch(self, policy_batch: DataProto, preference_probs: torch.Tensor) -> DataProto:
        ref_batch = self.reference_wg.compute_ref_log_prob(policy_batch)
        reference_seq_logps = (ref_batch.batch["ref_log_prob"] * policy_batch.batch["response_mask"]).sum(dim=-1)

        labels = policy_batch.batch["input_ids"].clone()
        labels[policy_batch.batch["attention_mask"] == 0] = -100
        prompt_width = policy_batch.batch["prompts"].shape[1]
        labels[:, :prompt_width] = -100

        return DataProto.from_dict(
            tensors={
                "input_ids": policy_batch.batch["input_ids"],
                "attention_mask": policy_batch.batch["attention_mask"],
                "position_ids": policy_batch.batch["position_ids"],
                "labels": labels,
                "response_mask": policy_batch.batch["response_mask"],
                "reference_seq_logps": reference_seq_logps,
                "preference_probs": preference_probs.to(reference_seq_logps.device),
            },
            meta_info={
                "tau": float(self.config.algorithm.tau),
                "center_preferences": bool(self.config.algorithm.get("center_preferences", True)),
                "global_step": self.global_steps,
            },
        )

    def _policy_model_file(self, local_dir: str) -> str:
        return os.path.join(local_dir, "model_world_size_1_rank_0.pt")

    def _capture_policy_weights(self, local_dir: str) -> dict[str, torch.Tensor]:
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)
        os.makedirs(local_dir, exist_ok=True)
        snapshot_step = int(getattr(self, "global_steps", 0))
        self.policy_wg.save_checkpoint(local_dir, None, snapshot_step, max_ckpt_to_keep=None)
        return torch.load(self._policy_model_file(local_dir), map_location="cpu", weights_only=False)

    def _write_model_weights(self, local_dir: str, state_dict: dict[str, torch.Tensor]) -> None:
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)
        os.makedirs(local_dir, exist_ok=True)
        torch.save(state_dict, self._policy_model_file(local_dir))

    def _reference_snapshot_root(self) -> str:
        return os.path.join(self.config.trainer.default_local_dir, "_reference_init")

    def _sync_root(self) -> str:
        return os.path.join(self.config.trainer.default_local_dir, "_sync")

    def _initialize_reference_and_alt_from_scratch(self) -> None:
        self.reference_snapshot_dir = self._reference_snapshot_root()
        initial_state = self._capture_policy_weights(self.reference_snapshot_dir)
        self.reference_wg.load_model_weights_only(self.reference_snapshot_dir, del_local_after_load=False)
        self.alt_ema_state = OrderedDict(
            (key, value.clone() if torch.is_tensor(value) else value)
            for key, value in initial_state.items()
        )
        ema_dir = os.path.join(self._sync_root(), "alt_policy_init")
        self._write_model_weights(ema_dir, self.alt_ema_state)
        self.alternative_policy_wg.load_model_weights_only(ema_dir, del_local_after_load=False)

    def _restore_reference_and_alt(self) -> None:
        if self.reference_snapshot_dir is None:
            self.reference_snapshot_dir = self._reference_snapshot_root()
        if not os.path.exists(self._policy_model_file(self.reference_snapshot_dir)):
            self._initialize_reference_and_alt_from_scratch()
            return

        self.reference_wg.load_model_weights_only(self.reference_snapshot_dir, del_local_after_load=False)
        if self.alt_ema_state is None:
            self._initialize_reference_and_alt_from_scratch()
            return
        ema_dir = os.path.join(self._sync_root(), "alt_policy_resume")
        self._write_model_weights(ema_dir, self.alt_ema_state)
        self.alternative_policy_wg.load_model_weights_only(ema_dir, del_local_after_load=False)

    def _update_alternative_policy_ema(self) -> None:
        if self.alternative_policy_strategy != "nash_ema_pg":
            raise ValueError(f"Unsupported alternative_policy_strategy: {self.alternative_policy_strategy}")

        current_state = self._capture_policy_weights(os.path.join(self._sync_root(), "policy_current"))
        if self.alt_ema_state is None:
            self.alt_ema_state = OrderedDict(
                (key, value.clone() if torch.is_tensor(value) else value)
                for key, value in current_state.items()
            )
        else:
            for key, current_value in current_state.items():
                ema_value = self.alt_ema_state[key]
                if torch.is_tensor(current_value) and torch.is_floating_point(current_value):
                    self.alt_ema_state[key] = self.alternative_policy_ema_decay * ema_value + (1.0 - self.alternative_policy_ema_decay) * current_value
                else:
                    self.alt_ema_state[key] = current_value

        ema_dir = os.path.join(self._sync_root(), "alt_policy_live")
        self._write_model_weights(ema_dir, self.alt_ema_state)
        self.alternative_policy_wg.load_model_weights_only(ema_dir, del_local_after_load=False)

    def _save_checkpoint(self):
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        policy_local_path = os.path.join(local_global_step_folder, "policy")
        max_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None)
        self.policy_wg.save_checkpoint(policy_local_path, None, self.global_steps, max_ckpt_to_keep=max_ckpt_to_keep)

        trainer_state = {
            "dataloader": self.train_dataloader.state_dict(),
            "reference_snapshot_dir": self.reference_snapshot_dir,
        }
        if self.alt_ema_state is not None:
            torch.save(self.alt_ema_state, os.path.join(local_global_step_folder, "alt_ema_state.pt"))
        torch.save(trainer_state, os.path.join(local_global_step_folder, "trainer_state.pt"))

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

        policy_path = os.path.join(global_step_folder, "policy")
        resume_weights_only = bool(OmegaConf.select(self.config, "trainer.resume_weights_only", default=False))
        if resume_weights_only:
            self.policy_wg.load_model_checkpoint_only(policy_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        else:
            self.policy_wg.load_checkpoint(policy_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        trainer_state_path = os.path.join(global_step_folder, "trainer_state.pt")
        if os.path.exists(trainer_state_path):
            trainer_state = torch.load(trainer_state_path, weights_only=False)
            dataloader_state = trainer_state.get("dataloader")
            if dataloader_state is not None:
                self.train_dataloader.load_state_dict(dataloader_state)
            self.reference_snapshot_dir = trainer_state.get("reference_snapshot_dir")

        ema_state_path = os.path.join(global_step_folder, "alt_ema_state.pt")
        if os.path.exists(ema_state_path):
            self.alt_ema_state = torch.load(ema_state_path, map_location="cpu", weights_only=False)

        return self.global_steps

    def _validate_duel(self, max_samples: int | None = None):
        example_index = 0
        preference_probs_all = []
        left_scores_all = []
        right_scores_all = []

        for batch_idx, test_data in enumerate(self.val_dataloader):
            if max_samples is not None and example_index >= max_samples:
                break

            batch = DataProto.from_single_dict(test_data)
            gen_batch = self._build_eval_gen_batch(deepcopy(batch))

            policy_output = self._generate_with_policy(self.policy_wg, gen_batch)
            alt_output = self._generate_with_policy(self.alternative_policy_wg, deepcopy(gen_batch))
            policy_texts = self._decode_responses(policy_output)
            alt_texts = self._decode_responses(alt_output)

            probs, extras = self._score_pairwise_preferences(
                batch=batch,
                policy_texts=policy_texts,
                alternative_texts=alt_texts,
            )

            if max_samples is not None:
                keep = min(probs.shape[0], max_samples - example_index)
                probs = probs[:keep]
                if "left_score" in extras:
                    extras["left_score"] = extras["left_score"][:keep]
                if "right_score" in extras:
                    extras["right_score"] = extras["right_score"][:keep]

            preference_probs_all.extend(probs.cpu().tolist())
            left_scores_all.extend(extras.get("left_score", [0.0] * len(probs)))
            right_scores_all.extend(extras.get("right_score", [0.0] * len(probs)))
            example_index += len(probs)
            pprint(f"[nlhf val] processed {example_index} samples after batch {batch_idx}")

        prob_np = np.asarray(preference_probs_all, dtype=np.float32)
        left_np = np.asarray(left_scores_all, dtype=np.float32)
        right_np = np.asarray(right_scores_all, dtype=np.float32)
        return {
            "val/policy_preference_prob_mean": float(prob_np.mean()) if prob_np.size else 0.0,
            "val/policy_win_rate": float((prob_np > 0.5).mean()) if prob_np.size else 0.0,
            "val/alternative_win_rate": float((prob_np < 0.5).mean()) if prob_np.size else 0.0,
            "val/tie_rate": float((prob_np == 0.5).mean()) if prob_np.size else 0.0,
            "val/policy_score_mean": float(left_np.mean()) if left_np.size else 0.0,
            "val/alternative_score_mean": float(right_np.mean()) if right_np.size else 0.0,
        }

    def fit_nlhf(self):
        if self.config.main_policy.rollout.n != 1:
            raise ValueError("NLHF recipe expects main_policy.rollout.n=1")

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True, throw_on_missing=False),
        )

        loaded_step = self._load_checkpoint()
        if loaded_step > 0:
            self._restore_reference_and_alt()
            self.global_steps = loaded_step + 1
        else:
            self._initialize_reference_and_alt_from_scratch()
            self.global_steps = 1

        if self.config.trainer.get("val_before_train", False):
            val_metrics = self._validate_duel(max_samples=self.config.evaluation.get("max_samples"))
            logger.log(data=val_metrics, step=max(0, self.global_steps - 1))
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps - 1, desc="NLHF Training Progress")

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
                    with _timer("gen_policy", timing_raw):
                        policy_gen = self._generate_with_policy(self.policy_wg, gen_batch)
                    with _timer("gen_alternative", timing_raw):
                        alternative_gen = self._generate_with_policy(self.alternative_policy_wg, deepcopy(gen_batch))

                    policy_batch = deepcopy(batch).union(policy_gen)
                    policy_batch.batch["response_mask"] = compute_response_mask(policy_batch)
                    policy_texts = self._decode_responses(policy_gen)
                    alternative_texts = self._decode_responses(alternative_gen)

                    with _timer("pairwise_judge", timing_raw):
                        preference_probs, pairwise_extras = self._score_pairwise_preferences(
                            batch=batch,
                            policy_texts=policy_texts,
                            alternative_texts=alternative_texts,
                        )

                    metrics["game/policy_preference_prob_mean"] = preference_probs.mean().item()
                    metrics["game/policy_win_rate"] = (preference_probs > 0.5).float().mean().item()
                    metrics["game/alternative_win_rate"] = (preference_probs < 0.5).float().mean().item()
                    metrics["game/tie_rate"] = (preference_probs == 0.5).float().mean().item()
                    if "left_score" in pairwise_extras:
                        metrics["game/policy_score_mean"] = float(np.mean(pairwise_extras["left_score"]))
                    if "right_score" in pairwise_extras:
                        metrics["game/alternative_score_mean"] = float(np.mean(pairwise_extras["right_score"]))
                    if "left_exact_match" in pairwise_extras:
                        metrics["game/policy_exact_match_mean"] = float(np.mean(pairwise_extras["left_exact_match"]))
                    if "right_exact_match" in pairwise_extras:
                        metrics["game/alternative_exact_match_mean"] = float(np.mean(pairwise_extras["right_exact_match"]))

                    update_batch = self._build_nlhf_update_batch(policy_batch, preference_probs)
                    with _timer("update_policy", timing_raw):
                        output = self.policy_wg.update_actor_nlhf(update_batch)
                    metrics.update(reduce_metrics(output.meta_info["metrics"]))

                    if self.global_steps % self.alternative_policy_sync_freq == 0:
                        with _timer("sync_alternative_policy", timing_raw):
                            self._update_alternative_policy_ema()

                if self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    with _timer("testing", timing_raw):
                        metrics.update(self._validate_duel(max_samples=self.config.evaluation.get("max_samples")))

                should_save = False
                if self.config.trainer.save_freq > 0:
                    should_save = is_last_step or self.global_steps == 1 or self.global_steps % self.config.trainer.save_freq == 0
                if should_save:
                    with _timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                        **{f"time/{name}": value for name, value in timing_raw.items()},
                    }
                )
                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1
