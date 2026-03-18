# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import json
import os
from copy import deepcopy
from pprint import pprint
import traceback

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from recipe.spin.spin_trainer import ResourcePoolManager, Role, RaySPINTrainer, _timer, compute_onlineDPO_pref, compute_response_mask, reduce_metrics
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.utils.tracking import Tracking


class RayRefPlayTrainer(RaySPINTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_opponent_rollout = Role.Rollout in self.role_worker_mapping
        assert self.use_opponent_rollout, "Frozen opponent rollout is required"
        # RefPlay reuses the frozen opponent worker for both generation and
        # reference log-probs to keep memory use bounded on a single GPU.
        self.use_reference_policy = True

    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        actor_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.actor_rollout_ref, role="actor_rollout")
        self.resource_pool_to_cls[actor_pool]["actor_rollout"] = actor_rollout_cls

        opponent_pool = self.resource_pool_manager.get_resource_pool(Role.Rollout)
        opponent_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Rollout], config=self.config.reference_rollout, role="rollout")
        self.resource_pool_to_cls[opponent_pool]["opponent_rollout"] = opponent_rollout_cls

        if self.use_rm:
            rm_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[rm_pool]["rm"] = rm_cls

        all_wg = {}
        self.wg_dicts = []
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            self.wg_dicts.append(wg_dict)

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        self.opponent_rollout_wg = all_wg["opponent_rollout"]
        self.opponent_rollout_wg.init_model()
        self.ref_policy_wg = self.opponent_rollout_wg

        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _score_batch(self, batch: DataProto, reward_callable=None):
        reward_extra_infos_dict = {}
        if self.use_rm:
            rm_scores = self.rm_wg.compute_rm_score(batch)
            batch = batch.union(rm_scores)

        reward_callable = self.reward_fn if reward_callable is None else reward_callable
        try:
            if reward_callable is None:
                reward_tensor = batch.batch.get("rm_scores", torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32))
            else:
                reward_result = reward_callable(batch, return_dict=True)
                reward_tensor = reward_result["reward_tensor"]
                reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
        except Exception:
            traceback.print_exc()
            reward_tensor = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
            reward_extra_infos_dict = {}

        batch.batch["token_level_scores"] = reward_tensor
        batch.batch["token_level_rewards"] = reward_tensor
        batch.batch["seq_level_rewards"] = reward_tensor.sum(dim=-1)
        if reward_extra_infos_dict:
            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
        return batch, reward_extra_infos_dict

    def _pop_generation_batch(self, batch: DataProto) -> DataProto:
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        return batch.pop(batch_keys=batch_keys_to_pop, non_tensor_batch_keys=non_tensor_batch_keys_to_pop)

    def _build_eval_gen_batch(self, batch: DataProto) -> DataProto:
        gen_batch = self._pop_generation_batch(batch)
        gen_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
            "validate": True,
        }
        return gen_batch

    def _decode_rows(self, rows) -> list[str]:
        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in rows]

    def _write_eval_outputs(self, output_dir: str, summary: dict, samples: list[dict], preview_count: int):
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "summary.json")
        samples_path = os.path.join(output_dir, "samples.jsonl")
        report_path = os.path.join(output_dir, "report.md")

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")

        with open(samples_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample, sort_keys=True) + "\n")

        preview_samples = samples[:preview_count]
        report_lines = [
            "# RefPlay Evaluation",
            "",
            f"- checkpoint: `{summary['checkpoint_path']}`",
            f"- reference_model: `{summary['reference_model_path']}`",
            f"- num_examples: `{summary['num_examples']}`",
            f"- actor_reward_mean: `{summary['actor_reward_mean']:.6f}`",
            f"- reference_reward_mean: `{summary['reference_reward_mean']:.6f}`",
            f"- actor_win_rate: `{summary['actor_win_rate']:.6f}`",
            f"- reference_win_rate: `{summary['reference_win_rate']:.6f}`",
            f"- tie_rate: `{summary['tie_rate']:.6f}`",
            f"- reward_margin_mean: `{summary['reward_margin_mean']:.6f}`",
            "",
            "## Preview",
            "",
        ]
        for idx, sample in enumerate(preview_samples, start=1):
            report_lines.extend([
                f"### Sample {idx}",
                f"- winner: `{sample['winner']}`",
                f"- actor_score: `{sample['actor_score']:.6f}`",
                f"- reference_score: `{sample['reference_score']:.6f}`",
                f"- margin: `{sample['reward_margin']:.6f}`",
                "",
                "**Prompt**",
                "",
                "```text",
                sample["prompt"],
                "```",
                "",
                "**Actor**",
                "",
                "```text",
                sample["actor_response"],
                "```",
                "",
                "**Reference**",
                "",
                "```text",
                sample["reference_response"],
                "```",
                "",
            ])

        with open(report_path, "w") as f:
            f.write("\n".join(report_lines).rstrip() + "\n")

        return {
            "summary_path": summary_path,
            "samples_path": samples_path,
            "report_path": report_path,
        }

    def evaluate_against_reference(self, checkpoint_path: str, output_dir: str, preview_count: int = 5):
        if self.config.actor_rollout_ref.rollout.val_kwargs.n != 1:
            raise ValueError("RefPlay eval expects actor_rollout_ref.rollout.val_kwargs.n=1")

        checkpoint_path = os.path.abspath(checkpoint_path)
        if os.path.basename(checkpoint_path) == "actor":
            actor_checkpoint_path = checkpoint_path
            checkpoint_root = os.path.dirname(checkpoint_path)
        else:
            actor_checkpoint_path = os.path.join(checkpoint_path, "actor")
            checkpoint_root = checkpoint_path

        checkpoint_name = os.path.basename(checkpoint_root.rstrip("/"))
        self.actor_rollout_wg.load_checkpoint(
            actor_checkpoint_path,
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
        )

        actor_scores_all = []
        reference_scores_all = []
        samples = []
        numeric_actor_extras = {}
        numeric_reference_extras = {}

        example_index = 0
        for batch_idx, test_data in enumerate(self.val_dataloader):
            test_batch = DataProto.from_single_dict(test_data)
            prompt_texts = self._decode_rows(test_batch.batch["input_ids"])
            batch_data_sources = test_batch.non_tensor_batch.get("data_source", ["unknown"] * len(prompt_texts))

            eval_batch = deepcopy(test_batch)
            test_gen_batch = self._build_eval_gen_batch(eval_batch)

            actor_gen_batch_padded, actor_pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            actor_output_padded = self.actor_rollout_wg.generate_sequences(actor_gen_batch_padded)
            actor_output = unpad_dataproto(actor_output_padded, pad_size=actor_pad_size)

            reference_gen_batch = deepcopy(test_gen_batch)
            reference_gen_batch_padded, reference_pad_size = pad_dataproto_to_divisor(reference_gen_batch, self.opponent_rollout_wg.world_size)
            reference_output_padded = self.opponent_rollout_wg.generate_sequences(reference_gen_batch_padded)
            reference_output = unpad_dataproto(reference_output_padded, pad_size=reference_pad_size)

            actor_batch = deepcopy(test_batch).union(actor_output)
            reference_batch = deepcopy(test_batch).union(reference_output)
            actor_batch.batch["response_mask"] = compute_response_mask(actor_batch)
            reference_batch.batch["response_mask"] = compute_response_mask(reference_batch)

            actor_batch, actor_reward_extra = self._score_batch(actor_batch, reward_callable=self.val_reward_fn)
            reference_batch, reference_reward_extra = self._score_batch(reference_batch, reward_callable=self.val_reward_fn)

            actor_scores = actor_batch.batch["token_level_rewards"].sum(dim=-1).cpu().tolist()
            reference_scores = reference_batch.batch["token_level_rewards"].sum(dim=-1).cpu().tolist()
            actor_scores_all.extend(actor_scores)
            reference_scores_all.extend(reference_scores)

            actor_responses = self._decode_rows(actor_batch.batch["responses"])
            reference_responses = self._decode_rows(reference_batch.batch["responses"])

            for key, values in actor_reward_extra.items():
                if len(values) == len(actor_scores) and all(isinstance(v, (bool, int, float, np.bool_, np.integer, np.floating)) for v in values):
                    numeric_actor_extras.setdefault(key, []).extend([float(v) for v in values])
            for key, values in reference_reward_extra.items():
                if len(values) == len(reference_scores) and all(isinstance(v, (bool, int, float, np.bool_, np.integer, np.floating)) for v in values):
                    numeric_reference_extras.setdefault(key, []).extend([float(v) for v in values])

            for i in range(len(actor_scores)):
                actor_score = float(actor_scores[i])
                reference_score = float(reference_scores[i])
                if actor_score > reference_score:
                    winner = "actor"
                elif reference_score > actor_score:
                    winner = "reference"
                else:
                    winner = "tie"

                sample = {
                    "index": example_index,
                    "batch_index": batch_idx,
                    "data_source": str(batch_data_sources[i]),
                    "prompt": prompt_texts[i],
                    "actor_response": actor_responses[i],
                    "reference_response": reference_responses[i],
                    "actor_score": actor_score,
                    "reference_score": reference_score,
                    "reward_margin": actor_score - reference_score,
                    "winner": winner,
                }
                for key, values in numeric_actor_extras.items():
                    if len(values) > example_index:
                        sample[f"actor_{key}"] = values[example_index]
                for key, values in numeric_reference_extras.items():
                    if len(values) > example_index:
                        sample[f"reference_{key}"] = values[example_index]
                samples.append(sample)
                example_index += 1

        actor_scores_np = np.asarray(actor_scores_all, dtype=np.float32)
        reference_scores_np = np.asarray(reference_scores_all, dtype=np.float32)
        summary = {
            "checkpoint_name": checkpoint_name,
            "checkpoint_path": checkpoint_root,
            "reference_model_path": self.config.reference_rollout.model.path,
            "num_examples": int(actor_scores_np.shape[0]),
            "actor_reward_mean": float(actor_scores_np.mean()),
            "actor_reward_std": float(actor_scores_np.std()),
            "reference_reward_mean": float(reference_scores_np.mean()),
            "reference_reward_std": float(reference_scores_np.std()),
            "actor_win_rate": float((actor_scores_np > reference_scores_np).mean()),
            "reference_win_rate": float((reference_scores_np > actor_scores_np).mean()),
            "tie_rate": float((reference_scores_np == actor_scores_np).mean()),
            "actor_non_loss_rate": float((actor_scores_np >= reference_scores_np).mean()),
            "reward_margin_mean": float((actor_scores_np - reference_scores_np).mean()),
            "reward_margin_std": float((actor_scores_np - reference_scores_np).std()),
        }
        for key, values in sorted(numeric_actor_extras.items()):
            summary[f"actor_{key}_mean"] = float(np.mean(values))
        for key, values in sorted(numeric_reference_extras.items()):
            summary[f"reference_{key}_mean"] = float(np.mean(values))

        output_paths = self._write_eval_outputs(output_dir=output_dir, summary=summary, samples=samples, preview_count=preview_count)
        summary.update(output_paths)
        return summary

    def _build_pair_batch(self, actor_batch: DataProto, opponent_batch: DataProto):
        pair_batch = DataProto.concat([actor_batch, opponent_batch])
        batch_size = actor_batch.batch.batch_size[0]
        interleave_index = torch.arange(batch_size * 2, dtype=torch.long)
        interleave_index = interleave_index.view(2, batch_size).transpose(0, 1).reshape(-1)
        pair_batch.reorder(interleave_index)
        return pair_batch

    def _prepare_dpo_batch(self, batch: DataProto):
        preferences_mask = batch.batch["preferences"]
        not_preferences_mask = ~preferences_mask

        chosen_input_ids = batch.batch["input_ids"][preferences_mask]
        chosen_attention_mask = batch.batch["attention_mask"][preferences_mask]
        rejected_input_ids = batch.batch["input_ids"][not_preferences_mask]
        rejected_attention_mask = batch.batch["attention_mask"][not_preferences_mask]
        chosen_position_ids = batch.batch.get("position_ids")[preferences_mask] if "position_ids" in batch.batch else None
        rejected_position_ids = batch.batch.get("position_ids")[not_preferences_mask] if "position_ids" in batch.batch else None

        prompt_len = self.config.data.max_prompt_length
        chosen_labels = chosen_input_ids.clone()
        chosen_labels[:, :prompt_len] = -100
        rejected_labels = rejected_input_ids.clone()
        rejected_labels[:, :prompt_len] = -100

        ref_sequence_logps = (batch.batch["ref_log_prob"] * batch.batch["response_mask"]).sum(dim=-1)
        reference_chosen_logps = ref_sequence_logps[preferences_mask]
        reference_rejected_logps = ref_sequence_logps[not_preferences_mask]

        dpo_tensors = {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "rejected_labels": rejected_labels,
            "reference_chosen_logps": reference_chosen_logps,
            "reference_rejected_logps": reference_rejected_logps,
        }
        if chosen_position_ids is not None:
            dpo_tensors["chosen_position_ids"] = chosen_position_ids
        if rejected_position_ids is not None:
            dpo_tensors["rejected_position_ids"] = rejected_position_ids

        dpo_meta = {
            "dpo_beta": self.config.algorithm.dpo_beta,
            "dpo_loss_type": self.config.algorithm.dpo_loss_type,
            "dpo_label_smoothing": self.config.algorithm.dpo_label_smoothing,
            "use_reference_policy": True,
            "reference_free": False,
            "global_step": self.global_steps,
        }
        return DataProto.from_dict(tensors=dpo_tensors, meta_info=dpo_meta)

    def fit_dpo(self):
        if self.config.actor_rollout_ref.rollout.n != 1 or self.config.reference_rollout.rollout.n != 1:
            raise ValueError("RefPlay MVP expects one rollout per side: set actor_rollout_ref.rollout.n=1 and reference_rollout.rollout.n=1")

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True, throw_on_missing=False),
        )

        loaded_step = self._load_checkpoint()
        self.global_steps = loaded_step + 1 if loaded_step is not None and loaded_step > 0 else 1

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=max(0, self.global_steps - 1))
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="RefPlay Training Progress")
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if self.global_steps > self.total_training_steps:
                    progress_bar.close()
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    return

                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = batch.pop(batch_keys=batch_keys_to_pop, non_tensor_batch_keys=non_tensor_batch_keys_to_pop)

                is_last_step = self.global_steps >= self.total_training_steps
                debug_step = self.global_steps <= 2

                def log_stage(stage: str):
                    if debug_step:
                        pprint(f"[refplay step {self.global_steps}] {stage}")

                with _timer("step", timing_raw):
                    log_stage("start")
                    with _timer("gen_actor", timing_raw):
                        log_stage("gen_actor")
                        actor_gen = self.actor_rollout_wg.generate_sequences(gen_batch)
                    with _timer("gen_reference", timing_raw):
                        log_stage("gen_reference")
                        opponent_gen = self.opponent_rollout_wg.generate_sequences(deepcopy(gen_batch))

                    log_stage("build_actor_batch")
                    actor_batch = deepcopy(batch).union(actor_gen)
                    log_stage("build_reference_batch")
                    opponent_batch = deepcopy(batch).union(opponent_gen)
                    actor_batch.batch["response_mask"] = compute_response_mask(actor_batch)
                    opponent_batch.batch["response_mask"] = compute_response_mask(opponent_batch)

                    with _timer("reward_actor", timing_raw):
                        log_stage("reward_actor")
                        actor_batch, _ = self._score_batch(actor_batch)
                    with _timer("reward_reference", timing_raw):
                        log_stage("reward_reference")
                        opponent_batch, _ = self._score_batch(opponent_batch)

                    actor_seq_rewards = actor_batch.batch["token_level_rewards"].sum(dim=-1)
                    opponent_seq_rewards = opponent_batch.batch["token_level_rewards"].sum(dim=-1)
                    metrics["game/actor_reward_mean"] = actor_seq_rewards.mean().item()
                    metrics["game/reference_reward_mean"] = opponent_seq_rewards.mean().item()
                    metrics["game/actor_win_rate"] = (actor_seq_rewards >= opponent_seq_rewards).float().mean().item()
                    metrics["game/reward_margin_mean"] = (actor_seq_rewards - opponent_seq_rewards).mean().item()

                    log_stage("build_pair_batch")
                    pair_batch = self._build_pair_batch(actor_batch, opponent_batch)
                    pair_batch.meta_info["global_token_num"] = torch.sum(pair_batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("policy_log_prob", timing_raw):
                        log_stage("policy_log_prob")
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(pair_batch)
                        pair_batch = pair_batch.union(old_log_prob)

                    with _timer("ref_log_prob", timing_raw):
                        log_stage("ref_log_prob")
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(pair_batch)
                        pair_batch = pair_batch.union(ref_log_prob)

                    with _timer("preference", timing_raw):
                        log_stage("preference")
                        pair_batch = compute_onlineDPO_pref(pair_batch)

                    log_stage("prepare_dpo_batch")
                    dpo_update_batch = self._prepare_dpo_batch(pair_batch)

                    with _timer("update_actor", timing_raw):
                        log_stage("update_actor")
                        actor_output = self.actor_rollout_wg.update_actor_dpo(dpo_update_batch)
                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
                    log_stage("step_complete")

                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    with _timer("testing", timing_raw):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
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
