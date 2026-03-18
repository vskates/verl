# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from recipe.spin.spin_trainer import ResourcePoolManager, Role, RaySPINTrainer, _timer, compute_onlineDPO_pref, compute_response_mask, reduce_metrics
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.reward import compute_reward
from verl.utils.tracking import Tracking


class RayRefPlayTrainer(RaySPINTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_opponent_rollout = Role.Rollout in self.role_worker_mapping
        assert self.use_opponent_rollout, "Frozen opponent rollout is required"
        assert self.use_reference_policy, "Frozen reference policy is required for DPO log-probs"

    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        actor_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.actor_rollout_ref, role="actor_rollout")
        self.resource_pool_to_cls[actor_pool]["actor_rollout"] = actor_rollout_cls

        opponent_pool = self.resource_pool_manager.get_resource_pool(Role.Rollout)
        opponent_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Rollout], config=self.config.reference_rollout, role="rollout")
        self.resource_pool_to_cls[opponent_pool]["opponent_rollout"] = opponent_rollout_cls

        ref_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
        ref_policy_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.RefPolicy], config=self.config.reference_rollout, role="ref")
        self.resource_pool_to_cls[ref_pool]["ref"] = ref_policy_cls

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

        self.ref_policy_wg = all_wg["ref"]
        self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        self.opponent_rollout_wg = all_wg["opponent_rollout"]
        self.opponent_rollout_wg.init_model()

        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _score_batch(self, batch: DataProto):
        reward_extra_infos_dict = {}
        if self.use_rm:
            rm_scores = self.rm_wg.compute_rm_score(batch)
            batch = batch.union(rm_scores)

        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
        batch.batch["token_level_scores"] = reward_tensor
        batch.batch["token_level_rewards"] = reward_tensor
        batch.batch["seq_level_rewards"] = reward_tensor.sum(dim=-1)
        return batch, reward_extra_infos_dict

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

                with _timer("step", timing_raw):
                    with _timer("gen_actor", timing_raw):
                        actor_gen = self.actor_rollout_wg.generate_sequences(gen_batch)
                    with _timer("gen_reference", timing_raw):
                        opponent_gen = self.opponent_rollout_wg.generate_sequences(deepcopy(gen_batch))

                    actor_batch = batch.union(actor_gen)
                    opponent_batch = batch.union(opponent_gen)
                    actor_batch.batch["response_mask"] = compute_response_mask(actor_batch)
                    opponent_batch.batch["response_mask"] = compute_response_mask(opponent_batch)

                    with _timer("reward_actor", timing_raw):
                        actor_batch, _ = self._score_batch(actor_batch)
                    with _timer("reward_reference", timing_raw):
                        opponent_batch, _ = self._score_batch(opponent_batch)

                    actor_seq_rewards = actor_batch.batch["token_level_rewards"].sum(dim=-1)
                    opponent_seq_rewards = opponent_batch.batch["token_level_rewards"].sum(dim=-1)
                    metrics["game/actor_reward_mean"] = actor_seq_rewards.mean().item()
                    metrics["game/reference_reward_mean"] = opponent_seq_rewards.mean().item()
                    metrics["game/actor_win_rate"] = (actor_seq_rewards >= opponent_seq_rewards).float().mean().item()
                    metrics["game/reward_margin_mean"] = (actor_seq_rewards - opponent_seq_rewards).mean().item()

                    pair_batch = self._build_pair_batch(actor_batch, opponent_batch)
                    pair_batch.meta_info["global_token_num"] = torch.sum(pair_batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("policy_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(pair_batch)
                        pair_batch = pair_batch.union(old_log_prob)

                    with _timer("ref_log_prob", timing_raw):
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(pair_batch)
                        pair_batch = pair_batch.union(ref_log_prob)

                    with _timer("preference", timing_raw):
                        pair_batch = compute_onlineDPO_pref(pair_batch)

                    dpo_update_batch = self._prepare_dpo_batch(pair_batch)

                    with _timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_wg.update_actor_dpo(dpo_update_batch)
                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

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

