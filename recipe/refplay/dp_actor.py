# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import itertools
import math
from collections import defaultdict

import numpy as np
import torch

from recipe.refplay.core_algos import compute_online_dpo_loss, get_batch_logps
from verl import DataProto
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.workers.actor import DataParallelPPOActor

__all__ = ["DataParallelPPOActor"]


class RefPlayDataParallelPPOActor(DataParallelPPOActor):
    def compute_log_prob(
        self,
        data: DataProto,
        calculate_entropy: bool = False,
        calculate_sum_pi_squared: bool = False,
        **_: dict,
    ) -> torch.Tensor:
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        if not calculate_entropy and not calculate_sum_pi_squared:
            return log_probs

        extra_outputs = [log_probs]
        if calculate_entropy:
            extra_outputs.append(torch.zeros_like(log_probs))
        if calculate_sum_pi_squared:
            extra_outputs.append(torch.zeros_like(log_probs))
        return tuple(extra_outputs)

    def update_policy_dpo_with_ref(self, data: DataProto):
        self.actor_module.train()

        batch_td = data.batch
        chosen_labels = batch_td["chosen_labels"]
        rejected_labels = batch_td["rejected_labels"]
        reference_chosen_logps = batch_td["reference_chosen_logps"]
        reference_rejected_logps = batch_td["reference_rejected_logps"]

        beta = self.config.get("dpo_beta", 0.1)
        loss_type = data.meta_info.get("dpo_loss_type", "sigmoid")
        label_smoothing = data.meta_info.get("dpo_label_smoothing", 0.0)
        reference_free = data.meta_info.get("reference_free", False)

        micro_batch_size = self.config.get("ppo_micro_batch_size_per_gpu", 1)
        batch_size = batch_td["chosen_input_ids"].shape[0]
        if batch_size == 0:
            return {"actor/dpo_loss": 0.0, "actor/grad_norm": 0.0}

        num_micro_batches = math.ceil(batch_size / micro_batch_size)
        gradient_accumulation_steps = num_micro_batches

        total_loss = 0.0
        accumulated_metrics = defaultdict(list)
        metrics = {}

        self.actor_optimizer.zero_grad(set_to_none=True)

        for i in range(num_micro_batches):
            start_idx = i * micro_batch_size
            end_idx = min(start_idx + micro_batch_size, batch_size)
            if start_idx >= end_idx:
                continue

            micro_batch_chosen_labels = chosen_labels[start_idx:end_idx]
            micro_batch_rejected_labels = rejected_labels[start_idx:end_idx]
            micro_batch_chosen_inputs = {
                "input_ids": batch_td["chosen_input_ids"][start_idx:end_idx],
                "attention_mask": batch_td["chosen_attention_mask"][start_idx:end_idx],
            }
            if "chosen_position_ids" in batch_td:
                micro_batch_chosen_inputs["position_ids"] = batch_td["chosen_position_ids"][start_idx:end_idx]

            micro_batch_rejected_inputs = {
                "input_ids": batch_td["rejected_input_ids"][start_idx:end_idx],
                "attention_mask": batch_td["rejected_attention_mask"][start_idx:end_idx],
            }
            if "rejected_position_ids" in batch_td:
                micro_batch_rejected_inputs["position_ids"] = batch_td["rejected_position_ids"][start_idx:end_idx]

            autocast_dtype = torch.bfloat16
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                policy_chosen_outputs = self.actor_module(**micro_batch_chosen_inputs, use_cache=False)
                policy_rejected_outputs = self.actor_module(**micro_batch_rejected_inputs, use_cache=False)

                policy_chosen_logps = get_batch_logps(policy_chosen_outputs.logits, micro_batch_chosen_labels, average_log_prob=False)
                policy_rejected_logps = get_batch_logps(policy_rejected_outputs.logits, micro_batch_rejected_labels, average_log_prob=False)

                micro_ref_chosen_logps = reference_chosen_logps[start_idx:end_idx]
                micro_ref_rejected_logps = reference_rejected_logps[start_idx:end_idx]

                logits = (policy_chosen_logps - policy_rejected_logps) - (micro_ref_chosen_logps - micro_ref_rejected_logps)
                loss = compute_online_dpo_loss(
                    policy_chosen_logps=policy_chosen_logps,
                    policy_rejected_logps=policy_rejected_logps,
                    reference_chosen_logps=micro_ref_chosen_logps,
                    reference_rejected_logps=micro_ref_rejected_logps,
                    beta=beta,
                    label_smoothing=label_smoothing,
                    loss_type=loss_type,
                    reference_free=reference_free,
                )
                scaled_loss = loss / gradient_accumulation_steps

                total_loss += loss.item()
                accumulated_metrics["actor/dpo_loss_batch"].append(loss.item())
                accumulated_metrics["actor/dpo_logits_batch"].append(logits.mean().item())
                accumulated_metrics["actor/policy_chosen_logps_batch"].append(policy_chosen_logps.mean().item())
                accumulated_metrics["actor/policy_rejected_logps_batch"].append(policy_rejected_logps.mean().item())
                accumulated_metrics["actor/reference_chosen_logps_batch"].append(micro_ref_chosen_logps.mean().item())
                accumulated_metrics["actor/reference_rejected_logps_batch"].append(micro_ref_rejected_logps.mean().item())

            if scaled_loss.requires_grad:
                scaled_loss.backward()

        grad_norm = self._optimizer_step()

        metrics["actor/dpo_loss"] = total_loss / num_micro_batches
        metrics["actor/grad_norm"] = grad_norm.item() if torch.is_tensor(grad_norm) and torch.isfinite(grad_norm) else float("inf")
        for key, val_list in accumulated_metrics.items():
            if val_list:
                metrics[key.replace("_batch", "")] = np.mean(val_list)

        if "actor/policy_chosen_logps" in metrics:
            policy_ratio_mean = metrics["actor/policy_chosen_logps"] - metrics["actor/policy_rejected_logps"]
            ref_ratio_mean = metrics["actor/reference_chosen_logps"] - metrics["actor/reference_rejected_logps"]
            logits_mean = policy_ratio_mean - ref_ratio_mean
            metrics["actor/rewards_chosen"] = beta * (metrics["actor/policy_chosen_logps"] - metrics["actor/reference_chosen_logps"])
            metrics["actor/rewards_rejected"] = beta * (metrics["actor/policy_rejected_logps"] - metrics["actor/reference_rejected_logps"])
            metrics["actor/rewards_accuracies"] = float(logits_mean > 0)
            metrics["actor/rewards_margins"] = metrics["actor/rewards_chosen"] - metrics["actor/rewards_rejected"]

        return metrics
