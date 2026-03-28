from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch

from recipe.refplay.dp_actor import RefPlayDataParallelPPOActor
from recipe.refplay.core_algos import get_batch_logps
from verl import DataProto


class NLHFDataParallelPPOActor(RefPlayDataParallelPPOActor):
    def update_policy_nlhf(self, data: DataProto):
        self.actor_module.train()

        batch_td = data.batch
        labels = batch_td["labels"]
        reference_seq_logps = batch_td["reference_seq_logps"]
        preference_probs = batch_td["preference_probs"]

        tau = float(data.meta_info.get("tau", 0.05))
        center_preferences = bool(data.meta_info.get("center_preferences", True))

        micro_batch_size = self.config.get("ppo_micro_batch_size_per_gpu", 1)
        batch_size = batch_td["input_ids"].shape[0]
        if batch_size == 0:
            return {"actor/nlhf_loss": 0.0, "actor/grad_norm": 0.0}

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

            micro_labels = labels[start_idx:end_idx]
            micro_inputs = {
                "input_ids": batch_td["input_ids"][start_idx:end_idx],
                "attention_mask": batch_td["attention_mask"][start_idx:end_idx],
            }
            if "position_ids" in batch_td:
                micro_inputs["position_ids"] = batch_td["position_ids"][start_idx:end_idx]

            micro_ref_seq_logps = reference_seq_logps[start_idx:end_idx]
            micro_preference_probs = preference_probs[start_idx:end_idx]

            autocast_dtype = torch.bfloat16
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                policy_outputs = self.actor_module(**micro_inputs, use_cache=False)
                policy_seq_logps = get_batch_logps(
                    policy_outputs.logits,
                    micro_labels,
                    average_log_prob=False,
                )

                preference_advantage = micro_preference_probs
                if center_preferences:
                    preference_advantage = preference_advantage - 0.5

                kl_estimate = policy_seq_logps - micro_ref_seq_logps
                reinforce_weight = preference_advantage - tau * kl_estimate
                loss = -(reinforce_weight.detach() * policy_seq_logps).mean()
                scaled_loss = loss / gradient_accumulation_steps

                total_loss += loss.item()
                accumulated_metrics["actor/nlhf_loss_batch"].append(loss.item())
                accumulated_metrics["actor/policy_seq_logps_batch"].append(policy_seq_logps.mean().item())
                accumulated_metrics["actor/reference_seq_logps_batch"].append(micro_ref_seq_logps.mean().item())
                accumulated_metrics["actor/preference_prob_batch"].append(micro_preference_probs.mean().item())
                accumulated_metrics["actor/preference_advantage_batch"].append(preference_advantage.mean().item())
                accumulated_metrics["actor/kl_estimate_batch"].append(kl_estimate.mean().item())
                accumulated_metrics["actor/reinforce_weight_batch"].append(reinforce_weight.mean().item())

            if scaled_loss.requires_grad:
                scaled_loss.backward()

        grad_norm = self._optimizer_step()

        metrics["actor/nlhf_loss"] = total_loss / num_micro_batches
        metrics["actor/grad_norm"] = grad_norm.item() if torch.is_tensor(grad_norm) and torch.isfinite(grad_norm) else float("inf")
        for key, val_list in accumulated_metrics.items():
            if val_list:
                metrics[key.replace("_batch", "")] = np.mean(val_list)

        return metrics


__all__ = ["NLHFDataParallelPPOActor"]
