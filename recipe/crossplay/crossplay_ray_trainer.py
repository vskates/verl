from __future__ import annotations

import os
import re
import shutil
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from recipe.spin.spin_trainer import RaySPINTrainer, Role, _timer, compute_response_mask, reduce_metrics
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.utils.tracking import Tracking


def _sanitize_metric_key(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_") or "unknown"


class RayCrossPlayTrainer(RaySPINTrainer):
    def __init__(self, *args, policy_tokenizers: dict[str, Any], text_judge: Any, **kwargs):
        super().__init__(*args, reward_fn=None, val_reward_fn=None, **kwargs)
        self.policy_tokenizers = policy_tokenizers
        self.text_judge = text_judge
        self.benchmark_stats = defaultdict(self._make_benchmark_state)
        self.use_reference_policy = bool(self.config.algorithm.get("use_reference_policy", True))
        if not self.use_reference_policy:
            raise ValueError("CrossPlay ideal pipeline requires per-policy frozen anchors and reference DPO")
        self.policy_failure_replay = {
            "policy_a": defaultdict(list),
            "policy_b": defaultdict(list),
        }
        self.replay_rng = np.random.default_rng(int(self.config.data.get("seed", 0)))
        self.anchor_a_wg = None
        self.anchor_b_wg = None

    def _validate_config(self):
        """Relax SPIN's global-GPU batch divisibility check for cross-play.

        Cross-play runs one trainable policy per resource pool/GPU instead of a single
        data-parallel policy replicated across all visible GPUs. Because of that,
        requiring `real_train_batch_size % total_n_gpus == 0` is overly strict here
        and incorrectly rejects otherwise valid configurations such as global batch 3
        on a 2-policy duel. We temporarily validate the inherited SPIN constraints
        under an effective single-GPU policy update assumption, then restore the
        original trainer settings unchanged.
        """
        original_n_gpus_per_node = self.config.trainer.n_gpus_per_node
        try:
            OmegaConf.set_struct(self.config, False)
            with open_dict(self.config):
                self.config.trainer.n_gpus_per_node = 1
            super()._validate_config()
        finally:
            OmegaConf.set_struct(self.config, False)
            with open_dict(self.config):
                self.config.trainer.n_gpus_per_node = original_n_gpus_per_node

    @staticmethod
    def _make_benchmark_state():
        return {
            "count": 0,
            "policy_a_wins": 0,
            "policy_b_wins": 0,
            "ties": 0,
            "policy_a_reward_sum": 0.0,
            "policy_b_reward_sum": 0.0,
        }

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

        if self.use_reference_policy:
            anchor_a_pool = self.resource_pool_manager.get_resource_pool(Role.Rollout)
            anchor_a_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Rollout],
                config=self.config.anchor_a,
                role="rollout",
            )
            self.resource_pool_to_cls[anchor_a_pool]["anchor_a"] = anchor_a_cls

            anchor_b_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            anchor_b_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RefPolicy],
                config=self.config.anchor_b,
                role="rollout",
            )
            self.resource_pool_to_cls[anchor_b_pool]["anchor_b"] = anchor_b_cls

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

        self.policy_a_wg = all_wg["policy_a"]
        self.policy_a_wg.init_model()
        self.policy_b_wg = all_wg["policy_b"]
        self.policy_b_wg.init_model()
        if self.use_reference_policy:
            self.anchor_a_wg = all_wg["anchor_a"]
            self.anchor_a_wg.init_model()
            self.anchor_b_wg = all_wg["anchor_b"]
            self.anchor_b_wg.init_model()

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

        resume_weights_only = bool(OmegaConf.select(self.config, "trainer.resume_weights_only", default=False))
        checkpoints = {
            "policy_a": os.path.join(global_step_folder, "policy_a"),
            "policy_b": os.path.join(global_step_folder, "policy_b"),
        }
        for name, wg in (("policy_a", self.policy_a_wg), ("policy_b", self.policy_b_wg)):
            if resume_weights_only:
                wg.load_model_checkpoint_only(checkpoints[name], del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
            else:
                wg.load_checkpoint(checkpoints[name], del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        trainer_state_path = os.path.join(global_step_folder, "trainer_state.pt")
        if os.path.exists(trainer_state_path):
            trainer_state = torch.load(trainer_state_path, weights_only=False)
            dataloader_state = trainer_state.get("dataloader")
            if dataloader_state is not None:
                self.train_dataloader.load_state_dict(dataloader_state)
            benchmark_stats = trainer_state.get("benchmark_stats")
            if benchmark_stats is not None:
                self.benchmark_stats = defaultdict(
                    self._make_benchmark_state,
                    benchmark_stats,
                )
            failure_replay = trainer_state.get("policy_failure_replay")
            if failure_replay is not None:
                self.policy_failure_replay = {
                    "policy_a": defaultdict(list, failure_replay.get("policy_a", {})),
                    "policy_b": defaultdict(list, failure_replay.get("policy_b", {})),
                }
        return self.global_steps

    def _save_checkpoint(self):
        os.makedirs(self.config.trainer.default_local_dir, exist_ok=True)

        max_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None)
        if max_ckpt_to_keep is not None and max_ckpt_to_keep > 0:
            checkpoint_dirs = []
            for name in os.listdir(self.config.trainer.default_local_dir):
                if not name.startswith("global_step_"):
                    continue
                step_str = name.removeprefix("global_step_")
                if not step_str.isdigit():
                    continue
                checkpoint_dirs.append((int(step_str), os.path.join(self.config.trainer.default_local_dir, name)))
            checkpoint_dirs.sort(key=lambda item: item[0])
            while len(checkpoint_dirs) >= max_ckpt_to_keep:
                _, old_path = checkpoint_dirs.pop(0)
                shutil.rmtree(old_path, ignore_errors=True)

        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        self.policy_a_wg.save_checkpoint(os.path.join(local_global_step_folder, "policy_a"), None, self.global_steps, max_ckpt_to_keep=max_ckpt_to_keep)
        self.policy_b_wg.save_checkpoint(os.path.join(local_global_step_folder, "policy_b"), None, self.global_steps, max_ckpt_to_keep=max_ckpt_to_keep)

        trainer_state = {
            "dataloader": self.train_dataloader.state_dict(),
            "benchmark_stats": dict(self.benchmark_stats),
            "policy_failure_replay": {
                "policy_a": dict(self.policy_failure_replay["policy_a"]),
                "policy_b": dict(self.policy_failure_replay["policy_b"]),
            },
        }
        torch.save(trainer_state, os.path.join(local_global_step_folder, "trainer_state.pt"))

        latest_file = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(latest_file, "w") as f:
            f.write(str(self.global_steps))

        if max_ckpt_to_keep is not None and max_ckpt_to_keep > 0:
            checkpoint_dirs = []
            for name in os.listdir(self.config.trainer.default_local_dir):
                if not name.startswith("global_step_"):
                    continue
                step_str = name.removeprefix("global_step_")
                if not step_str.isdigit():
                    continue
                checkpoint_dirs.append((int(step_str), os.path.join(self.config.trainer.default_local_dir, name)))
            checkpoint_dirs.sort(key=lambda item: item[0])
            while len(checkpoint_dirs) > max_ckpt_to_keep:
                _, old_path = checkpoint_dirs.pop(0)
                shutil.rmtree(old_path, ignore_errors=True)

    def _update_targets_for_step(self):
        mode = self.config.algorithm.get("update_mode", "both")
        if mode == "both":
            return ["policy_a", "policy_b"]
        if mode == "policy_a_only":
            return ["policy_a"]
        if mode == "policy_b_only":
            return ["policy_b"]
        if mode == "alternating":
            return ["policy_a"] if self.global_steps % 2 == 1 else ["policy_b"]
        raise ValueError(f"Unsupported update_mode: {mode}")

    def _prefix_metrics(self, metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
        renamed = {}
        for key, value in metrics.items():
            if key.startswith("actor/"):
                renamed[key.replace("actor/", f"{prefix}/", 1)] = value
            else:
                renamed[f"{prefix}/{key}"] = value
        return renamed

    def _to_python_list(self, values: Any, batch_size: int) -> list[Any]:
        if values is None:
            return [None] * batch_size
        if hasattr(values, "tolist") and not isinstance(values, list):
            values = values.tolist()
        if isinstance(values, tuple):
            values = list(values)
        if not isinstance(values, list):
            return [values] * batch_size
        return values

    def _extract_raw_prompts(self, batch: DataProto) -> list[Any]:
        raw_prompts = batch.non_tensor_batch.get("raw_prompt")
        if raw_prompts is None:
            raise KeyError("CrossPlay expects raw_prompt in non_tensor_batch")
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

    def _extract_reward_entries(self, batch: DataProto, key: str) -> list[Any]:
        return self._to_python_list(batch.non_tensor_batch.get(key), batch.batch.batch_size[0])

    def _serialize_prompt(self, tokenizer, raw_prompt: Any) -> str:
        chat = raw_prompt if isinstance(raw_prompt, list) else raw_prompt.tolist()
        return tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

    def _build_policy_generation_batch(self, batch: DataProto, policy_name: str, validate: bool = False) -> DataProto:
        tokenizer = self.policy_tokenizers[policy_name]
        raw_prompts = self._extract_raw_prompts(batch)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        input_ids_rows = []
        attention_mask_rows = []
        max_prompt_length = self.config.data.max_prompt_length
        for raw_prompt in raw_prompts:
            prompt_text = self._serialize_prompt(tokenizer, raw_prompt)
            encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
            input_ids = encoded["input_ids"][0]
            attention_mask = encoded["attention_mask"][0]
            if input_ids.numel() > max_prompt_length:
                input_ids = input_ids[-max_prompt_length:]
                attention_mask = attention_mask[-max_prompt_length:]
            input_ids_rows.append(input_ids)
            attention_mask_rows.append(attention_mask)

        prompt_width = max(row.size(0) for row in input_ids_rows)
        padded_input_ids = []
        padded_attention_mask = []
        padded_position_ids = []
        for input_ids, attention_mask in zip(input_ids_rows, attention_mask_rows, strict=True):
            pad_len = prompt_width - input_ids.size(0)
            if pad_len > 0:
                input_ids = torch.cat([torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype), input_ids], dim=0)
                attention_mask = torch.cat([torch.zeros(pad_len, dtype=attention_mask.dtype), attention_mask], dim=0)
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)
            padded_position_ids.append(position_ids)

        rollout_cfg = getattr(self.config, policy_name).rollout
        meta_info = {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": pad_token_id,
            "recompute_log_prob": False,
            "do_sample": rollout_cfg.val_kwargs.do_sample if validate else rollout_cfg.do_sample,
            "validate": validate,
        }
        if not validate:
            meta_info.update(
                {
                    "temperature": rollout_cfg.temperature,
                    "top_p": rollout_cfg.top_p,
                    "top_k": rollout_cfg.top_k,
                    "response_length": rollout_cfg.response_length,
                }
            )

        return DataProto.from_dict(
            tensors={
                "input_ids": torch.stack(padded_input_ids, dim=0),
                "attention_mask": torch.stack(padded_attention_mask, dim=0),
                "position_ids": torch.stack(padded_position_ids, dim=0),
            },
            non_tensors={key: batch.non_tensor_batch[key] for key in batch.non_tensor_batch.keys()},
            meta_info=meta_info,
        )

    def _generate_with_policy(self, worker_group, gen_batch: DataProto) -> DataProto:
        batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, worker_group.world_size)
        output = worker_group.generate_sequences(batch_padded)
        return unpad_dataproto(output, pad_size=pad_size)

    def _decode_responses(self, tokenizer, output_batch: DataProto) -> list[str]:
        responses = output_batch.batch["responses"].cpu()
        response_mask = compute_response_mask(output_batch).cpu()
        texts = []
        for response_ids, mask in zip(responses, response_mask, strict=True):
            valid_len = int(mask.sum().item())
            texts.append(tokenizer.decode(response_ids[:valid_len], skip_special_tokens=True))
        return texts

    def _score_response_texts(self, *, raw_prompts: list[Any], responses: list[str], benchmark_ids: list[str], reward_model_entries: list[Any], extra_info_entries: list[Any]):
        result = self.text_judge.score_texts(
            prompts=raw_prompts,
            responses=responses,
            benchmark_ids=benchmark_ids,
            reward_model_entries=reward_model_entries,
            extra_info_entries=extra_info_entries,
        )
        seq_rewards = torch.tensor(result["sequence_rewards"], dtype=torch.float32)
        extras = result.get("reward_extra_info", {})
        return seq_rewards, extras

    def _build_sequence_batch(self, *, tokenizer, raw_prompts: list[Any], response_texts: list[str]) -> tuple[DataProto, torch.Tensor]:
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        prompt_rows = []
        response_rows = []
        for raw_prompt, response_text in zip(raw_prompts, response_texts, strict=True):
            prompt_text = self._serialize_prompt(tokenizer, raw_prompt)
            prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            response_ids = tokenizer(response_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

            prompt_ids = prompt_ids[-self.config.data.max_prompt_length :]
            response_ids = response_ids[: self.config.data.max_response_length]
            prompt_rows.append(prompt_ids)
            response_rows.append(response_ids)

        prompt_width = max((row.size(0) for row in prompt_rows), default=1)
        response_width = max((max(row.size(0), 1) for row in response_rows), default=1)
        input_id_rows = []
        attention_mask_rows = []
        position_ids_rows = []
        response_token_rows = []
        response_mask_rows = []
        prompt_widths = []

        for prompt_ids, response_ids in zip(prompt_rows, response_rows, strict=True):
            prompt_pad = prompt_width - prompt_ids.size(0)
            if prompt_pad > 0:
                padded_prompt = torch.cat([torch.full((prompt_pad,), pad_token_id, dtype=prompt_ids.dtype), prompt_ids], dim=0)
                prompt_mask = torch.cat([torch.zeros(prompt_pad, dtype=torch.long), torch.ones(prompt_ids.size(0), dtype=torch.long)], dim=0)
            else:
                padded_prompt = prompt_ids
                prompt_mask = torch.ones(prompt_ids.size(0), dtype=torch.long)

            valid_response_len = response_ids.size(0)
            if valid_response_len == 0:
                padded_response = torch.full((response_width,), pad_token_id, dtype=prompt_ids.dtype)
                response_mask = torch.zeros(response_width, dtype=torch.long)
            else:
                response_pad = response_width - valid_response_len
                padded_response = torch.cat(
                    [response_ids, torch.full((response_pad,), pad_token_id, dtype=response_ids.dtype)],
                    dim=0,
                )
                response_mask = torch.cat(
                    [torch.ones(valid_response_len, dtype=torch.long), torch.zeros(response_pad, dtype=torch.long)],
                    dim=0,
                )

            input_ids = torch.cat([padded_prompt, padded_response], dim=0)
            attention_mask = torch.cat([prompt_mask, response_mask], dim=0)
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)

            input_id_rows.append(input_ids)
            attention_mask_rows.append(attention_mask)
            position_ids_rows.append(position_ids)
            response_token_rows.append(padded_response)
            response_mask_rows.append(response_mask)
            prompt_widths.append(prompt_width)

        batch = DataProto.from_dict(
            tensors={
                "input_ids": torch.stack(input_id_rows, dim=0),
                "attention_mask": torch.stack(attention_mask_rows, dim=0),
                "position_ids": torch.stack(position_ids_rows, dim=0),
                "responses": torch.stack(response_token_rows, dim=0),
                "response_mask": torch.stack(response_mask_rows, dim=0),
            }
        )
        return batch, torch.tensor(prompt_widths, dtype=torch.long)

    def _build_policy_update_batch(
        self,
        *,
        policy_name: str,
        raw_prompts: list[Any],
        chosen_texts: list[str],
        rejected_texts: list[str],
    ) -> DataProto | None:
        if not raw_prompts:
            return None

        tokenizer = self.policy_tokenizers[policy_name]
        chosen_batch, prompt_widths = self._build_sequence_batch(tokenizer=tokenizer, raw_prompts=raw_prompts, response_texts=chosen_texts)
        rejected_batch, _ = self._build_sequence_batch(tokenizer=tokenizer, raw_prompts=raw_prompts, response_texts=rejected_texts)
        if self.use_reference_policy:
            anchor_wg = self.anchor_a_wg if policy_name == "policy_a" else self.anchor_b_wg
            chosen_ref = anchor_wg.compute_ref_log_prob(chosen_batch)
            rejected_ref = anchor_wg.compute_ref_log_prob(rejected_batch)
            reference_chosen_logps = (chosen_ref.batch["ref_log_prob"] * chosen_batch.batch["response_mask"]).sum(dim=-1)
            reference_rejected_logps = (rejected_ref.batch["ref_log_prob"] * rejected_batch.batch["response_mask"]).sum(dim=-1)
        else:
            batch_size = chosen_batch.batch["input_ids"].shape[0]
            reference_chosen_logps = torch.zeros(batch_size, dtype=torch.float32)
            reference_rejected_logps = torch.zeros(batch_size, dtype=torch.float32)

        chosen_labels = chosen_batch.batch["input_ids"].clone()
        rejected_labels = rejected_batch.batch["input_ids"].clone()
        chosen_labels[chosen_batch.batch["attention_mask"] == 0] = -100
        rejected_labels[rejected_batch.batch["attention_mask"] == 0] = -100
        for row_idx, prompt_width in enumerate(prompt_widths.tolist()):
            chosen_labels[row_idx, :prompt_width] = -100
            rejected_labels[row_idx, :prompt_width] = -100

        dpo_tensors = {
            "chosen_input_ids": chosen_batch.batch["input_ids"],
            "chosen_attention_mask": chosen_batch.batch["attention_mask"],
            "chosen_position_ids": chosen_batch.batch["position_ids"],
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_batch.batch["input_ids"],
            "rejected_attention_mask": rejected_batch.batch["attention_mask"],
            "rejected_position_ids": rejected_batch.batch["position_ids"],
            "rejected_labels": rejected_labels,
            "reference_chosen_logps": reference_chosen_logps,
            "reference_rejected_logps": reference_rejected_logps,
        }
        dpo_meta = {
            "dpo_beta": self.config.algorithm.dpo_beta,
            "dpo_loss_type": self.config.algorithm.dpo_loss_type,
            "dpo_label_smoothing": self.config.algorithm.dpo_label_smoothing,
            "use_reference_policy": bool(self.config.algorithm.get("use_reference_policy", True)),
            "reference_free": not bool(self.config.algorithm.get("use_reference_policy", True)),
            "global_step": self.global_steps,
        }
        return DataProto.from_dict(tensors=dpo_tensors, meta_info=dpo_meta)

    def _collect_policy_loss_examples(
        self,
        *,
        target_policy: str,
        raw_prompts: list[Any],
        benchmark_ids: list[str],
        policy_a_responses: list[str],
        policy_b_responses: list[str],
        a_rewards: torch.Tensor,
        b_rewards: torch.Tensor,
    ) -> tuple[list[Any], list[str], list[str], list[str], list[float]]:
        if target_policy == "policy_a":
            loss_mask = b_rewards > a_rewards
            chosen_texts = policy_b_responses
            rejected_texts = policy_a_responses
            margins = (b_rewards - a_rewards).cpu().tolist()
        else:
            loss_mask = a_rewards > b_rewards
            chosen_texts = policy_a_responses
            rejected_texts = policy_b_responses
            margins = (a_rewards - b_rewards).cpu().tolist()

        if not self.config.algorithm.get("focus_failed_examples_only", True):
            loss_mask = torch.ones_like(loss_mask, dtype=torch.bool)

        selected_prompts = []
        selected_benchmarks = []
        selected_chosen = []
        selected_rejected = []
        selected_margins = []
        for idx, is_selected in enumerate(loss_mask.tolist()):
            if not is_selected:
                continue
            selected_prompts.append(raw_prompts[idx])
            selected_benchmarks.append(benchmark_ids[idx])
            selected_chosen.append(chosen_texts[idx])
            selected_rejected.append(rejected_texts[idx])
            selected_margins.append(float(margins[idx]))
        return selected_prompts, selected_benchmarks, selected_chosen, selected_rejected, selected_margins

    def _record_policy_failures(
        self,
        *,
        policy_name: str,
        raw_prompts: list[Any],
        benchmark_ids: list[str],
        chosen_texts: list[str],
        rejected_texts: list[str],
        margins: list[float],
    ) -> None:
        max_per_benchmark = int(self.config.algorithm.get("replay_buffer_size_per_benchmark", 64))
        replay_store = self.policy_failure_replay[policy_name]
        for raw_prompt, benchmark_id, chosen_text, rejected_text, margin in zip(
            raw_prompts, benchmark_ids, chosen_texts, rejected_texts, margins, strict=True
        ):
            replay_store[benchmark_id].append(
                {
                    "raw_prompt": deepcopy(raw_prompt),
                    "benchmark_id": benchmark_id,
                    "chosen_text": chosen_text,
                    "rejected_text": rejected_text,
                    "margin": float(max(margin, 0.0)),
                }
            )
            if len(replay_store[benchmark_id]) > max_per_benchmark:
                replay_store[benchmark_id] = replay_store[benchmark_id][-max_per_benchmark:]

    def _benchmark_priority(self, policy_name: str, benchmark_id: str) -> float:
        state = self.benchmark_stats.get(benchmark_id)
        if state is None or state["count"] <= 0:
            return 1.0
        count = float(max(state["count"], 1))
        if policy_name == "policy_a":
            own_wins = state["policy_a_wins"] / count
            opp_wins = state["policy_b_wins"] / count
            reward_gap = (state["policy_b_reward_sum"] - state["policy_a_reward_sum"]) / count
        else:
            own_wins = state["policy_b_wins"] / count
            opp_wins = state["policy_a_wins"] / count
            reward_gap = (state["policy_a_reward_sum"] - state["policy_b_reward_sum"]) / count
        deficit = max(opp_wins - own_wins, 0.0)
        reward_deficit = max(reward_gap, 0.0)
        return 1.0 + deficit + reward_deficit

    def _sample_benchmark_replay_examples(self, policy_name: str, requested_count: int) -> tuple[list[Any], list[str], list[str], list[str], list[float]]:
        replay_store = self.policy_failure_replay[policy_name]
        available = {benchmark_id: items for benchmark_id, items in replay_store.items() if items}
        if requested_count <= 0 or not available:
            return [], [], [], [], []

        benchmark_ids = list(available.keys())
        priorities = np.asarray(
            [self._benchmark_priority(policy_name, benchmark_id) for benchmark_id in benchmark_ids],
            dtype=np.float64,
        )
        replay_power = float(self.config.algorithm.get("benchmark_replay_power", 1.0))
        priorities = np.power(np.maximum(priorities, 1e-6), replay_power)
        probabilities = priorities / priorities.sum()

        prompts = []
        selected_benchmarks = []
        chosen = []
        rejected = []
        margins = []
        margin_power = float(self.config.algorithm.get("benchmark_margin_power", 1.0))
        for _ in range(requested_count):
            benchmark_idx = int(self.replay_rng.choice(len(benchmark_ids), p=probabilities))
            benchmark_id = benchmark_ids[benchmark_idx]
            candidate_items = available[benchmark_id]
            if len(candidate_items) == 1:
                sample = candidate_items[0]
            else:
                candidate_weights = np.asarray(
                    [1.0 + max(float(item.get("margin", 0.0)), 0.0) for item in candidate_items],
                    dtype=np.float64,
                )
                candidate_weights = np.power(candidate_weights, margin_power)
                candidate_weights = candidate_weights / candidate_weights.sum()
                sample = candidate_items[int(self.replay_rng.choice(len(candidate_items), p=candidate_weights))]
            prompts.append(deepcopy(sample["raw_prompt"]))
            selected_benchmarks.append(sample["benchmark_id"])
            chosen.append(sample["chosen_text"])
            rejected.append(sample["rejected_text"])
            margins.append(float(sample.get("margin", 0.0)))
        return prompts, selected_benchmarks, chosen, rejected, margins

    def _select_update_examples(
        self,
        *,
        policy_name: str,
        fresh_prompts: list[Any],
        fresh_benchmarks: list[str],
        fresh_chosen: list[str],
        fresh_rejected: list[str],
        fresh_margins: list[float],
    ) -> tuple[list[Any], list[str], list[str], list[str], list[float]]:
        self._record_policy_failures(
            policy_name=policy_name,
            raw_prompts=fresh_prompts,
            benchmark_ids=fresh_benchmarks,
            chosen_texts=fresh_chosen,
            rejected_texts=fresh_rejected,
            margins=fresh_margins,
        )

        if self.config.algorithm.get("include_fresh_failures_in_update", True) and fresh_prompts:
            return fresh_prompts, fresh_benchmarks, fresh_chosen, fresh_rejected, fresh_margins

        requested_count = len(fresh_prompts)
        if requested_count == 0 and self.config.algorithm.get("allow_buffer_only_updates", True):
            requested_count = int(self.config.algorithm.get("buffer_only_update_batch_size", self.config.data.train_batch_size))
        return self._sample_benchmark_replay_examples(policy_name, requested_count)

    def _update_benchmark_stats(self, benchmark_ids: list[str], a_rewards: torch.Tensor, b_rewards: torch.Tensor) -> dict[str, float]:
        metrics = {}
        a_rewards_np = a_rewards.cpu().numpy()
        b_rewards_np = b_rewards.cpu().numpy()
        grouped = defaultdict(list)
        for idx, benchmark_id in enumerate(benchmark_ids):
            grouped[benchmark_id].append(idx)

        for benchmark_id, indices in grouped.items():
            a_subset = a_rewards_np[indices]
            b_subset = b_rewards_np[indices]
            wins_a = int((a_subset > b_subset).sum())
            wins_b = int((b_subset > a_subset).sum())
            ties = int((a_subset == b_subset).sum())
            state = self.benchmark_stats[benchmark_id]
            state["count"] += len(indices)
            state["policy_a_wins"] += wins_a
            state["policy_b_wins"] += wins_b
            state["ties"] += ties
            state["policy_a_reward_sum"] += float(a_subset.sum())
            state["policy_b_reward_sum"] += float(b_subset.sum())

            metric_key = _sanitize_metric_key(benchmark_id)
            metrics[f"bench/{metric_key}/policy_a_reward_mean"] = float(a_subset.mean())
            metrics[f"bench/{metric_key}/policy_b_reward_mean"] = float(b_subset.mean())
            metrics[f"bench/{metric_key}/policy_a_win_rate"] = float(wins_a / len(indices))
            metrics[f"bench/{metric_key}/policy_b_win_rate"] = float(wins_b / len(indices))
            metrics[f"bench/{metric_key}/tie_rate"] = float(ties / len(indices))
            metrics[f"bench/{metric_key}/reward_margin_mean"] = float((a_subset - b_subset).mean())
            metrics[f"bench/{metric_key}/count"] = float(state["count"])
        return metrics

    def _evaluate_duel(self, max_samples: int | None = None) -> dict[str, float]:
        example_index = 0
        policy_a_scores_all = []
        policy_b_scores_all = []
        benchmark_scores = defaultdict(lambda: {"a": [], "b": []})

        for batch_idx, test_data in enumerate(self.val_dataloader):
            if max_samples is not None and example_index >= max_samples:
                break

            batch = DataProto.from_single_dict(test_data)
            raw_prompts = self._extract_raw_prompts(batch)
            benchmark_ids = self._extract_benchmark_ids(batch)
            reward_model_entries = self._extract_reward_entries(batch, "reward_model")
            extra_info_entries = self._extract_reward_entries(batch, "extra_info")

            policy_a_output = self._generate_with_policy(self.policy_a_wg, self._build_policy_generation_batch(batch, "policy_a", validate=True))
            policy_b_output = self._generate_with_policy(self.policy_b_wg, self._build_policy_generation_batch(batch, "policy_b", validate=True))
            policy_a_texts = self._decode_responses(self.policy_tokenizers["policy_a"], policy_a_output)
            policy_b_texts = self._decode_responses(self.policy_tokenizers["policy_b"], policy_b_output)

            a_scores, _ = self._score_response_texts(
                raw_prompts=raw_prompts,
                responses=policy_a_texts,
                benchmark_ids=benchmark_ids,
                reward_model_entries=reward_model_entries,
                extra_info_entries=extra_info_entries,
            )
            b_scores, _ = self._score_response_texts(
                raw_prompts=raw_prompts,
                responses=policy_b_texts,
                benchmark_ids=benchmark_ids,
                reward_model_entries=reward_model_entries,
                extra_info_entries=extra_info_entries,
            )

            if max_samples is not None:
                keep = min(a_scores.shape[0], max_samples - example_index)
                a_scores = a_scores[:keep]
                b_scores = b_scores[:keep]
                benchmark_ids = benchmark_ids[:keep]

            policy_a_scores_all.extend(a_scores.cpu().tolist())
            policy_b_scores_all.extend(b_scores.cpu().tolist())
            for benchmark_id, a_score, b_score in zip(benchmark_ids, a_scores.cpu().tolist(), b_scores.cpu().tolist(), strict=True):
                benchmark_scores[benchmark_id]["a"].append(a_score)
                benchmark_scores[benchmark_id]["b"].append(b_score)
            example_index += len(benchmark_ids)
            pprint(f"[crossplay val] processed {example_index} samples after batch {batch_idx}")

        a_np = np.asarray(policy_a_scores_all, dtype=np.float32)
        b_np = np.asarray(policy_b_scores_all, dtype=np.float32)
        metrics = {
            "val/policy_a_reward_mean": float(a_np.mean()) if a_np.size else 0.0,
            "val/policy_b_reward_mean": float(b_np.mean()) if b_np.size else 0.0,
            "val/policy_a_win_rate": float((a_np > b_np).mean()) if a_np.size else 0.0,
            "val/policy_b_win_rate": float((b_np > a_np).mean()) if a_np.size else 0.0,
            "val/tie_rate": float((a_np == b_np).mean()) if a_np.size else 0.0,
        }
        for benchmark_id, values in benchmark_scores.items():
            a_arr = np.asarray(values["a"], dtype=np.float32)
            b_arr = np.asarray(values["b"], dtype=np.float32)
            metric_key = _sanitize_metric_key(benchmark_id)
            metrics[f"val/bench/{metric_key}/policy_a_win_rate"] = float((a_arr > b_arr).mean()) if a_arr.size else 0.0
            metrics[f"val/bench/{metric_key}/policy_b_win_rate"] = float((b_arr > a_arr).mean()) if b_arr.size else 0.0
            metrics[f"val/bench/{metric_key}/tie_rate"] = float((a_arr == b_arr).mean()) if a_arr.size else 0.0
        return metrics

    def fit_crossplay(self):
        if self.config.policy_a.rollout.n != 1 or self.config.policy_b.rollout.n != 1:
            raise ValueError("CrossPlay MVP expects policy_a.rollout.n=1 and policy_b.rollout.n=1")

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True, throw_on_missing=False),
        )

        loaded_step = self._load_checkpoint()
        self.global_steps = loaded_step + 1 if loaded_step is not None and loaded_step > 0 else 1

        if self.config.trainer.get("val_before_train", False):
            val_metrics = self._evaluate_duel(max_samples=self.config.evaluation.get("max_samples"))
            logger.log(data=val_metrics, step=max(0, self.global_steps - 1))
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps - 1, desc="CrossPlay Training Progress")

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if self.global_steps > self.total_training_steps:
                    progress_bar.close()
                    return

                batch = DataProto.from_single_dict(batch_dict)
                raw_prompts = self._extract_raw_prompts(batch)
                benchmark_ids = self._extract_benchmark_ids(batch)
                reward_model_entries = self._extract_reward_entries(batch, "reward_model")
                extra_info_entries = self._extract_reward_entries(batch, "extra_info")

                metrics = {}
                timing_raw = {}
                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    with _timer("gen_policy_a", timing_raw):
                        policy_a_output = self._generate_with_policy(
                            self.policy_a_wg,
                            self._build_policy_generation_batch(batch, "policy_a", validate=False),
                        )
                    with _timer("gen_policy_b", timing_raw):
                        policy_b_output = self._generate_with_policy(
                            self.policy_b_wg,
                            self._build_policy_generation_batch(batch, "policy_b", validate=False),
                        )

                    policy_a_texts = self._decode_responses(self.policy_tokenizers["policy_a"], policy_a_output)
                    policy_b_texts = self._decode_responses(self.policy_tokenizers["policy_b"], policy_b_output)

                    with _timer("judge_policy_a", timing_raw):
                        a_rewards, a_extras = self._score_response_texts(
                            raw_prompts=raw_prompts,
                            responses=policy_a_texts,
                            benchmark_ids=benchmark_ids,
                            reward_model_entries=reward_model_entries,
                            extra_info_entries=extra_info_entries,
                        )
                    with _timer("judge_policy_b", timing_raw):
                        b_rewards, b_extras = self._score_response_texts(
                            raw_prompts=raw_prompts,
                            responses=policy_b_texts,
                            benchmark_ids=benchmark_ids,
                            reward_model_entries=reward_model_entries,
                            extra_info_entries=extra_info_entries,
                        )

                    metrics["game/policy_a_reward_mean"] = a_rewards.mean().item()
                    metrics["game/policy_b_reward_mean"] = b_rewards.mean().item()
                    metrics["game/policy_a_win_rate"] = (a_rewards > b_rewards).float().mean().item()
                    metrics["game/policy_b_win_rate"] = (b_rewards > a_rewards).float().mean().item()
                    metrics["game/tie_rate"] = (a_rewards == b_rewards).float().mean().item()
                    metrics["game/reward_margin_mean"] = (a_rewards - b_rewards).mean().item()
                    metrics.update(self._update_benchmark_stats(benchmark_ids, a_rewards, b_rewards))

                    if "exact_match" in a_extras:
                        metrics["game/policy_a_exact_match_mean"] = float(np.mean(a_extras["exact_match"]))
                    if "exact_match" in b_extras:
                        metrics["game/policy_b_exact_match_mean"] = float(np.mean(b_extras["exact_match"]))

                    for target_policy in self._update_targets_for_step():
                        fresh_prompts, fresh_benchmarks, fresh_chosen, fresh_rejected, fresh_margins = self._collect_policy_loss_examples(
                            target_policy=target_policy,
                            raw_prompts=raw_prompts,
                            benchmark_ids=benchmark_ids,
                            policy_a_responses=policy_a_texts,
                            policy_b_responses=policy_b_texts,
                            a_rewards=a_rewards,
                            b_rewards=b_rewards,
                        )

                        metrics[f"{target_policy}/fresh_failed_examples"] = float(len(fresh_prompts))
                        if fresh_margins:
                            metrics[f"{target_policy}/fresh_reward_margin_mean"] = float(np.mean(fresh_margins))
                        selected_prompts, selected_benchmarks, selected_chosen, selected_rejected, selected_margins = self._select_update_examples(
                            policy_name=target_policy,
                            fresh_prompts=fresh_prompts,
                            fresh_benchmarks=fresh_benchmarks,
                            fresh_chosen=fresh_chosen,
                            fresh_rejected=fresh_rejected,
                            fresh_margins=fresh_margins,
                        )
                        metrics[f"{target_policy}/selected_examples"] = float(len(selected_prompts))
                        if selected_margins:
                            metrics[f"{target_policy}/selected_reward_margin_mean"] = float(np.mean(selected_margins))
                        metrics[f"{target_policy}/replay_buffer_examples"] = float(
                            sum(len(items) for items in self.policy_failure_replay[target_policy].values())
                        )
                        if selected_benchmarks:
                            bench_counts = defaultdict(int)
                            for benchmark_id in selected_benchmarks:
                                bench_counts[_sanitize_metric_key(benchmark_id)] += 1
                            for benchmark_id, count in bench_counts.items():
                                metrics[f"{target_policy}/bench/{benchmark_id}/selected_examples"] = float(count)

                        if not selected_prompts:
                            metrics[f"{target_policy}/update_skipped"] = 1.0
                            continue

                        update_batch = self._build_policy_update_batch(
                            policy_name=target_policy,
                            raw_prompts=selected_prompts,
                            chosen_texts=selected_chosen,
                            rejected_texts=selected_rejected,
                        )
                        if update_batch is None:
                            metrics[f"{target_policy}/update_skipped"] = 1.0
                            continue

                        with _timer(f"update_{target_policy}", timing_raw):
                            if target_policy == "policy_a":
                                output = self.policy_a_wg.update_actor_dpo(update_batch)
                            else:
                                output = self.policy_b_wg.update_actor_dpo(update_batch)
                        metrics.update(self._prefix_metrics(reduce_metrics(output.meta_info["metrics"]), target_policy))

                if self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    with _timer("testing", timing_raw):
                        metrics.update(self._evaluate_duel(max_samples=self.config.evaluation.get("max_samples")))

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
