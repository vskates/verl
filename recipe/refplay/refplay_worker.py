# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import contextlib
import logging
import os
from collections import OrderedDict

import psutil
import torch
from codetiming import Timer
from omegaconf import open_dict
from tensordict import TensorDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, load_fsdp_optimizer, offload_fsdp_model_to_cpu, offload_fsdp_optimizer
from verl.utils.import_utils import import_external_libs
from verl.utils.torch_functional import get_response_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))


class _LegacyHFRollout:
    """Compatibility rollout for recipes that expect local HF generation."""

    def __init__(self, module, config):
        self.module = module
        self.config = config

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        micro_batch_size = self.config.get("micro_batch_size", batch_size) or batch_size
        num_chunks = max(batch_size // micro_batch_size, 1)
        outputs = [self._generate_minibatch(chunk) for chunk in prompts.chunk(chunks=num_chunks)]
        return DataProto.concat(outputs)

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        is_validate = prompts.meta_info.get("validate", False)

        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))

        if not do_sample:
            kwargs = {"do_sample": False, "num_beams": 1}
        elif is_validate:
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_k": max(0, self.config.val_kwargs.top_k),
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "num_return_sequences": 1,
            }
        else:
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "num_return_sequences": 1,
            }

        generation_config = GenerationConfig(**kwargs)
        idx = prompts.batch["input_ids"]
        prompt_length = idx.size(1)
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]

        self.module.eval()
        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)

        with param_ctx, torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            output = self.module.generate(
                input_ids=idx,
                attention_mask=attention_mask,
                position_ids=position_ids,
                do_sample=do_sample,
                max_new_tokens=response_length,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                generation_config=generation_config,
                output_scores=False,
                return_dict_in_generate=True,
                use_cache=True,
            )

        seq = output.sequences
        generated_batch_size = seq.size(0)
        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]
        if delta_length > 0:
            delta_tokens = torch.full(
                (generated_batch_size, delta_length),
                pad_token_id,
                device=seq.device,
                dtype=seq.dtype,
            )
            seq = torch.cat((seq, delta_tokens), dim=1)

        prompt = seq[:, :prompt_length]
        response = seq[:, prompt_length:]
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response,
            eos_token=eos_token_id,
            dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=generated_batch_size,
        )
        get_torch_device().empty_cache()
        self.module.train()
        return DataProto(batch=batch)


class RefPlayActorRolloutRefWorker(ActorRolloutRefWorker):
    def _get_rollout_fsdp_module(self):
        if hasattr(self, "actor_module_fsdp"):
            return self.actor_module_fsdp
        if hasattr(self, "ref_module_fsdp"):
            return self.ref_module_fsdp
        return None

    def _fsdp_module_on_cpu(self, module) -> bool:
        try:
            param = next(module.parameters())
        except StopIteration:
            return False
        return param.device.type == "cpu"

    def _build_rollout(self, trust_remote_code=False):
        if self.config.rollout.name != "hf":
            result = super()._build_rollout(trust_remote_code=trust_remote_code)
            if isinstance(result, tuple):
                return result
            return self.rollout, getattr(self, "rollout_sharding_manager", self.ulysses_sharding_manager)

        self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)
        self.rollout_device_mesh = self.device_mesh
        self.rollout = _LegacyHFRollout(module=self.actor_module_fsdp, config=self.config.rollout)
        self.base_sync_done = True
        self.layered_summon = self.config.rollout.get("layered_summon", False)
        self.rollout_sharding_manager = self.ulysses_sharding_manager
        return self.rollout, self.rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from .dp_actor import RefPlayDataParallelPPOActor

        import_external_libs(self.config.model.get("external_lib", None))

        from omegaconf import OmegaConf

        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        use_remove_padding = self.config.model.get("use_remove_padding", False)

        # Newer verl builds expect QAT state to exist before _build_model_optimizer()
        # is called, but older refplay recipes never initialized it explicitly.
        if not hasattr(self, "_qat_enabled"):
            init_qat_config = getattr(self, "_init_qat_config", None)
            if callable(init_qat_config):
                init_qat_config()
            else:
                self._qat_enabled = False
                self.qat_config = None

        if self._is_actor or self._is_rollout:
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                # HF rollout still uses the local FSDP model for generation, so a
                # pure rollout worker needs the same sharding policy as the actor.
                if self.config.rollout.name == "hf":
                    fsdp_config = self.config.actor.fsdp_config
                else:
                    fsdp_config = OmegaConf.create()
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
            )

            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
            self.actor = RefPlayDataParallelPPOActor(config=self.config.actor, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer)

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))
            # Reuse the frozen rollout-side model for ref log-probs to avoid
            # keeping a third model copy resident just for DPO reference scores.
            if not self._is_ref:
                OmegaConf.set_struct(self.config.ref, True)
                with open_dict(self.config.ref):
                    self.config.ref.use_remove_padding = use_remove_padding
                self.ref_policy = RefPlayDataParallelPPOActor(config=self.config.ref, actor_module=self.actor_module_fsdp)

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
            self.ref_policy = RefPlayDataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_contents=self.config.actor.checkpoint.contents,
            )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref or self._is_rollout

        ref_module = self._get_rollout_fsdp_module()
        loaded_for_ref = ref_module is not None and self._fsdp_module_on_cpu(ref_module)
        if loaded_for_ref:
            load_fsdp_model_to_gpu(ref_module)

        try:
            data = data.to(torch.cuda.current_device())
            data.meta_info["micro_batch_size"] = self.config.ref.log_prob_micro_batch_size_per_gpu
            data.meta_info["temperature"] = self.config.rollout.temperature
            data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
            data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz

            with self.ulysses_sharding_manager:
                data = self.ulysses_sharding_manager.preprocess_data(data)
                output = self.ref_policy.compute_log_prob(data=data)
                if isinstance(output, dict):
                    ref_log_prob = output.get("log_probs")
                    if ref_log_prob is None:
                        raise KeyError("Expected ref_policy.compute_log_prob() to return 'log_probs'")
                else:
                    ref_log_prob = output
                output = DataProto.from_dict(tensors={"ref_log_prob": ref_log_prob})
                output = self.ulysses_sharding_manager.postprocess_data(output)

            output = output.to("cpu")

            if self.world_size > 1:
                self.ref_policy.actor_module._handle.reshard(True)

            return output
        finally:
            if loaded_for_ref and (self._is_offload_param or not self._is_actor):
                offload_fsdp_model_to_cpu(ref_module)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        rollout_module = self._get_rollout_fsdp_module()
        loaded_for_rollout = rollout_module is not None and self._fsdp_module_on_cpu(rollout_module)
        if loaded_for_rollout:
            load_fsdp_model_to_gpu(rollout_module)

        try:
            eos_token_id = prompts.meta_info.get("eos_token_id")
            if eos_token_id is None:
                eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

            pad_token_id = prompts.meta_info.get("pad_token_id")
            if pad_token_id is None:
                pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                generation_config = getattr(self, "generation_config", None)
                pad_token_id = getattr(generation_config, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = eos_token_id

            if prompts.meta_info.get("eos_token_id") is None and eos_token_id is not None:
                prompts.meta_info["eos_token_id"] = eos_token_id
            if prompts.meta_info.get("pad_token_id") is None and pad_token_id is not None:
                prompts.meta_info["pad_token_id"] = pad_token_id

            # Base ActorRolloutRefWorker overwrites prompt meta_info from
            # self.generation_config/tokenizer, so keep the worker-level source
            # consistent too.
            generation_config = getattr(self, "generation_config", None)
            if generation_config is not None:
                if getattr(generation_config, "eos_token_id", None) is None and eos_token_id is not None:
                    generation_config.eos_token_id = eos_token_id
                if getattr(generation_config, "pad_token_id", None) is None and pad_token_id is not None:
                    generation_config.pad_token_id = pad_token_id
            if getattr(self.tokenizer, "pad_token_id", None) is None and pad_token_id is not None:
                self.tokenizer.pad_token_id = pad_token_id

            if self.config.rollout.name == "hf":
                prompts = prompts.to(torch.cuda.current_device())
                prompts.meta_info.update({"eos_token_id": eos_token_id, "pad_token_id": pad_token_id})
                output = self.rollout.generate_sequences(prompts=prompts)
                output = output.to("cpu")
                get_torch_device().empty_cache()
                return output

            return super().generate_sequences(prompts)
        finally:
            if loaded_for_rollout and (self._is_offload_param or not self._is_actor):
                offload_fsdp_model_to_cpu(rollout_module)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor_dpo(self, data: DataProto):
        data = data.to(torch.cuda.current_device())

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())

        log_gpu_memory_usage("Before refplay actor update", logger=logger)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_policy_dpo_via_ppo", logger=None) as timer:
                metrics = self.actor.update_policy_dpo_with_ref(data=data)
            metrics["perf/max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = torch.cuda.max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)
            metrics["perf/update_actor_seconds"] = timer.last
            self.actor_lr_scheduler.step()
            metrics["actor/lr"] = self.actor_lr_scheduler.get_last_lr()[0]
            output = DataProto(meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)

        log_gpu_memory_usage("After refplay actor update", logger=logger)
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_model_checkpoint_only(self, local_path, del_local_after_load=True):
        import torch

        assert self._is_actor

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        model_path = os.path.join(local_path, "model_world_size_1_rank_0.pt")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        if not isinstance(state_dict, OrderedDict):
            raise TypeError(f"Expected OrderedDict state dict in {model_path}, got {type(state_dict)}")

        self.actor_module_fsdp.load_state_dict(state_dict, strict=True)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if del_local_after_load:
            # Keep parity with checkpoint_manager.load_checkpoint signature,
            # but never delete the source checkpoint during resume.
            pass
