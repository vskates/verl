# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging
import os

import psutil
import torch
from codetiming import Timer
from omegaconf import open_dict

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, load_fsdp_optimizer, offload_fsdp_model_to_cpu, offload_fsdp_optimizer
from verl.utils.import_utils import import_external_libs
from verl.workers.fsdp_workers import ActorRolloutRefWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))


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

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from .dp_actor import RefPlayDataParallelPPOActor

        import_external_libs(self.config.model.get("external_lib", None))

        from omegaconf import OmegaConf

        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        use_remove_padding = self.config.model.get("use_remove_padding", False)

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
                output = DataProto.from_dict(tensors={"ref_log_prob": output})
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
