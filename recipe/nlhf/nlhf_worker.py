from __future__ import annotations

import os
from collections import OrderedDict

import torch

from recipe.refplay.refplay_worker import RefPlayActorRolloutRefWorker
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_torch_device
from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, load_fsdp_optimizer, offload_fsdp_model_to_cpu, offload_fsdp_optimizer


class NLHFActorRolloutRefWorker(RefPlayActorRolloutRefWorker):
    """Worker for paper-faithful NLHF with one trainable policy, one EMA alt policy, and one frozen reference."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from .dp_actor import NLHFDataParallelPPOActor

        super().init_model()

        if self._is_actor:
            self.actor = NLHFDataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )

        if self._is_rollout:
            self.ref_policy = NLHFDataParallelPPOActor(
                config=self.config.ref,
                actor_module=self.actor_module_fsdp,
            )

        if self._is_ref:
            self.ref_policy = NLHFDataParallelPPOActor(
                config=self.config.ref,
                actor_module=self.ref_module_fsdp,
            )

    def _get_load_target_module(self):
        if self._is_ref and hasattr(self, "ref_module_fsdp"):
            return self.ref_module_fsdp
        if hasattr(self, "actor_module_fsdp"):
            return self.actor_module_fsdp
        raise RuntimeError("No FSDP module available for NLHF weight loading")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_model_weights_only(self, local_path, del_local_after_load=True):
        module = self._get_load_target_module()
        if self._is_offload_param or not self._is_actor:
            load_fsdp_model_to_gpu(module)

        model_path = os.path.join(local_path, "model_world_size_1_rank_0.pt")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], (dict, OrderedDict)):
            state_dict = state_dict["state_dict"]
        if isinstance(state_dict, dict) and not isinstance(state_dict, OrderedDict):
            state_dict = OrderedDict(state_dict)
        if not isinstance(state_dict, OrderedDict):
            raise TypeError(f"Expected state dict-like payload in {model_path}, got {type(state_dict)}")

        module.load_state_dict(state_dict, strict=True)
        torch.distributed.barrier()

        if self._is_offload_param or not self._is_actor:
            offload_fsdp_model_to_cpu(module)

        if del_local_after_load:
            pass

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor_nlhf(self, data: DataProto):
        data = data.to(torch.cuda.current_device())

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())

        log_gpu_memory_usage("Before NLHF actor update", logger=None)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            metrics = self.actor.update_policy_nlhf(data=data)
            metrics["perf/max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = torch.cuda.max_memory_reserved() / (1024**3)
            self.actor_lr_scheduler.step()
            metrics["actor/lr"] = self.actor_lr_scheduler.get_last_lr()[0]
            output = DataProto(meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)

        get_torch_device().empty_cache()
        log_gpu_memory_usage("After NLHF actor update", logger=None)
        return output
