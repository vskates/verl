# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import os

import hydra
import ray

from verl.trainer.ppo.reward import load_reward_manager

from .refplay_ray_trainer import RayRefPlayTrainer


@hydra.main(config_path="config", config_name="refplay_trainer", version_base=None)
def main(config):
    run_refplay(config)


def run_refplay(config) -> None:
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not ray.is_initialized():
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from recipe.spin.spin_trainer import ResourcePoolManager, Role
        from verl.single_controller.ray import RayWorkerGroup
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local

        from .refplay_worker import RefPlayActorRolloutRefWorker

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_local_path = copy_to_local(config.actor_rollout_ref.model.path)
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(actor_local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(actor_local_path, use_fast=True)

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(RefPlayActorRolloutRefWorker),
            Role.Rollout: ray.remote(RefPlayActorRolloutRefWorker),
            Role.RefPolicy: ray.remote(RefPlayActorRolloutRefWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Rollout: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy == "fsdp":
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1)
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = RayRefPlayTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit_dpo()


if __name__ == "__main__":
    main()
