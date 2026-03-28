import os

import hydra
import ray

from recipe.refplay.main_refplay import get_custom_reward_fn
from .crossplay_ray_trainer import RayCrossPlayTrainer


@hydra.main(config_path="config", config_name="crossplay_trainer", version_base=None)
def main(config):
    run_crossplay(config)


def run_crossplay(config) -> None:
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not ray.is_initialized():
        ray_init_kwargs = {
            "runtime_env": {
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                    "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID"),
                    "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
                }
            }
        }
        ray_init_cfg = config.get("ray_init")
        if ray_init_cfg is not None and ray_init_cfg.get("num_cpus") is not None:
            ray_init_kwargs["num_cpus"] = ray_init_cfg.num_cpus
        ray.init(**ray_init_kwargs)

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

        from .crossplay_worker import CrossPlayActorRolloutRefWorker
        from .text_judges import CallableTextJudge, RuleBasedGSM8KTextJudge

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        policy_a_local_path = copy_to_local(config.policy_a.model.path)
        policy_b_local_path = copy_to_local(config.policy_b.model.path)
        policy_a_tokenizer = hf_tokenizer(policy_a_local_path, trust_remote_code=config.policy_a.model.get("trust_remote_code", False))
        policy_b_tokenizer = hf_tokenizer(policy_b_local_path, trust_remote_code=config.policy_b.model.get("trust_remote_code", False))
        processor = hf_processor(policy_a_local_path, use_fast=True)

        use_reference_policy = bool(config.algorithm.get("use_reference_policy", True))

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(CrossPlayActorRolloutRefWorker),
            Role.Actor: ray.remote(CrossPlayActorRolloutRefWorker),
        }
        if use_reference_policy:
            role_worker_mapping.update(
                {
                    Role.Rollout: ray.remote(CrossPlayActorRolloutRefWorker),
                    Role.RefPolicy: ray.remote(CrossPlayActorRolloutRefWorker),
                }
            )

        resource_pool_spec = {
            "policy_a_pool": [1] * config.trainer.nnodes,
            "policy_b_pool": [1] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: "policy_a_pool",
            Role.Actor: "policy_b_pool",
        }
        if use_reference_policy:
            resource_pool_spec.update(
                {
                    "anchor_a_pool": [1] * config.trainer.nnodes,
                    "anchor_b_pool": [1] * config.trainer.nnodes,
                }
            )
            mapping.update(
                {
                    Role.Rollout: "anchor_a_pool",
                    Role.RefPolicy: "anchor_b_pool",
                }
            )

        reward_manager_name = None
        reward_cfg = config.get("reward")
        if reward_cfg is not None:
            reward_manager_cfg = reward_cfg.get("reward_manager")
            if isinstance(reward_manager_cfg, str):
                reward_manager_name = reward_manager_cfg
            elif reward_manager_cfg is not None and reward_manager_cfg.get("name") is not None:
                reward_manager_name = reward_manager_cfg.name

        reward_manager_name = reward_manager_name or "crossplay_gsm8k_rule"
        if reward_manager_name in {"crossplay_gsm8k_rule", "refplay_gsm8k_rule"}:
            judge = RuleBasedGSM8KTextJudge()
        else:
            custom_fn = get_custom_reward_fn(config)
            if custom_fn is None:
                raise NotImplementedError(
                    f"Unsupported crossplay reward manager {reward_manager_name!r} without custom_reward_function.path"
                )
            judge = CallableTextJudge(custom_fn)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        trainer = RayCrossPlayTrainer(
            config=config,
            tokenizer=policy_a_tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            policy_tokenizers={
                "policy_a": policy_a_tokenizer,
                "policy_b": policy_b_tokenizer,
                "anchor_a": policy_a_tokenizer,
                "anchor_b": policy_b_tokenizer,
            },
            text_judge=judge,
        )
        trainer.init_workers()
        trainer.fit_crossplay()


if __name__ == "__main__":
    main()
