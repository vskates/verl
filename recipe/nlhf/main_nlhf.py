import os

import hydra
import ray

from recipe.refplay.main_refplay import get_custom_reward_fn
from .nlhf_ray_trainer import RayNLHFTrainer


@hydra.main(config_path="config", config_name="nlhf_trainer", version_base=None)
def main(config):
    run_nlhf(config)


def run_nlhf(config) -> None:
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not ray.is_initialized():
        ray_init_kwargs = {
            "include_dashboard": False,
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
        from omegaconf import OmegaConf

        from recipe.spin.spin_trainer import ResourcePoolManager, Role
        from verl.single_controller.ray import RayWorkerGroup
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local

        from .nlhf_worker import NLHFActorRolloutRefWorker
        from .pairwise_judges import CallablePairwiseJudge, RuleBasedGSM8KPairwiseJudge

        print(
            f"[nlhf] experiment={config.trainer.experiment_name} "
            f"model={config.main_policy.model.path} "
            f"steps={config.trainer.total_training_steps} "
            f"batch={config.data.train_batch_size}"
        )
        OmegaConf.resolve(config)

        policy_local_path = copy_to_local(config.main_policy.model.path)
        trust_remote_code = config.main_policy.model.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(policy_local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(policy_local_path, use_fast=True)

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(NLHFActorRolloutRefWorker),
            Role.Rollout: ray.remote(NLHFActorRolloutRefWorker),
            Role.RefPolicy: ray.remote(NLHFActorRolloutRefWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Rollout: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }

        reward_manager_name = None
        reward_cfg = config.get("reward")
        if reward_cfg is not None:
            reward_manager_cfg = reward_cfg.get("reward_manager")
            if isinstance(reward_manager_cfg, str):
                reward_manager_name = reward_manager_cfg
            elif reward_manager_cfg is not None and reward_manager_cfg.get("name") is not None:
                reward_manager_name = reward_manager_cfg.name

        reward_manager_name = reward_manager_name or "nlhf_gsm8k_pairwise"
        if reward_manager_name in {"nlhf_gsm8k_pairwise", "refplay_gsm8k_rule"}:
            judge = RuleBasedGSM8KPairwiseJudge(
                mode=config.algorithm.get("preference_source", "pairwise_probability"),
                reward_temperature=float(config.algorithm.get("reward_temperature", 1.0)),
            )
        else:
            custom_fn = get_custom_reward_fn(config)
            if custom_fn is None:
                raise NotImplementedError(
                    f"Unsupported NLHF reward manager {reward_manager_name!r} without custom_reward_function.path"
                )
            judge = CallablePairwiseJudge(custom_fn)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        trainer = RayNLHFTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=None,
            val_reward_fn=None,
            pairwise_judge=judge,
        )
        trainer.init_workers()
        trainer.fit_nlhf()


if __name__ == "__main__":
    main()
