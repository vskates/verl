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

        from recipe.refplay.gsm8k_dense_reward import RuleBasedGSM8KReward
        from recipe.spin.spin_trainer import ResourcePoolManager, Role
        from verl.single_controller.ray import RayWorkerGroup
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local

        from .nlhf_worker import NLHFActorRolloutRefWorker

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_local_path = copy_to_local(config.policy_a.model.path)
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(actor_local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(actor_local_path, use_fast=True)

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(NLHFActorRolloutRefWorker),
            Role.Actor: ray.remote(NLHFActorRolloutRefWorker),
            Role.Rollout: ray.remote(NLHFActorRolloutRefWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Actor: global_pool_id,
            Role.Rollout: global_pool_id,
        }

        reward_manager_name = None
        reward_model_cfg = config.get("reward_model")
        if reward_model_cfg is not None:
            reward_manager_cfg = reward_model_cfg.get("reward_manager")
            if isinstance(reward_manager_cfg, str):
                reward_manager_name = reward_manager_cfg
            elif reward_manager_cfg is not None and reward_manager_cfg.get("name") is not None:
                reward_manager_name = reward_manager_cfg.name

        reward_cfg = config.get("reward")
        if reward_manager_name is None and reward_cfg is not None:
            reward_manager_cfg = reward_cfg.get("reward_manager")
            if isinstance(reward_manager_cfg, str):
                reward_manager_name = reward_manager_cfg
            elif reward_manager_cfg is not None and reward_manager_cfg.get("name") is not None:
                reward_manager_name = reward_manager_cfg.name

        reward_manager_name = reward_manager_name or "naive"
        if reward_manager_name == "refplay_gsm8k_rule":
            reward_fn = RuleBasedGSM8KReward(tokenizer=tokenizer, num_examine=0)
            val_reward_fn = RuleBasedGSM8KReward(tokenizer=tokenizer, num_examine=1)
        else:
            compute_score = get_custom_reward_fn(config)
            if reward_manager_name == "naive":
                from verl.workers.reward_manager import NaiveRewardManager

                reward_manager_cls = NaiveRewardManager
            elif reward_manager_name == "prime":
                from verl.workers.reward_manager import PrimeRewardManager

                reward_manager_cls = PrimeRewardManager
            elif reward_manager_name == "batch":
                from verl.workers.reward_manager import BatchRewardManager

                reward_manager_cls = BatchRewardManager
            elif reward_manager_name == "dapo":
                from verl.workers.reward_manager import DAPORewardManager

                reward_manager_cls = DAPORewardManager
            else:
                raise NotImplementedError(f"Unsupported reward manager: {reward_manager_name}")

            reward_kwargs = {}
            if reward_model_cfg is not None:
                reward_kwargs.update(dict(reward_model_cfg.get("reward_kwargs", {})))
            reward_fn = reward_manager_cls(
                tokenizer=tokenizer,
                num_examine=0,
                compute_score=compute_score,
                reward_fn_key=config.data.reward_fn_key,
                **reward_kwargs,
            )
            val_reward_fn = reward_manager_cls(
                tokenizer=tokenizer,
                num_examine=1,
                compute_score=compute_score,
                reward_fn_key=config.data.reward_fn_key,
            )

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        trainer = RayNLHFTrainer(
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
        trainer.fit_nlhf()


if __name__ == "__main__":
    main()

