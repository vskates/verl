import os
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf, open_dict

from .main_refplay import get_custom_reward_fn
from .refplay_ray_trainer import RayRefPlayTrainer


@hydra.main(config_path="config", config_name="refplay_trainer", version_base=None)
def main(config):
    run_refplay_eval(config)


def run_refplay_eval(config) -> None:
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

    runner = EvalTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class EvalTaskRunner:
    def run(self, config):
        from recipe.spin.spin_trainer import ResourcePoolManager, Role
        from verl.single_controller.ray import RayWorkerGroup
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local

        from .refplay_worker import RefPlayActorRolloutRefWorker

        evaluation_cfg = config.get("evaluation")
        if evaluation_cfg is None or evaluation_cfg.get("checkpoint_path") is None:
            raise ValueError("Set evaluation.checkpoint_path to the target global_step checkpoint folder")

        with open_dict(config):
            if evaluation_cfg.get("val_batch_size") is not None:
                config.data.val_batch_size = evaluation_cfg.val_batch_size
            if evaluation_cfg.get("do_sample") is not None:
                config.actor_rollout_ref.rollout.val_kwargs.do_sample = evaluation_cfg.do_sample
            if evaluation_cfg.get("n") is not None:
                config.actor_rollout_ref.rollout.val_kwargs.n = evaluation_cfg.n

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_local_path = copy_to_local(config.actor_rollout_ref.model.path)
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(actor_local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(actor_local_path, use_fast=True)

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(RefPlayActorRolloutRefWorker),
            Role.Rollout: ray.remote(RefPlayActorRolloutRefWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Rollout: global_pool_id,
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
            raise NotImplementedError

        compute_score = get_custom_reward_fn(config)
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

        output_dir = evaluation_cfg.get("output_dir")
        if output_dir is None:
            checkpoint_name = os.path.basename(str(evaluation_cfg.checkpoint_path).rstrip("/"))
            output_dir = os.path.join("outputs", "refplay_eval", checkpoint_name)
        summary = trainer.evaluate_against_reference(
            checkpoint_path=evaluation_cfg.checkpoint_path,
            output_dir=output_dir,
            preview_count=evaluation_cfg.get("preview_samples", 5),
            max_samples=evaluation_cfg.get("max_samples"),
        )
        pprint(summary)


if __name__ == "__main__":
    main()
