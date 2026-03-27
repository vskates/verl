import json
import os
import re
from pathlib import Path
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf, open_dict

from recipe.refplay.main_refplay import get_custom_reward_fn
from recipe.refplay.refplay_ray_trainer import RayRefPlayTrainer


def _sorted_checkpoint_paths(checkpoint_root: str) -> list[str]:
    root = Path(checkpoint_root)
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint root does not exist: {root}")

    candidates = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("global_step_")]

    def sort_key(path: Path):
        match = re.search(r"global_step_(\d+)", path.name)
        return int(match.group(1)) if match else path.name

    return [str(path) for path in sorted(candidates, key=sort_key)]


def _plot_metric(summaries: list[dict], metric_key: str, reference_key: str | None, output_path: Path, title: str, ylabel: str, actor_color: str, reference_color: str):
    import matplotlib.pyplot as plt

    steps = [item["step"] for item in summaries]
    actor_values = [item[metric_key] for item in summaries]

    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, actor_values, color="#94a3b8", alpha=0.35, linewidth=1.2)
    plt.scatter(steps, actor_values, color="#94a3b8", alpha=0.22, s=28)
    plt.plot(steps, actor_values, color=actor_color, linewidth=2.6, label=metric_key)

    if reference_key is not None:
        ref_values = [item[reference_key] for item in summaries]
        plt.plot(steps, ref_values, color=reference_color, linewidth=2.2, label=reference_key)

    plt.xlabel("Checkpoint step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


@hydra.main(config_path="config", config_name="refplay_trainer", version_base=None)
def main(config):
    run_checkpoint_sweep(config)


def run_checkpoint_sweep(config) -> None:
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
        ray.init(**ray_init_kwargs)

    runner = SweepTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class SweepTaskRunner:
    def run(self, config):
        from recipe.spin.spin_trainer import ResourcePoolManager, Role
        from verl.single_controller.ray import RayWorkerGroup
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local

        from recipe.refplay.gsm8k_dense_reward import RuleBasedGSM8KReward
        from recipe.refplay.refplay_worker import RefPlayActorRolloutRefWorker

        evaluation_cfg = config.get("evaluation")
        if evaluation_cfg is None or evaluation_cfg.get("checkpoint_root") is None:
            raise ValueError("Set evaluation.checkpoint_root to a folder containing global_step_* checkpoints")

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
        if reward_manager_name not in {"refplay_gsm8k_dense", "refplay_gsm8k_rule"}:
            compute_score = get_custom_reward_fn(config)
            from verl.workers.reward_manager import NaiveRewardManager

            reward_fn = NaiveRewardManager(
                tokenizer=tokenizer,
                num_examine=0,
                compute_score=compute_score,
                reward_fn_key=config.data.reward_fn_key,
            )
            val_reward_fn = NaiveRewardManager(
                tokenizer=tokenizer,
                num_examine=1,
                compute_score=compute_score,
                reward_fn_key=config.data.reward_fn_key,
            )
        else:
            reward_fn = RuleBasedGSM8KReward(tokenizer=tokenizer, num_examine=0)
            val_reward_fn = RuleBasedGSM8KReward(tokenizer=tokenizer, num_examine=1)

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

        checkpoint_paths = _sorted_checkpoint_paths(evaluation_cfg.checkpoint_root)
        output_root = Path(evaluation_cfg.get("output_dir", "outputs/refplay_eval/checkpoint_sweep"))
        output_root.mkdir(parents=True, exist_ok=True)

        summaries = []
        for checkpoint_path in checkpoint_paths:
            checkpoint_name = Path(checkpoint_path).name
            checkpoint_output_dir = output_root / checkpoint_name
            summary = trainer.evaluate_against_reference(
                checkpoint_path=checkpoint_path,
                output_dir=str(checkpoint_output_dir),
                preview_count=evaluation_cfg.get("preview_samples", 5),
                max_samples=evaluation_cfg.get("max_samples"),
            )
            match = re.search(r"global_step_(\d+)", checkpoint_name)
            summary["step"] = int(match.group(1)) if match else checkpoint_name
            summaries.append(summary)

        aggregate_path = output_root / "checkpoint_metrics.json"
        aggregate_path.write_text(json.dumps(summaries, indent=2) + "\n")

        _plot_metric(
            summaries,
            metric_key="actor_reward_mean",
            reference_key="reference_reward_mean",
            output_path=output_root / "reward_on_checkpoint.png",
            title="RefGame Eval Reward by Checkpoint",
            ylabel="Reward mean",
            actor_color="#2563eb",
            reference_color="#f97316",
        )
        _plot_metric(
            summaries,
            metric_key="actor_win_rate",
            reference_key="reference_win_rate",
            output_path=output_root / "winrate_on_checkpoint.png",
            title="RefGame Eval Win Rate by Checkpoint",
            ylabel="Win rate",
            actor_color="#16a34a",
            reference_color="#dc2626",
        )

        print(f"Saved checkpoint sweep metrics to {aggregate_path}")
        print(f"Saved checkpoint sweep plots to {output_root}")


if __name__ == "__main__":
    main()
