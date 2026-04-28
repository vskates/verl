import argparse
import json
import os

import hydra
import ray
from omegaconf import OmegaConf

from recipe.refplay.main_refplay import get_custom_reward_fn
from recipe.crossplay.crossplay_ray_trainer import RayCrossPlayTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline CrossPlay evaluation with optional per-policy checkpoints.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--policy-a-model", required=True)
    parser.add_argument("--policy-b-model", required=True)
    parser.add_argument("--policy-a-ckpt")
    parser.add_argument("--policy-b-ckpt")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--gpus-per-node", type=int, default=2)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                    "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID"),
                    "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
                }
            }
        )

    from recipe.crossplay.crossplay_worker import CrossPlayActorRolloutRefWorker
    from recipe.crossplay.text_judges import CallableTextJudge, RuleBasedGSM8KTextJudge
    from recipe.spin.spin_trainer import ResourcePoolManager, Role
    from verl.single_controller.ray import RayWorkerGroup
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.fs import copy_to_local

    with hydra.initialize_config_dir(config_dir=os.path.abspath("recipe/crossplay/config"), version_base=None):
        config = hydra.compose(config_name="crossplay_trainer")
    config.data.train_files = args.train_file
    config.data.val_files = args.val_file
    config.data.train_batch_size = 1
    config.data.val_batch_size = args.val_batch_size
    config.data.dataloader_num_workers = 0
    config.data.max_prompt_length = 96
    config.data.max_response_length = 24
    config.data.validation_shuffle = False

    config.actor_rollout_ref.model.path = args.policy_a_model
    config.policy_b.model.path = args.policy_b_model
    config.anchor_a.model.path = args.policy_a_model
    config.anchor_b.model.path = args.policy_b_model

    config.actor_rollout_ref.actor.ppo_mini_batch_size = 1
    config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 1
    config.actor_rollout_ref.actor.fsdp_config.param_offload = True
    config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = True
    config.actor_rollout_ref.actor.fsdp_config.model_dtype = "bfloat16"
    config.actor_rollout_ref.ref.fsdp_config.model_dtype = "bfloat16"
    config.actor_rollout_ref.rollout.gpu_memory_utilization = args.gpu_memory_utilization
    config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu = 1
    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu = 1

    config.reward.reward_manager.name = "crossplay_gsm8k_rule"
    config.algorithm.update_mode = "both"
    config.algorithm.focus_failed_examples_only = True
    config.algorithm.use_reference_policy = True
    config.algorithm.replay_buffer_size_per_benchmark = 8
    config.algorithm.benchmark_margin_power = 1.0
    config.algorithm.include_fresh_failures_in_update = True
    config.algorithm.buffer_only_update_batch_size = 1

    config.trainer.logger = ["console"]
    config.trainer.project_name = "verl_crossplay_eval"
    config.trainer.experiment_name = args.experiment_name
    config.trainer.n_gpus_per_node = args.gpus_per_node
    config.trainer.nnodes = args.nnodes
    config.trainer.save_freq = -1
    config.trainer.test_freq = -1
    config.trainer.total_epochs = 1
    config.trainer.total_training_steps = 1
    config.trainer.resume_mode = "disable"
    config.trainer.val_before_train = False
    config.trainer.val_only = True

    policy_a_local_path = copy_to_local(config.policy_a.model.path)
    policy_b_local_path = copy_to_local(config.policy_b.model.path)
    policy_a_tokenizer = hf_tokenizer(policy_a_local_path, trust_remote_code=config.policy_a.model.get("trust_remote_code", False))
    policy_b_tokenizer = hf_tokenizer(policy_b_local_path, trust_remote_code=config.policy_b.model.get("trust_remote_code", False))
    processor = hf_processor(policy_a_local_path, use_fast=True)

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(CrossPlayActorRolloutRefWorker),
        Role.Actor: ray.remote(CrossPlayActorRolloutRefWorker),
        Role.Rollout: ray.remote(CrossPlayActorRolloutRefWorker),
        Role.RefPolicy: ray.remote(CrossPlayActorRolloutRefWorker),
    }
    resource_pool_spec = {
        "policy_a_pool": [1] * config.trainer.nnodes,
        "policy_b_pool": [1] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: "policy_a_pool",
        Role.Rollout: "policy_a_pool",
        Role.Actor: "policy_b_pool",
        Role.RefPolicy: "policy_b_pool",
    }

    reward_manager_name = config.reward.reward_manager.name
    if reward_manager_name in {"crossplay_gsm8k_rule", "refplay_gsm8k_rule"}:
        judge = RuleBasedGSM8KTextJudge()
    else:
        custom_fn = get_custom_reward_fn(config)
        if custom_fn is None:
            raise NotImplementedError(f"Unsupported crossplay reward manager {reward_manager_name!r} without custom reward.")
        judge = CallableTextJudge(custom_fn)

    trainer = RayCrossPlayTrainer(
        config=config,
        tokenizer=policy_a_tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping),
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

    if args.policy_a_ckpt:
        trainer.policy_a_wg.load_model_checkpoint_only(args.policy_a_ckpt, del_local_after_load=False)
    if args.policy_b_ckpt:
        trainer.policy_b_wg.load_model_checkpoint_only(args.policy_b_ckpt, del_local_after_load=False)

    metrics = trainer._evaluate_duel(max_samples=args.max_samples)
    output = {
        "experiment_name": args.experiment_name,
        "policy_a_model": args.policy_a_model,
        "policy_b_model": args.policy_b_model,
        "policy_a_ckpt": args.policy_a_ckpt,
        "policy_b_ckpt": args.policy_b_ckpt,
        "max_samples": args.max_samples,
        "val_batch_size": args.val_batch_size,
        "metrics": metrics,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)

    print(json.dumps(output, indent=2, sort_keys=True))
    ray.shutdown()


if __name__ == "__main__":
    main()
