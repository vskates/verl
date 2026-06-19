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
    parser.add_argument("--max-response-length", type=int, default=128)
    parser.add_argument("--reward-manager", default="crossplay_gsm8k_dense")
    parser.add_argument("--examples-output")
    parser.add_argument("--num-examples", type=int, default=8)
    return parser


def collect_examples(trainer: RayCrossPlayTrainer, max_samples: int | None, limit: int) -> list[dict]:
    if limit <= 0:
        return []

    examples = []
    example_index = 0
    for batch_idx, test_data in enumerate(trainer.val_dataloader):
        if max_samples is not None and example_index >= max_samples:
            break
        if len(examples) >= limit:
            break

        batch = trainer.DataProto.from_single_dict(test_data) if hasattr(trainer, "DataProto") else None
        if batch is None:
            from verl import DataProto
            batch = DataProto.from_single_dict(test_data)

        raw_prompts = trainer._extract_raw_prompts(batch)
        benchmark_ids = trainer._extract_benchmark_ids(batch)
        reward_model_entries = trainer._extract_reward_entries(batch, "reward_model")
        extra_info_entries = trainer._extract_reward_entries(batch, "extra_info")

        policy_a_output = trainer._generate_with_policy(
            trainer.policy_a_wg,
            trainer._build_policy_generation_batch(batch, "policy_a", validate=True),
        )
        policy_b_output = trainer._generate_with_policy(
            trainer.policy_b_wg,
            trainer._build_policy_generation_batch(batch, "policy_b", validate=True),
        )
        policy_a_texts = trainer._decode_responses(trainer.policy_tokenizers["policy_a"], policy_a_output)
        policy_b_texts = trainer._decode_responses(trainer.policy_tokenizers["policy_b"], policy_b_output)

        a_scores, a_extras = trainer._score_response_texts(
            raw_prompts=raw_prompts,
            responses=policy_a_texts,
            benchmark_ids=benchmark_ids,
            reward_model_entries=reward_model_entries,
            extra_info_entries=extra_info_entries,
        )
        b_scores, b_extras = trainer._score_response_texts(
            raw_prompts=raw_prompts,
            responses=policy_b_texts,
            benchmark_ids=benchmark_ids,
            reward_model_entries=reward_model_entries,
            extra_info_entries=extra_info_entries,
        )

        keep = len(benchmark_ids)
        if max_samples is not None:
            keep = min(keep, max_samples - example_index)
        keep = min(keep, limit - len(examples))

        for idx in range(keep):
            prompt_text = trainer._serialize_prompt(trainer.policy_tokenizers["policy_a"], raw_prompts[idx])
            winner = "tie"
            a_score = float(a_scores[idx].item())
            b_score = float(b_scores[idx].item())
            if a_score > b_score:
                winner = "policy_a"
            elif b_score > a_score:
                winner = "policy_b"

            examples.append(
                {
                    "sample_index": example_index + idx,
                    "batch_index": batch_idx,
                    "benchmark_id": benchmark_ids[idx],
                    "prompt_text": prompt_text,
                    "ground_truth": a_extras.get("ground_truth", [""] * keep)[idx],
                    "policy_a_response": policy_a_texts[idx],
                    "policy_b_response": policy_b_texts[idx],
                    "policy_a_score": a_score,
                    "policy_b_score": b_score,
                    "policy_a_predicted_answer": a_extras.get("predicted_answer", [""] * keep)[idx],
                    "policy_b_predicted_answer": b_extras.get("predicted_answer", [""] * keep)[idx],
                    "policy_a_exact_match": a_extras.get("exact_match", [0.0] * keep)[idx],
                    "policy_b_exact_match": b_extras.get("exact_match", [0.0] * keep)[idx],
                    "policy_a_has_final_marker": a_extras.get("has_final_marker", [0.0] * keep)[idx],
                    "policy_b_has_final_marker": b_extras.get("has_final_marker", [0.0] * keep)[idx],
                    "policy_a_final_marker_type": a_extras.get("final_marker_type", ["none"] * keep)[idx],
                    "policy_b_final_marker_type": b_extras.get("final_marker_type", ["none"] * keep)[idx],
                    "policy_a_used_fallback_answer": a_extras.get("used_fallback_answer", [0.0] * keep)[idx],
                    "policy_b_used_fallback_answer": b_extras.get("used_fallback_answer", [0.0] * keep)[idx],
                    "policy_a_numeric_proximity": a_extras.get("numeric_proximity", [0.0] * keep)[idx],
                    "policy_b_numeric_proximity": b_extras.get("numeric_proximity", [0.0] * keep)[idx],
                    "winner": winner,
                }
            )
        example_index += keep

    return examples


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
    from recipe.crossplay.text_judges import CallableTextJudge, DenseGSM8KTextJudge, RuleBasedGSM8KTextJudge
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
    config.data.max_response_length = args.max_response_length
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

    config.reward.reward_manager.name = args.reward_manager
    config.algorithm.update_mode = "both"
    config.algorithm.focus_failed_examples_only = False
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
    elif reward_manager_name in {"crossplay_gsm8k_dense", "refplay_gsm8k_dense"}:
        judge = DenseGSM8KTextJudge()
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
        "max_response_length": args.max_response_length,
        "val_batch_size": args.val_batch_size,
        "reward_manager": args.reward_manager,
        "metrics": metrics,
    }
    if args.examples_output:
        output["examples_output"] = args.examples_output
        output["num_examples"] = args.num_examples

    if args.examples_output:
        examples = collect_examples(trainer, max_samples=args.max_samples, limit=args.num_examples)
        os.makedirs(os.path.dirname(args.examples_output), exist_ok=True)
        with open(args.examples_output, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment_name": args.experiment_name,
                    "policy_a_model": args.policy_a_model,
                    "policy_b_model": args.policy_b_model,
                    "policy_a_ckpt": args.policy_a_ckpt,
                    "policy_b_ckpt": args.policy_b_ckpt,
                    "max_samples": args.max_samples,
                    "num_examples": args.num_examples,
                    "examples": examples,
                },
                f,
                indent=2,
                sort_keys=True,
            )

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)

    print(json.dumps(output, indent=2, sort_keys=True))
    ray.shutdown()


if __name__ == "__main__":
    main()
