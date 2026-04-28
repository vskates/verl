#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-crossplay_budget_full_resume_safe_20260425}"
VISIBLE_DEVICES="${VISIBLE_DEVICES:-0,1}"
TRAIN_FILE="${TRAIN_FILE:-/workspace/runroot/data/gsm8k/train_plain.parquet}"
VAL_FILE="${VAL_FILE:-/workspace/runroot/data/gsm8k/test_plain.parquet}"
RESUME_PATH="${RESUME_PATH:-/workspace/runroot/runtime_src/checkpoints/verl_crossplay/crossplay_budget_full_20260422/global_step_373}"
LOG_DIR="${LOG_DIR:-/home/vskate/runroot/crossplay_logs}"
RAY_TMPDIR="${RAY_TMPDIR:-/workspace/runroot/ray_tmp}"
HF_HOME_DIR="${HF_HOME_DIR:-/workspace/runroot/hf_cache}"

mkdir -p "${LOG_DIR}" "${RAY_TMPDIR}" "${HF_HOME_DIR}/hub"

export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
export HF_HOME="${HF_HOME_DIR}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME_DIR}/hub"
export TMPDIR="${RAY_TMPDIR}"
export RAY_TMPDIR="${RAY_TMPDIR}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd /workspace/runroot/runtime_src

ray stop --force >/dev/null 2>&1 || true

CUDA_VISIBLE_DEVICES="${VISIBLE_DEVICES}" python3 -m recipe.crossplay.main_crossplay \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size=1 \
  data.dataloader_num_workers=0 \
  data.max_prompt_length=96 \
  data.max_response_length=24 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  policy_b.model.path=HuggingFaceTB/SmolLM2-1.7B-Instruct \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.checkpoint.save_contents='[model]' \
  actor_rollout_ref.actor.checkpoint.load_contents='[model]' \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.18 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  reward.reward_manager.name=crossplay_gsm8k_rule \
  algorithm.update_mode=both \
  algorithm.focus_failed_examples_only=true \
  algorithm.use_reference_policy=true \
  algorithm.replay_buffer_size_per_benchmark=4 \
  algorithm.benchmark_margin_power=1.0 \
  algorithm.include_fresh_failures_in_update=true \
  algorithm.buffer_only_update_batch_size=1 \
  'trainer.logger=[console]' \
  trainer.project_name=verl_crossplay \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.default_local_dir="/workspace/runroot/runtime_src/checkpoints/verl_crossplay/${RUN_NAME}" \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.save_freq=1000 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=3737 \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="${RESUME_PATH}"
