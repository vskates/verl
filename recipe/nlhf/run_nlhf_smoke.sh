#!/usr/bin/env bash
set -euxo pipefail

VISIBLE_DEVICES="${VISIBLE_DEVICES:-0}"
RUN_NAME="${RUN_NAME:-nlhf_smoke_exact}"
TRAIN_FILE="${TRAIN_FILE:-/root/data/gsm8k/train.parquet}"
VAL_FILE="${VAL_FILE:-/root/data/gsm8k/test.parquet}"
TOTAL_STEPS="${TOTAL_STEPS:-3}"
MODEL_PATH="${MODEL_PATH:-HuggingFaceTB/SmolLM2-135M-Instruct}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"

export HYDRA_FULL_ERROR=1
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.999}"
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES" python3 -m recipe.nlhf.main_nlhf \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=2 \
  data.max_prompt_length=256 \
  data.max_response_length=96 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.model_dtype="$MODEL_DTYPE" \
  actor_rollout_ref.ref.fsdp_config.model_dtype="$MODEL_DTYPE" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  reward.reward_manager.name=nlhf_gsm8k_pairwise \
  algorithm.preference_source=pairwise_probability \
  algorithm.alternative_policy_strategy=nash_ema_pg \
  algorithm.alternative_policy_sync_freq=1 \
  algorithm.alternative_policy_ema_decay=0.95 \
  algorithm.tau=0.05 \
  'trainer.logger=[console]' \
  trainer.project_name=verl_nlhf \
  trainer.experiment_name="$RUN_NAME" \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps="$TOTAL_STEPS"
