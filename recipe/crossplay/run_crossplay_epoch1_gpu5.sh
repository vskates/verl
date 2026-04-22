#!/usr/bin/env bash
set -euxo pipefail

VISIBLE_DEVICES="${VISIBLE_DEVICES:-5}"
RUN_NAME="${RUN_NAME:-crossplay_refdpo_epoch1_gpu5}"
TRAIN_FILE="${TRAIN_FILE:-/root/data/gsm8k/train.parquet}"
VAL_FILE="${VAL_FILE:-/root/data/gsm8k/test.parquet}"
TOTAL_STEPS="${TOTAL_STEPS:-3737}"
SAVE_FREQ="${SAVE_FREQ:-374}"

export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-disabled}"

CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES" python3 -m recipe.crossplay.main_crossplay \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=2 \
  data.max_prompt_length=256 \
  data.max_response_length=64 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  policy_b.model.path=HuggingFaceTB/SmolLM2-1.7B-Instruct \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  reward.reward_manager.name=crossplay_gsm8k_rule \
  algorithm.update_mode=both \
  algorithm.focus_failed_examples_only=true \
  algorithm.use_reference_policy=true \
  algorithm.replay_buffer_size_per_benchmark=64 \
  algorithm.benchmark_margin_power=1.0 \
  algorithm.include_fresh_failures_in_update=true \
  algorithm.buffer_only_update_batch_size=2 \
  'trainer.logger=[console]' \
  trainer.project_name=verl_crossplay \
  trainer.experiment_name="$RUN_NAME" \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.total_epochs=1 \
  trainer.total_training_steps="$TOTAL_STEPS"
