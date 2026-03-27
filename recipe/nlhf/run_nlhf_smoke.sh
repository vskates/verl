#!/usr/bin/env bash
set -euxo pipefail

VISIBLE_DEVICES="${VISIBLE_DEVICES:-0}"
RUN_NAME="${RUN_NAME:-nlhf_smoke}"
TRAIN_FILE="${TRAIN_FILE:-/root/data/gsm8k/train.parquet}"
VAL_FILE="${VAL_FILE:-/root/data/gsm8k/test.parquet}"
TOTAL_STEPS="${TOTAL_STEPS:-2}"

export HYDRA_FULL_ERROR=1
CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES" python3 -m recipe.nlhf.main_nlhf \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=4 \
  data.max_prompt_length=256 \
  data.max_response_length=96 \
  policy_a.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  policy_b.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  anchor_rollout.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  policy_a.actor.ppo_mini_batch_size=4 \
  policy_a.actor.ppo_micro_batch_size_per_gpu=1 \
  policy_a.actor.fsdp_config.param_offload=True \
  policy_a.actor.fsdp_config.optimizer_offload=True \
  policy_a.actor.fsdp_config.model_dtype=bfloat16 \
  policy_a.ref.fsdp_config.model_dtype=bfloat16 \
  policy_a.rollout.log_prob_micro_batch_size_per_gpu=1 \
  policy_a.ref.log_prob_micro_batch_size_per_gpu=1 \
  policy_b.actor.ppo_mini_batch_size=4 \
  policy_b.actor.ppo_micro_batch_size_per_gpu=1 \
  policy_b.actor.fsdp_config.param_offload=True \
  policy_b.actor.fsdp_config.optimizer_offload=True \
  policy_b.actor.fsdp_config.model_dtype=bfloat16 \
  policy_b.ref.fsdp_config.model_dtype=bfloat16 \
  policy_b.rollout.log_prob_micro_batch_size_per_gpu=1 \
  policy_b.ref.log_prob_micro_batch_size_per_gpu=1 \
  anchor_rollout.actor.fsdp_config.param_offload=True \
  anchor_rollout.actor.fsdp_config.optimizer_offload=True \
  anchor_rollout.actor.fsdp_config.model_dtype=bfloat16 \
  anchor_rollout.ref.fsdp_config.model_dtype=bfloat16 \
  anchor_rollout.rollout.log_prob_micro_batch_size_per_gpu=1 \
  anchor_rollout.ref.log_prob_micro_batch_size_per_gpu=1 \
  reward.reward_manager.name=refplay_gsm8k_rule \
  algorithm.update_mode=alternating \
  'trainer.logger=[console]' \
  trainer.project_name=verl_nlhf \
  trainer.experiment_name="$RUN_NAME" \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq="$TOTAL_STEPS" \
  trainer.total_epochs=1 \
  trainer.total_training_steps="$TOTAL_STEPS"

