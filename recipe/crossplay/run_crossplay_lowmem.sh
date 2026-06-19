#!/usr/bin/env bash
set -euxo pipefail

VISIBLE_DEVICES="${VISIBLE_DEVICES:-0,3}"
RUN_NAME="${RUN_NAME:-crossplay_refdpo_lowmem}"
TRAIN_FILE="${TRAIN_FILE:-/workspace/runroot/data/gsm8k/train.parquet}"
VAL_FILE="${VAL_FILE:-/workspace/runroot/data/gsm8k/test.parquet}"
TOTAL_STEPS="${TOTAL_STEPS:-3737}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-10}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-128}"
HF_HOME_DIR="${HF_HOME_DIR:-/workspace/runroot/hf_cache}"
RUNTIME_SRC_DIR="${RUNTIME_SRC_DIR:-/workspace/runroot/runtime_src}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"
MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-12}"

SAVE_FREQ="${SAVE_FREQ:-$(python3 - <<PY
import math
total_steps = max(int(${TOTAL_STEPS}), 1)
num_checkpoints = max(int(${NUM_CHECKPOINTS}), 1)
print(max(1, math.ceil(total_steps / num_checkpoints)))
PY
)}"

export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
export HF_HOME="${HF_HOME_DIR}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME_DIR}/hub"

mkdir -p "${HUGGINGFACE_HUB_CACHE}"

if [ -d "${RUNTIME_SRC_DIR}" ]; then
  cd "${RUNTIME_SRC_DIR}"
fi

CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES" python3 -m recipe.crossplay.main_crossplay \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
  data.val_max_samples="${VAL_MAX_SAMPLES}" \
  data.train_batch_size=1 \
  data.dataloader_num_workers=0 \
  data.max_prompt_length=96 \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
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
  actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  reward.reward_manager.name=crossplay_gsm8k_dense \
  algorithm.update_mode=both \
  algorithm.focus_failed_examples_only=false \
  algorithm.use_reference_policy=true \
  algorithm.replay_buffer_size_per_benchmark=8 \
  algorithm.benchmark_margin_power=1.0 \
  algorithm.include_fresh_failures_in_update=true \
  algorithm.buffer_only_update_batch_size=1 \
  'trainer.logger=[console]' \
  trainer.project_name=verl_crossplay \
  trainer.experiment_name="$RUN_NAME" \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.max_actor_ckpt_to_keep="${MAX_CKPT_TO_KEEP}" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.total_epochs=1 \
  trainer.total_training_steps="$TOTAL_STEPS"
