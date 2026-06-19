#!/usr/bin/env bash
set -euxo pipefail

VISIBLE_DEVICES="${VISIBLE_DEVICES:-0}"
RUN_NAME="${RUN_NAME:-refplay_crossplay_parity}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
REFERENCE_MODEL_PATH="${REFERENCE_MODEL_PATH:-$MODEL_PATH}"
TRAIN_FILE="${TRAIN_FILE:-/workspace/runroot/data/gsm8k/train_plain.parquet}"
VAL_FILE="${VAL_FILE:-/workspace/runroot/data/gsm8k/test_plain.parquet}"
TOTAL_STEPS="${TOTAL_STEPS:-2000}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-10}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-2000}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-500}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-96}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-128}"
HF_HOME_DIR="${HF_HOME_DIR:-/workspace/runroot/hf_cache}"
RUNTIME_SRC_DIR="${RUNTIME_SRC_DIR:-/workspace/runroot/runtime_src}"
RAY_TMPDIR="${RAY_TMPDIR:-/workspace/runroot/refplay_runtime/${RUN_NAME}/ray_tmp}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/runroot/runtime_src/checkpoints/verl_refplay/${RUN_NAME}}"
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
export TMPDIR="${RAY_TMPDIR}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

mkdir -p "${HUGGINGFACE_HUB_CACHE}" "${RAY_TMPDIR}" "$(dirname "${CHECKPOINT_DIR}")"

if [ -d "${RUNTIME_SRC_DIR}" ]; then
  cd "${RUNTIME_SRC_DIR}"
fi

ray stop --force >/dev/null 2>&1 || true

CUDA_VISIBLE_DEVICES="${VISIBLE_DEVICES}" python3 -m recipe.refplay.main_refplay \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
  data.val_max_samples="${VAL_MAX_SAMPLES}" \
  data.train_batch_size=1 \
  data.dataloader_num_workers=0 \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  '+data.gsm8k_answer_instruction="Please reason step by step, and put the final numeric answer on the last line as: #### <answer>"' \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  reference_rollout.model.path="${REFERENCE_MODEL_PATH}" \
  actor_rollout_ref.actor.dpo_beta=0.1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.checkpoint.contents='[model,optimizer,extra]' \
  actor_rollout_ref.actor.checkpoint.save_contents='[model,optimizer,extra]' \
  actor_rollout_ref.actor.checkpoint.load_contents='[model,optimizer,extra]' \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  reward.reward_manager.name=refplay_gsm8k_dense \
  algorithm.dpo_beta=0.1 \
  algorithm.dpo_loss_type=sigmoid \
  algorithm.dpo_label_smoothing=0.0 \
  'trainer.logger=[console]' \
  trainer.project_name=verl_refplay \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.default_local_dir="${CHECKPOINT_DIR}" \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.max_actor_ckpt_to_keep="${MAX_CKPT_TO_KEEP}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.total_epochs=1 \
  trainer.total_training_steps="${TOTAL_STEPS}"
