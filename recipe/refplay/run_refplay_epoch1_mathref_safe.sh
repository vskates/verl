set -e
set -x
set -o pipefail

VISIBLE_DEVICES="${VISIBLE_DEVICES:-2}"
RUN_NAME="${RUN_NAME:-refplay_epoch1_mathref_safe}"
BASE_DIR="${BASE_DIR:-/workspace/verl/verl}"
TRAIN_FILE="${TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}"
VAL_FILE="${VAL_FILE:-$HOME/data/gsm8k/test.parquet}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-384}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-96}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${BASE_DIR}/checkpoints/${RUN_NAME}}"
PLOT_DIR="${PLOT_DIR:-${BASE_DIR}/outputs/refplay_plots/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${BASE_DIR}/outputs/refplay_logs}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/${RUN_NAME}.log}"
RAY_TMPDIR="${RAY_TMPDIR:-/workspace/verl/r}"
RAY_MEMORY_USAGE_THRESHOLD="${RAY_MEMORY_USAGE_THRESHOLD:-0.99}"

ACTOR_MODEL="${ACTOR_MODEL:-Qwen/Qwen2.5-1.5B}"
REFERENCE_MODEL="${REFERENCE_MODEL:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
TOTAL_STEPS_OVERRIDE="${TOTAL_STEPS_OVERRIDE:-}"
RESUME_IF_AVAILABLE="${RESUME_IF_AVAILABLE:-1}"
RESUME_WEIGHTS_ONLY="${RESUME_WEIGHTS_ONLY:-1}"

TOTAL_STEPS=$(python3 - <<PY
import math
import pyarrow.parquet as pq
train_file = r"${TRAIN_FILE}"
batch_size = int("${TRAIN_BATCH_SIZE}")
num_rows = pq.ParquetFile(train_file).metadata.num_rows
print(math.ceil(num_rows / batch_size))
PY
)

if [ -n "${TOTAL_STEPS_OVERRIDE}" ]; then
  TOTAL_STEPS="${TOTAL_STEPS_OVERRIDE}"
fi

NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-10}"
SAVE_FREQ=$(python3 - <<PY
import math
total_steps = int("${TOTAL_STEPS}")
num_checkpoints = max(1, int("${NUM_CHECKPOINTS}"))
print(max(1, math.ceil(total_steps / num_checkpoints)))
PY
)

export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RAY_memory_usage_threshold="${RAY_MEMORY_USAGE_THRESHOLD}"
export RAY_TMPDIR="${RAY_TMPDIR}"

mkdir -p "${RAY_TMPDIR}"
mkdir -p "${LOG_DIR}"
LATEST_CKPT=""
if [ -d "${CHECKPOINT_DIR}" ]; then
  LATEST_CKPT=$(find "${CHECKPOINT_DIR}" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1 || true)
fi

if [ "${RESUME_IF_AVAILABLE}" != "1" ] || [ -z "${LATEST_CKPT}" ]; then
  rm -rf "${CHECKPOINT_DIR}" "${PLOT_DIR}"
  LATEST_CKPT=""
fi
rm -rf "${RAY_TMPDIR:?}/"*
if [ -z "${LATEST_CKPT}" ]; then
  rm -f "${LOG_PATH}"
fi

# Clean up stale local Ray state before a long run to avoid /tmp spills and
# stale object store pressure from previous jobs inside the container.
ray stop --force || true

RESUME_ARGS=()
if [ -n "${LATEST_CKPT}" ]; then
  echo "Resuming from ${LATEST_CKPT}" | tee -a "${LOG_PATH}"
  RESUME_ARGS+=(
    trainer.resume_mode=resume_path
    trainer.resume_from_path=${LATEST_CKPT}
    +trainer.resume_weights_only=${RESUME_WEIGHTS_ONLY}
  )
fi

CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} python3 -m recipe.refplay.main_refplay \
  data.train_files=${TRAIN_FILE} \
  data.val_files=${VAL_FILE} \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.dataloader_num_workers=${DATALOADER_NUM_WORKERS} \
  data.max_prompt_length=${MAX_PROMPT_LENGTH} \
  data.max_response_length=${MAX_RESPONSE_LENGTH} \
  actor_rollout_ref.model.path=${ACTOR_MODEL} \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  reference_rollout.model.path=${REFERENCE_MODEL} \
  reward.reward_manager.name=refplay_gsm8k_rule \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  trainer.logger=['console'] \
  trainer.project_name=verl_refplay \
  trainer.experiment_name=${RUN_NAME} \
  trainer.default_local_dir=${CHECKPOINT_DIR} \
  trainer.val_before_train=False \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.test_freq=-1 \
  +trainer.log_freq=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=${TOTAL_STEPS} \
  "${RESUME_ARGS[@]}" >> "${LOG_PATH}" 2>&1

python3 -m recipe.refplay.plot_refplay_metrics "${LOG_PATH}" --output-dir "${PLOT_DIR}" >> "${LOG_PATH}" 2>&1

echo "Finished 1 epoch with math-finetuned reference."
echo "Run: ${RUN_NAME}"
echo "Log: ${LOG_PATH}"
echo "Plots: ${PLOT_DIR}"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "Ray tmpdir: ${RAY_TMPDIR}"
echo "Checkpoint interval: every ${SAVE_FREQ} steps (~${NUM_CHECKPOINTS} saves total)"
