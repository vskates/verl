set -e
set -x
set -o pipefail

VISIBLE_DEVICES="${VISIBLE_DEVICES:-3}"
RUN_NAME="${RUN_NAME:-refplay_epoch1}"
BASE_DIR="${BASE_DIR:-/workspace/verl/verl}"
TRAIN_FILE="${TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}"
VAL_FILE="${VAL_FILE:-$HOME/data/gsm8k/test.parquet}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-6}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-128}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${BASE_DIR}/checkpoints/${RUN_NAME}}"
EVAL_DIR="${EVAL_DIR:-${BASE_DIR}/outputs/refplay_eval/${RUN_NAME}}"
PLOT_DIR="${PLOT_DIR:-${BASE_DIR}/outputs/refplay_plots/${RUN_NAME}}"
LOG_PATH="${LOG_PATH:-/tmp/${RUN_NAME}.log}"

ACTOR_MODEL="${ACTOR_MODEL:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
REFERENCE_MODEL="${REFERENCE_MODEL:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
PROJECT_NAME="${PROJECT_NAME:-verl_refplay}"
LOGGER_BACKENDS="${LOGGER_BACKENDS:-['console']}"
ENABLE_WANDB="${ENABLE_WANDB:-0}"
EXTRA_TRAIN_OVERRIDES="${EXTRA_TRAIN_OVERRIDES:-}"

if [ "${ENABLE_WANDB}" = "1" ]; then
  LOGGER_BACKENDS="['console','wandb']"
fi

TOTAL_STEPS=$(python3 - <<PY
import math
import pyarrow.parquet as pq

train_file = r"${TRAIN_FILE}"
batch_size = int("${TRAIN_BATCH_SIZE}")
num_rows = pq.ParquetFile(train_file).metadata.num_rows
print(math.ceil(num_rows / batch_size))
PY
)

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

rm -rf "${CHECKPOINT_DIR}" "${EVAL_DIR}" "${PLOT_DIR}"

CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} python3 -m recipe.refplay.main_refplay \
  data.train_files=${TRAIN_FILE} \
  data.val_files=${VAL_FILE} \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.dataloader_num_workers=2 \
  data.max_prompt_length=${MAX_PROMPT_LENGTH} \
  data.max_response_length=${MAX_RESPONSE_LENGTH} \
  actor_rollout_ref.model.path=${ACTOR_MODEL} \
  reference_rollout.model.path=${REFERENCE_MODEL} \
  reward.reward_manager.name=refplay_gsm8k_rule \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  trainer.logger=${LOGGER_BACKENDS} \
  trainer.project_name=${PROJECT_NAME} \
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
  ${EXTRA_TRAIN_OVERRIDES} >> "${LOG_PATH}" 2>&1

python3 -m recipe.refplay.plot_refplay_metrics "${LOG_PATH}" --output-dir "${PLOT_DIR}" >> "${LOG_PATH}" 2>&1

echo "Finished 1 epoch."
echo "Log: ${LOG_PATH}"
echo "Plots: ${PLOT_DIR}"
echo "Checkpoint: ${CHECKPOINT_DIR}/global_step_${TOTAL_STEPS}"
echo "Checkpoint interval: every ${SAVE_FREQ} steps (~${NUM_CHECKPOINTS} saves total)"
