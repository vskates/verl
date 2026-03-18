set -e
set -x
set -o pipefail

VISIBLE_DEVICES="${VISIBLE_DEVICES:-3}"
TOTAL_STEPS="${TOTAL_STEPS:-40}"
RUN_NAME="${RUN_NAME:-refplay_math40}"
BASE_DIR="${BASE_DIR:-/workspace/verl/verl}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${BASE_DIR}/checkpoints/${RUN_NAME}}"
EVAL_DIR="${EVAL_DIR:-${BASE_DIR}/outputs/refplay_eval/${RUN_NAME}}"
LOG_PATH="${LOG_PATH:-/tmp/${RUN_NAME}.log}"

ACTOR_MODEL="${ACTOR_MODEL:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
REFERENCE_MODEL="${REFERENCE_MODEL:-Qwen/Qwen2.5-Math-1.5B-Instruct}"

export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

rm -rf "${CHECKPOINT_DIR}" "${EVAL_DIR}"

CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} python3 -m recipe.refplay.main_refplay \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  data.train_batch_size=4 \
  data.dataloader_num_workers=2 \
  data.max_prompt_length=512 \
  data.max_response_length=256 \
  actor_rollout_ref.model.path=${ACTOR_MODEL} \
  reference_rollout.model.path=${REFERENCE_MODEL} \
  reward.reward_manager.name=refplay_gsm8k_dense \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
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
  trainer.logger=['console'] \
  trainer.project_name=verl_refplay \
  trainer.experiment_name=${RUN_NAME} \
  trainer.default_local_dir=${CHECKPOINT_DIR} \
  trainer.val_before_train=False \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=${TOTAL_STEPS} \
  trainer.test_freq=-1 \
  +trainer.log_freq=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=${TOTAL_STEPS} >> "${LOG_PATH}" 2>&1

CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} python3 -m recipe.refplay.eval_refplay \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  data.train_batch_size=2 \
  data.val_batch_size=16 \
  data.max_prompt_length=384 \
  data.max_response_length=192 \
  actor_rollout_ref.model.path=${ACTOR_MODEL} \
  reference_rollout.model.path=${REFERENCE_MODEL} \
  reward.reward_manager.name=refplay_gsm8k_dense \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
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
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  trainer.logger=['console'] \
  trainer.val_before_train=False \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  evaluation.checkpoint_path=${CHECKPOINT_DIR}/global_step_${TOTAL_STEPS} \
  evaluation.output_dir=${EVAL_DIR} \
  evaluation.val_batch_size=16 \
  evaluation.max_samples=null \
  evaluation.preview_samples=5 >> "${LOG_PATH}" 2>&1
