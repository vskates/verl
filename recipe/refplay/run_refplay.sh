set -e
set -x

VISIBLE_DEVICES="${VISIBLE_DEVICES:-3}"
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# HF rollout calls summon_full_params() during generation; param offload
# leaves the flat param on CPU and trips FSDP's compute-device assertion.
CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} python3 -m recipe.refplay.main_refplay \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  data.train_batch_size=16 \
  data.dataloader_num_workers=2 \
  data.max_prompt_length=512 \
  data.max_response_length=256 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  reference_rollout.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  trainer.logger=['console'] \
  trainer.val_before_train=False \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  +trainer.log_freq=1 \
  trainer.total_epochs=1000 2>&1 | tee verl_demo.log
