set -e
set -x

VISIBLE_DEVICES="0,1,2,3"
export HYDRA_FULL_ERROR=1

CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} python3 -m recipe.refplay.main_refplay \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  data.train_batch_size=256 \
  data.max_prompt_length=1024 \
  data.max_response_length=512 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  reference_rollout.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  reference_rollout.rollout.tensor_model_parallel_size=1 \
  reference_rollout.rollout.gpu_memory_utilization=0.4 \
  reference_rollout.rollout.n=1 \
  trainer.logger=['console'] \
  trainer.val_before_train=True \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=1 \
  +trainer.log_freq=1 \
  trainer.total_epochs=1000 2>&1 | tee verl_demo.log
