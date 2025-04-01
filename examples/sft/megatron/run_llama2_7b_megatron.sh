#!/bin/bash
# Get the number of GPUs as the first argument, default to 2
NUM_GPUS=${1:-2}
CKPT_DIR=${2:-"./checkpoints/llama2-7b-megatron-sft"}

# Ensure checkpoint directory exists
mkdir -p $CKPT_DIR

# Run the Megatron SFT trainer with the specified configuration
torchrun \
  --nnodes=1 \
  --nproc_per_node=$NUM_GPUS \
  --rdzv_id=1 \
  --rdzv_backend=c10d \
  -m verl.trainer.megatron_sft_trainer \
  model.partial_pretrain="meta-llama/Llama-2-7b-hf" \
  model.megatron.tensor_model_parallel_size=2 \
  model.megatron.pipeline_model_parallel_size=1 \
  model.megatron.virtual_pipeline_model_parallel_size=None \
  training.micro_batch_size_per_gpu=4 \
  training.gradient_accumulation_steps=8 \
  data.train_batch_size=32 \
  data.micro_batch_size_per_gpu=4 \
  data.train_files=["/workspace/data/sft/sample_train.parquet"] \
  data.val_files=["/workspace/data/sft/sample_val.parquet"] \
  optimizer.lr=1e-5 \
  optimizer.warmup_steps=10 \
  trainer.max_steps=20 \
  trainer.project_name=llama2-sft-megatron-test \
  trainer.experiment_name=llama2-7b-megatron-sft-test \
  checkpoint.save_dir=$CKPT_DIR \
  trainer.val_check_interval=10 \
  trainer.save_interval=10
