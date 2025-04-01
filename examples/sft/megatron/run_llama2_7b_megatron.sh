#!/bin/bash
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Get the number of GPUs as the first argument, default to 8
NUM_GPUS=${1:-8}
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
  model.megatron.tensor_model_parallel_size=4 \
  model.megatron.pipeline_model_parallel_size=2 \
  model.megatron.virtual_pipeline_model_parallel_size=2 \
  training.micro_batch_size_per_gpu=2 \
  training.gradient_accumulation_steps=8 \
  data.train_batch_size=16 \
  data.micro_batch_size_per_gpu=2 \
  data.train_files=["/path/to/your/train_data.parquet"] \
  data.val_files=["/path/to/your/val_data.parquet"] \
  optimizer.lr=1e-5 \
  optimizer.warmup_steps=100 \
  trainer.max_steps=5000 \
  trainer.project_name=llama2-sft-megatron \
  trainer.experiment_name=llama2-7b-megatron-sft \
  checkpoint.save_dir=$CKPT_DIR \
  trainer.val_check_interval=500 \
  trainer.save_interval=1000 