# Megatron SFT: Supervised Fine-Tuning with Megatron-LM

This document describes the Megatron SFT worker implementation for supervised fine-tuning of large language models using Megatron-LM parallelism techniques.

## Overview

The Megatron SFT worker extends VERL's capabilities by providing a specialized implementation for supervised fine-tuning that leverages Megatron-LM's model parallelism features. This enables efficient training of large language models across multiple GPUs using tensor, pipeline, and virtual pipeline parallelism.

Key features include:
- Tensor parallelism for distributing model parameters across multiple GPUs
- Pipeline parallelism for splitting model layers across multiple GPUs
- Virtual pipeline parallelism for efficient memory usage
- Support for distributed optimizers and gradient checkpointing
- Comprehensive checkpoint management
- Validation and inference capabilities

## Architecture

The implementation consists of several key components:

1. **MegatronSFTWorker** (`verl/workers/megatron_sft.py`): Core worker class that implements supervised fine-tuning using Megatron-LM parallel processing techniques.

2. **MegatronSFTTrainer** (`verl/trainer/megatron_sft_trainer.py`): Trainer that utilizes the worker for the full training lifecycle including dataset handling, training loops, validation, and metrics tracking.

3. **Configuration** (`verl/trainer/config/sft_megatron.yaml`): Configuration file with settings for model, training, optimization, and data loading.

4. **Example Script** (`examples/sft/megatron/run_llama2_7b_megatron.sh`): Shell script demonstrating how to use the implementation with LLaMA-2-7B.

## Usage

### Configuration

Create or modify a configuration file based on `verl/trainer/config/sft_megatron.yaml`. Key configuration sections include:

```yaml
# Model configuration
model:
  name: megatron-sft
  partial_pretrain: "your-model-path" # Path to pretrained model
  megatron:
    tensor_model_parallel_size: 2  # Number of GPUs for tensor parallelism
    pipeline_model_parallel_size: 2 # Number of GPUs for pipeline parallelism
    virtual_pipeline_model_parallel_size: 2 # Number of chunks in PP

# Training configuration
training:
  micro_batch_size_per_gpu: 4
  gradient_accumulation_steps: 4

# Data configuration
data:
  train_files:
    - "/path/to/train_data.parquet"
  val_files:
    - "/path/to/val_data.parquet"
  train_batch_size: 64
  sequence_length: 2048
```

### Running Training

Use the provided example script as a starting point:

```bash
# Run with 8 GPUs
bash examples/sft/megatron/run_llama2_7b_megatron.sh 8 /path/to/save/checkpoints
```

Or run directly with `torchrun`:

```bash
torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  --rdzv_id=1 \
  --rdzv_backend=c10d \
  -m verl.trainer.megatron_sft_trainer \
  model.partial_pretrain="meta-llama/Llama-2-7b-hf" \
  model.megatron.tensor_model_parallel_size=4 \
  model.megatron.pipeline_model_parallel_size=2 \
  # Add other configuration overrides...
```

### Dataset Format

The implementation expects data in the following format:

- Input data in parquet files
- Each example should contain:
  - `input_ids`: Token IDs for input sequence
  - `attention_mask`: Mask for valid input tokens
  - `labels`: Labels for language modeling (with `-100` for tokens that should not contribute to the loss)

## Implementation Details

### Parallelism Strategies

The implementation supports several parallelism strategies:

1. **Tensor Parallelism (TP)**: Splits model parameters across GPUs within a single layer
2. **Pipeline Parallelism (PP)**: Splits model across layers, with each GPU handling different layers
3. **Virtual Pipeline Parallelism (VPP)**: Splits model into chunks for improved memory efficiency

### Loss Computation

The SFT worker uses a vocabulary-parallel cross-entropy loss function that's compatible with tensor-parallel training:

```python
from verl.utils.megatron.tensor_parallel import vocab_parallel_compute_entropy_loss
loss = vocab_parallel_compute_entropy_loss(logits, labels, ignore_index=-100)
```

### Checkpoint Management

Checkpoints are managed through the `MegatronCheckpointManager` class, which handles:
- Saving model weights, optimizer states, and scheduler states
- Loading from checkpoints for resuming training
- Automatically managing the number of checkpoints kept

## Advanced Features

### Gradient Checkpointing

Enable gradient checkpointing to reduce memory usage during training:

```yaml
model:
  enable_gradient_checkpointing: true
```

### Distributed Optimizer

The implementation supports Megatron's distributed optimizer for efficient memory usage:

```yaml
model:
  megatron:
    use_distributed_optimizer: true
```

### Remove Padding

For improved performance with variable sequence lengths:

```yaml
training:
  use_rmpad: true
```

## Example

For a complete working example, refer to the provided script in `examples/sft/megatron/run_llama2_7b_megatron.sh`, which demonstrates fine-tuning LLaMA-2-7B with Megatron parallelism. 