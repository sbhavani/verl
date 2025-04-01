#!/bin/bash
# Setup script for the dockerized Megatron SFT test environment

echo "Setting up Megatron SFT test environment in GPU mode (2 GPU configuration)"

# Create necessary directories
mkdir -p /tmp/data/sft
mkdir -p /tmp/checkpoints/megatron_sft

# Download a small sample dataset for testing (if needed)
# This is just a placeholder - replace with actual data download if needed
echo "Downloading sample data..."
if [ ! -f "/tmp/data/sft/sample_train.parquet" ]; then
    echo "Creating sample data..."
    # Create a dummy parquet file for testing
    # In a real scenario, you would download or prepare actual data
    python -c "
import pandas as pd
import numpy as np

# Create a simple dummy dataset for SFT
data = {
    'input_ids': [np.random.randint(0, 30000, size=256).tolist() for _ in range(10)],
    'attention_mask': [np.ones(256).tolist() for _ in range(10)],
    'labels': [np.random.randint(-100, 30000, size=256).tolist() for _ in range(10)]
}

df = pd.DataFrame(data)
df.to_parquet('/tmp/data/sft/sample_train.parquet')
df.to_parquet('/tmp/data/sft/sample_val.parquet')
print('Created sample data files in /tmp/data/sft/')
"
fi

COMPOSE_FILE="docker-compose.yml"
# Update the GPU docker-compose.yml file with the correct paths
sed -i 's|/path/to/data|/tmp/data|g' $COMPOSE_FILE
sed -i 's|/path/to/checkpoints|/tmp/checkpoints|g' $COMPOSE_FILE

# Update the GPU example script
TMP_SCRIPT=$(mktemp)
cat > $TMP_SCRIPT << 'EOF'
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
EOF

cp $TMP_SCRIPT run_llama2_7b_megatron.sh
chmod +x run_llama2_7b_megatron.sh

echo "Setup complete. You can now run the test with: docker-compose up"
echo "This will test the Megatron SFT implementation with 2 GPUs in tensor parallel configuration."
echo "Use 'docker-compose logs -f megatron-sft' to follow the training process." 