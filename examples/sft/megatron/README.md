# Megatron SFT Docker Testing Environment

This directory contains Docker Compose configuration and scripts for testing the Megatron SFT implementation with distributed training capabilities. The setup creates a multi-container environment with Ray and Megatron-LM to validate the SFT worker implementation.

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA Container Toolkit installed (for GPU support)
- 2 GPUs (works well on consumer GPUs like RTX 4090)

## Setup

1. Run the setup script to prepare the environment:

```bash
chmod +x setup_dockerized_test.sh
./setup_dockerized_test.sh
```

This script will:
- Create necessary directories for data and checkpoints
- Generate sample data for testing
- Update paths in the configuration files
- Prepare a modified test script with shorter training duration

## Running the Test

Start the Docker Compose environment:

```bash
docker-compose up -d
```

This will create two containers:
- `ray-head`: Ray head node with 2 GPUs
- `megatron-sft`: Container that runs the Megatron SFT example with tensor parallelism across 2 GPUs

## Monitoring the Test

Follow the logs of the Megatron SFT container to monitor the training progress:

```bash
docker-compose logs -f megatron-sft
```

You can also access the Ray dashboard at http://localhost:8265 to monitor the cluster.

## Stopping the Test

To stop and remove all containers:

```bash
docker-compose down
```

## Configuration Options

The Docker Compose environment can be customized by modifying the following files:

- `docker-compose.yml`: Adjust resource allocation and network settings
- `run_llama2_7b_megatron.sh`: Modify training parameters, model configuration, and data paths

## About Model Parallelism

This setup demonstrates tensor parallelism across 2 GPUs, which is one of the key features of Megatron-LM. The 7B LLaMA model is split across two GPUs, with each GPU holding part of the model layers. This allows training larger models than would fit on a single GPU.

The configuration uses:
- Tensor parallel size: 2 (splits model parameters across 2 GPUs)
- Pipeline parallel size: 1 (no pipeline parallelism in this config)

## Using Your Own Data

To use your own data:

1. Place your parquet files in `/tmp/data/sft/` or update the volume mount paths in the docker-compose files
2. Modify the appropriate run script to point to your data files

## Troubleshooting

- If containers fail to start, check if the required GPUs are available and visible to Docker
- If out of memory errors occur, try reducing batch sizes or model parallelism dimensions
- For NCCL errors, ensure that all GPUs are connected properly (should be on the same PCIe switch)

## Further Customization

For more complex testing scenarios:

- Test with smaller models (e.g., TinyLlama) if memory is limited
- Try different configurations of tensor and pipeline parallelism
- Experiment with different micro batch sizes to optimize GPU utilization 