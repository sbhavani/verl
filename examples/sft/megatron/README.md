# Megatron SFT Docker Testing Environment

This directory contains Docker Compose configuration and scripts for testing the Megatron SFT implementation with distributed training capabilities. The setup creates a multi-container environment with Ray and Megatron-LM to validate the SFT worker implementation.

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA Container Toolkit installed (for GPU support)
- At least 8 GPUs available (configuration can be adjusted in the docker-compose.yml file)

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

This will create three containers:
- `ray-head`: Ray head node with 4 GPUs
- `ray-worker`: Ray worker node with 4 GPUs
- `megatron-sft`: Container that runs the Megatron SFT example

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

- `docker-compose.yml`: Adjust GPU allocation, container resources, and network settings
- `run_llama2_7b_megatron.sh`: Modify training parameters, model configuration, and data paths

## Using Your Own Data

To use your own data:

1. Place your parquet files in `/tmp/data/sft/` or update the volume mount paths in `docker-compose.yml`
2. Modify the `run_llama2_7b_megatron.sh` script to point to your data files

## Troubleshooting

- If containers fail to start, check if the required GPUs are available and visible to Docker
- If out of memory errors occur, try reducing batch sizes or model parallelism dimensions
- For NCCL errors, ensure that all GPUs are on the same NUMA node or adjust NCCL settings

## Further Customization

For more complex testing scenarios:

- Add more worker nodes by adding new services in the docker-compose.yml file
- Test different model sizes by adjusting the model path and parallelism settings
- Experiment with different parallelism configurations by modifying tensor/pipeline settings 