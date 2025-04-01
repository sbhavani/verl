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
"""
Megatron SFT worker implementation for supervised fine-tuning.
"""

import os
import logging
import ray
import torch
import torch.distributed
import torch.nn as nn
from omegaconf import DictConfig
from typing import Dict, Iterable

from verl.single_controller.base.megatron.worker import MegatronWorker
from verl.single_controller.base.decorator import register, Dispatch
from verl import DataProto
from verl.utils.fs import copy_to_local
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.model import load_megatron_model_weights
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
from verl.utils.megatron_utils import init_model_parallel_config
from verl.utils.megatron_utils import get_model_config
from verl.utils import hf_tokenizer

from megatron.core import parallel_state as mpu
from megatron.core import ModelParallelConfig
from megatron.core.distributed import finalize_model_grads
from megatron.core.optimizer import DistributedOptimizer, OptimizerConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_SFT_LOGGING_LEVEL', 'WARN'))


def set_random_seed(seed):
    import torch
    import numpy as np
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.device_count() > 0:
        from megatron.core import tensor_parallel
        tensor_parallel.model_parallel_cuda_manual_seed(seed)


class MegatronSFTWorker(MegatronWorker):
    """
    A worker for supervised fine-tuning using Megatron-LM.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        # Initialize distributed environment if not already done
        if not torch.distributed.is_initialized():
            rank = int(os.environ['LOCAL_RANK'])
            torch.distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(rank)

            if self.config.model.megatron.sequence_parallel:
                os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.model.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.model.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.model.megatron.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=1,
                expert_model_parallel_size=1,
                nccl_communicator_config_path=None,
            )

        set_random_seed(seed=self.config.model.megatron.seed)

        # Initialize model properties
        self.model = None
        self.optimizer = None
        self.tokenizer = None
        self.lr_scheduler = None
        self.checkpoint_manager = None

    def _build_model_optimizer(self, model_path, megatron_config: ModelParallelConfig, optim_config,
                              override_model_config, enable_gradient_checkpointing=False):
        """
        Build model and optimizer for SFT.
        """
        from verl.utils.megatron.optimizer import get_megatron_optimizer
        from megatron.core.models.gpt.gpt_model import ModelType
        from verl.utils.model import print_model_size, update_model_config, get_generation_config
        from verl.utils.megatron_utils import get_model, init_megatron_optim_config
        from transformers import AutoConfig

        # Step 1: initialize the tokenizer
        local_path = copy_to_local(model_path)
        self.tokenizer = hf_tokenizer(local_path)

        # Step 2: get the model_config
        model_config = AutoConfig.from_pretrained(local_path)
        self.generation_config = get_generation_config(local_path)

        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(model_config, override_config_kwargs=override_config_kwargs)

        if self.rank == 0:
            print(f'Model config after override: {model_config}')

        self.share_embeddings_and_output_weights = getattr(model_config, "tie_word_embeddings", False)
        self.architectures = getattr(model_config, "architectures", None)

        def megatron_model_provider(pre_process, post_process):
            from verl.utils.model import get_parallel_model_from_config
            # Get virtual pipeline model parallel rank
            vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
            parallel_model = get_parallel_model_from_config(
                config=model_config,
                megatron_config=megatron_config,
                pre_process=pre_process,
                post_process=post_process,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                value=False)
            parallel_model.cuda()
            return parallel_model

        # Step 3: Initialize the model
        sft_module = get_model(
            model_provider_func=megatron_model_provider,
            model_type=ModelType.encoder_or_decoder,
            wrap_with_ddp=False,
            use_distributed_optimizer=self.config.model.megatron.use_distributed_optimizer)

        if self.config.model.load_weight:
            self.hf_config = load_megatron_model_weights(
                self.config,
                model_config,
                sft_module,
                params_dtype=megatron_config.params_dtype,
                is_value_model=False)

        if self.rank == 0:
            print_model_size(sft_module[0] if isinstance(sft_module, list) else sft_module)
        log_gpu_memory_usage('After SFT model init', logger=logger)

        # Step 4: Initialize the optimizer and lr scheduler
        if optim_config is not None:
            optim_config = init_megatron_optim_config(optim_config)
            optimizer = get_megatron_optimizer(model=sft_module, config=optim_config)
            log_gpu_memory_usage('After optimizer init', logger=logger)

            # Create learning rate scheduler
            from verl.utils.torch_functional import get_cosine_schedule_with_warmup
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.config.optimizer.warmup_steps,
                num_training_steps=self.config.trainer.max_steps)
        else:
            optimizer = None
            lr_scheduler = None

        return sft_module, optimizer, lr_scheduler, model_config, optim_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize the SFT model, optimizer, and lr scheduler."""
        # Create Megatron config
        megatron_config = init_model_parallel_config(
            self.config.model.megatron,
            world_size=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank())

        # Initialize optimizer config
        if hasattr(self.config, 'optimizer'):
            optim_config = self.config.optimizer
        else:
            optim_config = None

        # Build model and optimizer
        self.model, self.optimizer, self.lr_scheduler, self.model_config, self.optim_config = self._build_model_optimizer(
            model_path=self.config.model.partial_pretrain,
            megatron_config=megatron_config,
            optim_config=optim_config,
            override_model_config=self.config.model.get('override_config', {}),
            enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False))
        
        # Initialize checkpoint manager
        self.checkpoint_manager = MegatronCheckpointManager(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            save_dir=self.config.checkpoint.save_dir,
            prefix=f"{self.config.model.name}-sft",
            load_directly=True)

        log_gpu_memory_usage('After model & optimizer init', logger=logger)
        return True

    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    def train_step(self, data: DataProto):
        """
        Perform a training step on a batch of data.
        
        Args:
            data (DataProto): a DataProto containing keys
                ``input_ids``: tensor of shape [batch_size, sequence_length].
                ``attention_mask``: tensor of shape [batch_size, sequence_length].
                ``labels``: tensor of shape [batch_size, sequence_length]. 
                    Contains -100 for tokens that should not be included in loss computation.
        
        Returns:
            Dict: Dictionary with loss information.
        """
        # Ensure batch is contiguous
        data.batch = data.batch.contiguous()

        def loss_func(output, data, meta_info):
            """Compute SFT loss from model output."""
            logits = output['logits']
            labels = data['labels']
            
            # Compute loss
            from verl.utils.megatron.tensor_parallel import vocab_parallel_compute_entropy_loss
            loss = vocab_parallel_compute_entropy_loss(logits, labels, ignore_index=-100)
            
            return {'loss': loss}

        # Calculate batch size
        if data.meta_info.get('micro_batch_size', None) is not None:
            batch_size = data.meta_info['micro_batch_size']
        else:
            batch_size = self.config.training.micro_batch_size_per_gpu
        
        # Get the total number of micro batches
        total_micro_batches = self.config.training.gradient_accumulation_steps
        
        # Process batch
        self.optimizer.zero_grad()
        
        # Forward and backward pass
        outputs = self.forward_backward_batch(data, loss_func=loss_func, forward_only=False)
        
        # Only last pp rank has the loss
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            loss = torch.mean(torch.stack([o.get('loss', torch.tensor(0.0, device='cuda')) for o in outputs]))
        else:
            loss = torch.tensor(0.0, device='cuda')
        
        # Broadcast loss to all ranks
        torch.distributed.broadcast(loss, 
                                   mpu.get_pipeline_model_parallel_last_rank(),
                                   group=mpu.get_pipeline_model_parallel_group())
        
        # Update weights
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        # Clear cache to prevent OOM
        torch.cuda.empty_cache()
        
        return {'loss': loss.item()}

    def forward_backward_batch(self, data: DataProto, loss_func=None, forward_only=False):
        """
        Forward and backward pass for a batch of data.
        
        Args:
            data (DataProto): Input data.
            loss_func: Function to compute loss.
            forward_only (bool): If True, only perform forward pass.
            
        Returns:
            List: Outputs from the forward pass.
        """
        from verl.utils.py_functional import append_to_dict
        from verl.utils.torch_functional import broadcast_dict_tensor, split_dict_tensor_into_batches
        from verl.utils.megatron.pipeline_parallel import (
            compute_transformers_input_shapes, 
            make_batch_generator
        )
        from megatron.core.pipeline_parallel import get_forward_backward_func
        
        # Ensure data is broadcast to all pp ranks
        broadcast_dict_tensor(data.batch,
                             src=mpu.get_pipeline_model_parallel_last_rank(),
                             group=mpu.get_pipeline_model_parallel_group())
        
        # Convert attention mask to bool
        data.batch['attention_mask'] = data.batch['attention_mask'].to(bool)
        
        # Get batch size
        if data.meta_info.get('micro_batch_size', None) is not None:
            batch_size = data.meta_info['micro_batch_size']
        else:
            batch_size = self.config.training.micro_batch_size_per_gpu
        
        # Create a function wrapper for the forward step
        def forward_step(batch_iter, model):
            curr_batch = next(batch_iter)
            # Extract the input data
            input_ids = curr_batch['input_ids']
            attention_mask = curr_batch['attention_mask']
            position_ids = curr_batch.get('position_ids', None)
            
            # If position_ids is None, generate them
            if position_ids is None:
                position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
                position_ids = position_ids.masked_fill(~attention_mask, 0)
            
            # Forward pass
            outputs = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           position_ids=position_ids,
                           return_dict=True)
            
            # Apply loss function if provided
            if loss_func is not None:
                loss_output = loss_func(outputs, curr_batch, data.meta_info)
                outputs.update(loss_output)
            
            return outputs
        
        # Create micro batches
        micro_batches = split_dict_tensor_into_batches(data.batch, batch_size=batch_size)
        batch_generator = make_batch_generator(data_iterator=iter(micro_batches))
        
        # Use Megatron's forward-backward function
        (
            input_tensor_shape, 
            output_tensor_shape, 
            data_type
        ) = compute_transformers_input_shapes(
            batch_size=batch_size,
            sequence_length=data.batch['input_ids'].size(1),
            hidden_size=self.model_config.hidden_size,
            sequence_parallel=self.config.model.megatron.sequence_parallel
        )
        
        forward_backward_func = get_forward_backward_func()
        
        # Execute the forward and backward pass
        output = forward_backward_func(
            forward_step_func=forward_step,
            batch_generator=batch_generator,
            model=self.model,
            forward_only=forward_only,
            tensor_shape=input_tensor_shape,
            dtype=data_type,
            sequence_parallel=self.config.model.megatron.sequence_parallel,
            enable_autocast=False,
            custom_sync_context_handler=None,
            custom_sync_function=None,
            sequence_parallel_enabled=self.config.model.megatron.sequence_parallel,
            ddp_config=None,
            grad_scaler=None,
            param_sync_func=None,
            param_sync_func_args=None,
            custom_sync_all_group_list=None,
            is_final_backward=True,
            return_tensors=True,
            async_grad_allreduce=False,
            custom_backward_all_group_list=None,
            finalize_model_grads_func=finalize_model_grads
        )
        
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, checkpoint_path, hdfs_path=None, del_local_after_load=True):
        """Load checkpoint from local or hdfs path."""
        if self.checkpoint_manager is None:
            raise ValueError("Checkpoint manager not initialized. Call init_model first.")
        return self.checkpoint_manager.load_checkpoint(checkpoint_path, hdfs_path, del_local_after_load)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, checkpoint_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        """Save checkpoint to local and optionally to hdfs."""
        if self.checkpoint_manager is None:
            raise ValueError("Checkpoint manager not initialized. Call init_model first.")
        return self.checkpoint_manager.save_checkpoint(
            checkpoint_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def validate(self, data: DataProto):
        """
        Validate the model on a dataset.
        
        Args:
            data (DataProto): Validation data.
            
        Returns:
            Dict: Validation metrics.
        """
        with torch.no_grad():
            def loss_func(output, data, meta_info):
                logits = output['logits']
                labels = data['labels']
                
                # Compute loss
                from verl.utils.megatron.tensor_parallel import vocab_parallel_compute_entropy_loss
                loss = vocab_parallel_compute_entropy_loss(logits, labels, ignore_index=-100)
                
                return {'loss': loss}
            
            outputs = self.forward_backward_batch(data, loss_func=loss_func, forward_only=True)
            
            # Only last pp rank has the loss
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                loss = torch.mean(torch.stack([o.get('loss', torch.tensor(0.0, device='cuda')) for o in outputs]))
            else:
                loss = torch.tensor(0.0, device='cuda')
            
            # Broadcast loss to all ranks
            torch.distributed.broadcast(loss, 
                                       mpu.get_pipeline_model_parallel_last_rank(),
                                       group=mpu.get_pipeline_model_parallel_group())
            
            return {'val_loss': loss.item()}

    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    def generate(self, prompts: DataProto):
        """
        Generate text from prompts using the model.
        
        Args:
            prompts (DataProto): DataProto containing prompts.
            
        Returns:
            Dict: Generated text outputs.
        """
        from verl.utils.torch_functional import broadcast_dict_tensor
        
        # Ensure data is broadcast to all pp ranks
        broadcast_dict_tensor(prompts.batch,
                             src=mpu.get_pipeline_model_parallel_last_rank(),
                             group=mpu.get_pipeline_model_parallel_group())
        
        # Set model to eval mode
        from verl.utils.megatron_utils import get_inference_backend
        for layer in self.model:
            layer.eval()
        
        # Get inference backend
        inference_backend = get_inference_backend(backend="default")
        
        # Generate text
        outputs = inference_backend.generate(
            model=self.model,
            prompts=prompts.batch['input_ids'],
            attention_mask=prompts.batch['attention_mask'],
            generation_config=self.generation_config,
            tokenizer=self.tokenizer
        )
        
        # Return the generated text
        return outputs 