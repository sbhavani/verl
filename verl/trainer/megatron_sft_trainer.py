"""
Megatron SFT Trainer - A trainer that utilizes the MegatronSFTWorker for supervised fine-tuning.
"""

import os
import time
import logging
import ray
import torch
import torch.distributed

import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from tensordict import TensorDict
from torch.utils.data import DataLoader, DistributedSampler
from verl.workers.megatron_sft import MegatronSFTWorker
from verl.utils.dataset import SFTDataset
from verl.utils.fs import copy_to_local
from verl.utils.tracking import Tracking
from verl.utils.import_utils import load_extern_type
from verl import DataProto
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_SFT_LOGGING_LEVEL', 'WARN'))


class MegatronSFTTrainer:
    """
    A trainer that uses the MegatronSFTWorker for supervised fine-tuning.
    """
    
    def __init__(self, config: DictConfig):
        """Initialize the trainer.
        
        Args:
            config (DictConfig): The configuration object.
        """
        self.config = config
        self.worker = None
        
        # Initialize tracking
        self.tracking = Tracking(
            project_name=config.trainer.project_name,
            experiment_name=config.trainer.experiment_name,
            config=config)
        
        # Initialize worker
        self._init_worker()
        
        # Initialize datasets and dataloaders
        self._init_data()
        
        # Counter for global steps
        self.global_step = 0
        
        # Calculate total steps based on data size
        train_size = len(self.train_dataset)
        self.total_steps = min(
            self.config.trainer.max_steps,
            train_size // (self.config.data.train_batch_size * torch.distributed.get_world_size())
        )
        
    def _init_worker(self):
        """Initialize the SFT worker."""
        self.worker = MegatronSFTWorker(config=self.config)
        self.worker.init_model()
        
    def _init_data(self):
        """Initialize datasets and dataloaders."""
        # Build dataset
        if self.config.data.custom_cls.get("path", None):
            dataset_cls = load_extern_type(
                self.config.data.custom_cls.path, 
                self.config.data.custom_cls.name)
        else:
            dataset_cls = SFTDataset
            
        # Initialize tokenizer 
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)
        from verl.utils import hf_tokenizer
        tokenizer = hf_tokenizer(local_model_path, trust_remote_code=self.config.model.trust_remote_code)
        
        # Create datasets
        self.train_dataset = dataset_cls(
            parquet_files=self.config.data.train_files,
            tokenizer=tokenizer,
            config=self.config.data)
        
        self.val_dataset = dataset_cls(
            parquet_files=self.config.data.val_files,
            tokenizer=tokenizer,
            config=self.config.data)
        
        # Create samplers
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        
        self.train_sampler = DistributedSampler(
            self.train_dataset,
            shuffle=True,
            num_replicas=world_size,
            rank=rank,
            drop_last=True)
        
        self.val_sampler = DistributedSampler(
            self.val_dataset,
            shuffle=False,
            num_replicas=world_size,
            rank=rank,
            drop_last=True)
        
        # Create dataloaders
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True)
        
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True)
    
    def _validate(self):
        """Perform validation."""
        val_losses = []
        
        # Set model to eval mode
        self.worker.model.eval()
        
        for batch in tqdm(self.val_dataloader, disable=torch.distributed.get_rank() != 0):
            data_proto = DataProto(batch=TensorDict(batch, batch_size=batch['input_ids'].shape[0]))
            
            # Validate
            metrics = self.worker.validate(data_proto)
            val_losses.append(metrics['val_loss'])
        
        # Average validation loss
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
        
        # Set model back to train mode
        self.worker.model.train()
        
        return avg_val_loss
    
    def train(self):
        """Train the model."""
        # Resume from checkpoint if specified
        if self.config.checkpoint.resume_from_checkpoint:
            self.worker.load_checkpoint(checkpoint_path=self.config.checkpoint.resume_from_checkpoint)
            # TODO: Resume step count
        
        # Set model to train mode
        for layer in self.worker.model:
            layer.train()
        
        # Training loop
        if torch.distributed.get_rank() == 0:
            print(f"Starting training for {self.total_steps} steps")
        
        start_time = time.time()
        
        for epoch in range(100):  # Arbitrary large number, we'll stop based on steps
            self.train_sampler.set_epoch(epoch)
            
            for batch in tqdm(self.train_dataloader, disable=torch.distributed.get_rank() != 0):
                # Skip steps if resuming
                self.global_step += 1
                
                # Convert batch to DataProto
                data_proto = DataProto(batch=TensorDict(batch, batch_size=batch['input_ids'].shape[0]))
                
                # Perform training step
                metrics = self.worker.train_step(data_proto)
                
                # Log metrics
                if torch.distributed.get_rank() == 0:
                    self.tracking.log_metrics({
                        "loss": metrics['loss'],
                        "learning_rate": self.worker.optimizer.param_groups[0]['lr']
                    }, step=self.global_step)
                
                # Validate if needed
                if self.global_step % self.config.trainer.val_check_interval == 0:
                    val_loss = self._validate()
                    if torch.distributed.get_rank() == 0:
                        print(f"Step {self.global_step}: Validation loss: {val_loss}")
                        self.tracking.log_metrics({"val_loss": val_loss}, step=self.global_step)
                
                # Save checkpoint if needed
                if (self.global_step % self.config.trainer.save_interval == 0 and 
                    torch.distributed.get_rank() == 0):
                    self.worker.save_checkpoint(
                        checkpoint_path=os.path.join(
                            self.config.checkpoint.save_dir, 
                            f"global_step_{self.global_step}"),
                        global_step=self.global_step,
                        max_ckpt_to_keep=self.config.checkpoint.save_top_k)
                
                # Check if we've reached max steps
                if self.global_step >= self.total_steps:
                    break
            
            # Break if we've reached max steps
            if self.global_step >= self.total_steps:
                break
        
        # Final validation
        final_val_loss = self._validate()
        if torch.distributed.get_rank() == 0:
            print(f"Final validation loss: {final_val_loss}")
            self.tracking.log_metrics({"val_loss": final_val_loss}, step=self.global_step)
        
        # Final checkpoint
        if torch.distributed.get_rank() == 0:
            self.worker.save_checkpoint(
                checkpoint_path=os.path.join(
                    self.config.checkpoint.save_dir, 
                    f"final_step_{self.global_step}"),
                global_step=self.global_step,
                max_ckpt_to_keep=self.config.checkpoint.save_top_k)
        
        # Report training time
        total_time = time.time() - start_time
        if torch.distributed.get_rank() == 0:
            print(f"Training completed in {total_time:.2f} seconds")
            print(f"Average time per step: {total_time / self.global_step:.2f} seconds")


@hydra.main(config_path='config', config_name='sft_megatron', version_base=None)
def main(config: DictConfig):
    """Main entry point for training."""
    # Initialize distributed training
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
        
    # Create trainer and start training
    trainer = MegatronSFTTrainer(config=config)
    trainer.train()


if __name__ == "__main__":
    main() 
