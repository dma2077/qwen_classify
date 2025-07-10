import os
import json
import time
import argparse
import logging
from typing import Dict, Any

import torch
import torch.distributed as dist
import deepspeed
from transformers import AutoProcessor
from deepspeed.ops.adam import FusedAdam

# Import existing project modules
from data.dataloader import build_dataloader
from training.models import load_config, build_model
from optimizer.optimizer import build_optimizer
from training.lr_scheduler import build_scheduler

logger = logging.getLogger(__name__)


class DistributedContext:
    """Distributed training context manager"""
    
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.rank = None
        self.local_rank = None
        self.world_size = None
        
    def setup(self):
        """Setup distributed training environment"""
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # Set device
        torch.cuda.set_device(self.local_rank)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO if self.rank == 0 else logging.WARNING,
            format=f'[Rank {self.rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if self.rank == 0:
            logger.info(f"Distributed training setup: world_size={self.world_size}, rank={self.rank}")
            
        return self


class TrainingMonitor:
    """Simple training monitor for metrics tracking"""
    
    def __init__(self, config: Dict[str, Any], ctx: DistributedContext):
        self.config = config
        self.ctx = ctx
        self.metrics = {}
        self.step_count = 0
        self.total_samples = 0
        self.total_loss = 0.0
        self.start_time = time.time()
        
    def update(self, loss: float, batch_size: int, learning_rate: float, **kwargs):
        """Update training metrics"""
        self.step_count += 1
        self.total_samples += batch_size * self.ctx.world_size
        self.total_loss += loss
        
        # Store current metrics
        self.metrics.update({
            'step': self.step_count,
            'loss': loss,
            'avg_loss': self.total_loss / self.step_count,
            'learning_rate': learning_rate,
            'total_samples': self.total_samples,
            **kwargs
        })
        
    def log(self, force_log: bool = False):
        """Log metrics"""
        log_steps = self.config.get("training", {}).get("logging_steps", 50)
        
        if self.step_count % log_steps == 0 or force_log:
            if self.ctx.rank == 0:
                elapsed_time = time.time() - self.start_time
                samples_per_sec = self.total_samples / elapsed_time if elapsed_time > 0 else 0
                
                log_msg = (
                    f"Step {self.step_count}: "
                    f"Loss={self.metrics.get('loss', 0):.4f}, "
                    f"AvgLoss={self.metrics.get('avg_loss', 0):.4f}, "
                    f"LR={self.metrics.get('learning_rate', 0):.2e}, "
                    f"Samples/sec={samples_per_sec:.2f}"
                )
                
                if 'grad_norm' in self.metrics:
                    log_msg += f", GradNorm={self.metrics['grad_norm']:.4f}"
                    
                logger.info(log_msg)
    
    def save_checkpoint_info(self, output_dir: str):
        """Save training info to checkpoint directory"""
        if self.ctx.rank == 0:
            checkpoint_info = {
                'step': self.step_count,
                'total_samples': self.total_samples,
                'avg_loss': self.total_loss / max(self.step_count, 1),
                'config': self.config
            }
            
            info_path = os.path.join(output_dir, 'training_info.json')
            with open(info_path, 'w') as f:
                json.dump(checkpoint_info, f, indent=2)


def prepare_config(args, ctx, config):
    """Prepare and validate configuration"""
    # Set output directory from args
    config["training"]["output_dir"] = args.output_dir
    
    # Create output directory
    if ctx.rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save config for reproducibility
        config_path = os.path.join(args.output_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Configuration saved to {config_path}")
    
    return config


def evaluate_model(model, eval_loader, ctx):
    """Simple evaluation function"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            # Move batch to device
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            labels = batch['labels']
            
            # Aggregate metrics across all GPUs
            loss_tensor = torch.tensor(loss.item()).cuda()
            correct_tensor = torch.tensor((logits.argmax(dim=-1) == labels).sum().item()).cuda()
            samples_tensor = torch.tensor(labels.size(0)).cuda()
            
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
            
            total_loss += loss_tensor.item()
            total_correct += correct_tensor.item()
            total_samples += samples_tensor.item()
    
    avg_loss = total_loss / len(eval_loader) if len(eval_loader) > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    model.train()
    return {'eval_loss': avg_loss, 'eval_accuracy': accuracy}


def train(args):
    """Main training function"""
    
    # Initialize DeepSpeed distributed training
    deepspeed.init_distributed()
    
    # Load configuration
    config = load_config(args.config_file)
    logger.info(f"Loaded config from {args.config_file}")
    
    # Setup distributed context
    ctx = DistributedContext(args=args, config=config).setup()
    
    # Prepare configuration
    config = prepare_config(args, ctx, config)
    
    # Build dataloaders
    if ctx.rank == 0:
        logger.info("Building dataloaders...")
    
    train_loader = build_dataloader(
        split_file=config["data"]["train_jsonl"],
        pretrained_model_name=config["model"]["pretrained_name"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        shuffle=True,
    )
    
    eval_loader = build_dataloader(
        split_file=config["data"]["val_jsonl"],
        pretrained_model_name=config["model"]["pretrained_name"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        shuffle=False,
    )
    
    # Build model
    if ctx.rank == 0:
        logger.info("Building model...")
    
    # Initialize model with DeepSpeed Zero
    with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config, enabled=False):
        model = build_model(config)
    
    # Build optimizer - use FusedAdam for better performance
    if config.get("training", {}).get("use_fused_adam", True):
        optimizer = FusedAdam(
            model.parameters(),
            lr=config["training"]["lr"],
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=config["training"]["weight_decay"]
        )
    else:
        optimizer = build_optimizer(
            model,
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
        )
    
    # Build learning rate scheduler
    num_steps = len(train_loader) * config["training"]["epochs"]
    lr_scheduler = build_scheduler(
        optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=num_steps,
    )
    
    # Initialize DeepSpeed engine
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )
    
    if ctx.rank == 0:
        logger.info("DeepSpeed engine initialized successfully")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Initialize training monitor
    monitor = TrainingMonitor(config, ctx)
    
    # Set model to training mode
    model.train()
    
    # Training loop
    if ctx.rank == 0:
        logger.info("Starting training...")
    
    for epoch in range(config["training"]["epochs"]):
        if ctx.rank == 0:
            logger.info(f"Starting epoch {epoch + 1}/{config['training']['epochs']}")
        
        epoch_start_time = time.time()
        
        for step, batch in enumerate(train_loader, 1):
            step_start_time = time.time()
            
            # Move batch to device (handled by DeepSpeed automatically for most tensors)
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass using DeepSpeed
            model.backward(loss)
            
            # Optimizer step using DeepSpeed
            model.step()
            
            # Get current learning rate
            current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler else config["training"]["lr"]
            
            # Get gradient norm if available
            grad_norm = None
            if hasattr(model, 'get_global_grad_norm'):
                grad_norm = model.get_global_grad_norm()
            
            # Update monitor
            step_elapsed = time.time() - step_start_time
            monitor.update(
                loss=loss.item(),
                batch_size=batch['labels'].size(0),
                learning_rate=current_lr,
                grad_norm=grad_norm.item() if grad_norm is not None else None,
                step_time=step_elapsed
            )
            
            # Log metrics
            monitor.log()
            
            # Save checkpoint periodically
            save_steps = config.get("training", {}).get("save_steps", 1000)
            if step % save_steps == 0:
                checkpoint_dir = os.path.join(config["training"]["output_dir"], f"checkpoint-{monitor.step_count}")
                model.save_checkpoint(checkpoint_dir)
                monitor.save_checkpoint_info(checkpoint_dir)
                
                if ctx.rank == 0:
                    logger.info(f"Checkpoint saved at step {monitor.step_count}")
        
        # End of epoch evaluation
        if ctx.rank == 0:
            logger.info(f"Epoch {epoch + 1} completed in {time.time() - epoch_start_time:.2f}s")
        
        # Evaluate model
        eval_metrics = evaluate_model(model, eval_loader, ctx)
        if ctx.rank == 0:
            logger.info(f"Evaluation metrics: {eval_metrics}")
        
        # Save epoch checkpoint
        checkpoint_dir = os.path.join(config["training"]["output_dir"], f"checkpoint-epoch-{epoch + 1}")
        model.save_checkpoint(checkpoint_dir)
        monitor.save_checkpoint_info(checkpoint_dir)
        
        if ctx.rank == 0:
            logger.info(f"Epoch checkpoint saved: {checkpoint_dir}")
    
    # Final checkpoint
    final_checkpoint_dir = os.path.join(config["training"]["output_dir"], "final_checkpoint")
    model.save_checkpoint(final_checkpoint_dir)
    monitor.save_checkpoint_info(final_checkpoint_dir)
    
    # Final logging
    monitor.log(force_log=True)
    
    if ctx.rank == 0:
        logger.info("Training completed successfully!")
        logger.info(f"Final checkpoint saved: {final_checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="DeepSpeed Training Script")
    parser.add_argument('--config_file', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for checkpoints and logs')
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="Local rank for distributed training")
    
    # Add DeepSpeed arguments
    parser = deepspeed.add_config_arguments(parser)
    
    args = parser.parse_args()
    
    # Run training
    train(args)


if __name__ == "__main__":
    main() 