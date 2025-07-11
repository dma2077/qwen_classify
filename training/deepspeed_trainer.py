import os
import time
import json
import torch
import deepspeed
from tqdm import tqdm
from transformers import AutoProcessor
from .utils.model_utils import save_hf_model
from .utils.distributed import DistributedContext
from .utils.monitor import TrainingMonitor, make_json_serializable
from .utils.evaluation import evaluate_model

class DeepSpeedTrainer:
    def __init__(self, config):
        # 假设配置已经通过prepare_config处理过
        self.config = config
        self.dist_ctx = DistributedContext()
        self.monitor = TrainingMonitor(self.config['output_dir'])
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.lr_scheduler = None
        self.current_step = 0
        self.current_epoch = 0
        
    def setup_model(self, model, train_loader, val_loader, optimizer, lr_scheduler):
        """设置模型和相关组件"""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        # 初始化DeepSpeed
        self.model, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=self.config['deepspeed']
        )
        
        self.dist_ctx.print_info()
        self.dist_ctx.print_main(f"模型初始化完成，设备: {self.dist_ctx.device}")
        
    def save_checkpoint(self, step):
        """保存检查点"""
        checkpoint_dir = os.path.join(self.config['output_dir'], f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存训练信息
        training_info = {
            'step': step,
            'epoch': self.current_epoch,
            'config': self.config,
            'timestamp': time.time()
        }
        
        with open(os.path.join(checkpoint_dir, 'training_info.json'), 'w') as f:
            # 使用make_json_serializable确保所有数据都可以序列化
            json.dump(make_json_serializable(training_info), f, indent=2)
        
        # 保存DeepSpeed格式（可选）
        if self.config.get('save_deepspeed_format', True):
            deepspeed_dir = os.path.join(checkpoint_dir, 'deepspeed')
            self.model.save_checkpoint(deepspeed_dir)
            self.dist_ctx.print_main(f"DeepSpeed检查点保存到: {deepspeed_dir}")
        
        # 保存HuggingFace格式（可选）
        if self.config.get('save_hf_format', True):
            if self.dist_ctx.is_main_process:
                hf_dir = save_hf_model(self.model, self.config, checkpoint_dir)
                if hf_dir:
                    self.dist_ctx.print_main(f"HuggingFace检查点保存到: {hf_dir}")
        
        self.dist_ctx.barrier()
        
    def evaluate(self):
        """评估模型"""
        self.dist_ctx.print_main("开始评估...")
        eval_loss, eval_accuracy = evaluate_model(self.model, self.val_loader, self.dist_ctx.device)
        self.dist_ctx.print_main(f"验证损失: {eval_loss:.4f}, 准确率: {eval_accuracy:.4f}")
        return eval_loss, eval_accuracy
        
    def train(self):
        """训练模型"""
        self.dist_ctx.print_main("开始训练...")
        self.monitor.start_training()
        
        num_epochs = self.config['training']['num_epochs']
        logging_steps = self.config['logging_steps']
        save_steps = self.config['save_steps']
        eval_steps = self.config['eval_steps']
        
        # 计算有效训练步数（考虑DeepSpeed的分布式训练和梯度累积）
        deepspeed_config = self.config.get('deepspeed', {})
        if isinstance(deepspeed_config, str):
            import json
            with open(deepspeed_config, 'r') as f:
                deepspeed_config = json.load(f)
        
        # 获取DeepSpeed参数
        micro_batch_size_per_gpu = deepspeed_config.get('train_micro_batch_size_per_gpu', 1)
        gradient_accumulation_steps = deepspeed_config.get('gradient_accumulation_steps', 1)
        train_batch_size = deepspeed_config.get('train_batch_size', 32)
        
        # 计算有效训练步数（基于实际的DataLoader长度）
        dataloader_steps_per_epoch = len(self.train_loader)
        effective_steps_per_epoch = dataloader_steps_per_epoch // gradient_accumulation_steps
        total_effective_steps = effective_steps_per_epoch * num_epochs
        
        # 创建进度条（基于有效训练步数）
        pbar = tqdm(total=total_effective_steps, desc="Training Steps", disable=not self.dist_ctx.is_main_process)
        
        # 计算验证信息
        dataset_size = len(self.train_loader.dataset)
        samples_per_gpu = dataloader_steps_per_epoch * micro_batch_size_per_gpu
        
        # 使用更清晰的格式输出训练配置信息
        if self.dist_ctx.is_main_process:
            print("="*80)
            print("🚀 训练配置信息")
            print("="*80)
            print(f"📊 数据集配置:")
            print(f"  • 总数据集大小: {dataset_size:,}")
            print(f"  • 每GPU处理样本数: {samples_per_gpu:,}")
            print(f"📦 批次配置:")
            print(f"  • 每GPU微批次大小: {micro_batch_size_per_gpu}")
            print(f"  • 梯度累积步数: {gradient_accumulation_steps}")
            print(f"  • 总有效批次大小: {train_batch_size}")
            print(f"📈 步数统计:")
            print(f"  • 每GPU DataLoader步数: {dataloader_steps_per_epoch:,}")
            print(f"  • 有效训练步数每epoch: {effective_steps_per_epoch:,}")
            print(f"  • 总有效训练步数: {total_effective_steps:,}")
            print("="*80)
        
        effective_step = 0  # 用于跟踪有效步数
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.model.train()
            
            # 为分布式采样器设置epoch（确保每个epoch的shuffle正确）
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            epoch_loss = 0
            epoch_start_time = time.time()
            
            for batch_idx, batch in enumerate(self.train_loader):
                self.current_step += 1
                
                # 前向传播
                inputs = batch["input_ids"].to(self.dist_ctx.device)
                attention_mask = batch["attention_mask"].to(self.dist_ctx.device)
                pixel_values = batch["pixel_values"].to(self.dist_ctx.device)
                labels = batch["labels"].to(self.dist_ctx.device)
                
                # 添加image_grid_thw参数（如果存在）
                forward_kwargs = {
                    "input_ids": inputs,
                    "attention_mask": attention_mask,
                    "pixel_values": pixel_values,
                    "labels": labels
                }
                
                # 检查并添加image_grid_thw参数
                if "image_grid_thw" in batch:
                    forward_kwargs["image_grid_thw"] = batch["image_grid_thw"].to(self.dist_ctx.device)
                
                outputs = self.model(**forward_kwargs)
                
                loss = outputs.loss
                epoch_loss += loss.item()
                
                # 反向传播
                self.model.backward(loss)
                
                # 获取梯度范数
                grad_norm = self.model.get_global_grad_norm()
                
                # 更新参数
                self.model.step()
                
                # 更新进度条（只在有效步数时更新）
                if self.current_step % gradient_accumulation_steps == 0:
                    effective_step += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                        'epoch': f'{epoch + batch_idx/len(self.train_loader):.2f}'
                    })
                
                # 记录训练指标
                current_lr = self.optimizer.param_groups[0]['lr']
                # 确保grad_norm是float类型，避免JSON序列化错误
                # 处理grad_norm可能为None的情况
                if grad_norm is None:
                    grad_norm_value = 0.0
                elif hasattr(grad_norm, 'item'):
                    grad_norm_value = float(grad_norm.item())
                else:
                    grad_norm_value = float(grad_norm)
                self.monitor.log_step(self.current_step, epoch, loss.item(), grad_norm_value, current_lr)
                
                # 详细日志记录
                if self.current_step % logging_steps == 0:
                    # 使用tqdm.write()来避免与进度条冲突
                    log_message = (
                        f"Step {self.current_step:,} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Grad Norm: {grad_norm_value:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Epoch: {epoch + batch_idx/len(self.train_loader):.2f}"
                    )
                    if self.dist_ctx.is_main_process:
                        pbar.write(log_message)
                
                # 定期评估（基于有效步数）
                if effective_step > 0 and effective_step % eval_steps == 0:
                    # 暂时刷新进度条以避免输出冲突
                    pbar.clear()
                    eval_loss, eval_accuracy = self.evaluate()
                    self.model.train()
                    # 重新显示进度条
                    pbar.refresh()
                
                # 定期保存检查点（基于有效步数）
                if effective_step > 0 and effective_step % save_steps == 0:
                    pbar.clear()
                    self.save_checkpoint(effective_step)
                    pbar.refresh()
            
            # Epoch结束统计
            epoch_time = time.time() - epoch_start_time
            avg_loss = epoch_loss / len(self.train_loader)
            self.monitor.log_epoch(epoch, avg_loss, epoch_time)
            
            # 使用tqdm.write()输出epoch统计信息
            epoch_message = (
                f"📊 Epoch {epoch+1}/{num_epochs} 完成 | "
                f"平均损失: {avg_loss:.4f} | "
                f"耗时: {epoch_time:.2f}秒 | "
                f"有效步数: {effective_step:,}"
            )
            if self.dist_ctx.is_main_process:
                pbar.write(epoch_message)
        
        pbar.close()
        
        # 训练结束前进行最终评估
        if self.dist_ctx.is_main_process:
            print("\n🎯 训练即将完成，进行最终评估...")
        eval_loss, eval_accuracy = self.evaluate()
        
        # 保存最终检查点
        if self.dist_ctx.is_main_process:
            print(f"💾 保存最终检查点...")
        self.save_checkpoint(effective_step)
        
        if self.dist_ctx.is_main_process:
            print("🎉 训练完成！")
            print(f"📊 最终评估结果 - 损失: {eval_loss:.4f}, 准确率: {eval_accuracy:.4f}")
        self.monitor.save_logs()
        
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if os.path.exists(checkpoint_path):
            self.model.load_checkpoint(checkpoint_path)
            self.dist_ctx.print_main(f"检查点加载成功: {checkpoint_path}")
        else:
            self.dist_ctx.print_main(f"检查点不存在: {checkpoint_path}")
            
    def get_training_stats(self):
        """获取训练统计信息"""
        return self.monitor.get_avg_metrics() 