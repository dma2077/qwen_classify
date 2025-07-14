#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-VL食物分类多GPU训练脚本
"""

import os
import sys
import argparse
import yaml
import torch
import deepspeed

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import create_dataloaders
from models.qwen2_5_vl_classify import Qwen2_5_VLForImageClassification
from optimizer.optimizer import create_optimizer
from training.deepspeed_trainer import DeepSpeedTrainer
from training.lr_scheduler import create_lr_scheduler

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Qwen2.5-VL食物分类多GPU训练")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--local_rank", type=int, default=-1, help="本地进程排名")
    parser.add_argument("--resume_from", type=str, help="恢复训练的检查点路径")
    
    # 支持DeepSpeed参数 (包含 --deepspeed_config)
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def setup_model(config):
    """设置模型"""
    # 获取损失函数配置
    loss_config = config.get('loss', {'type': 'cross_entropy'})
    
    # 打印损失函数信息（只在主进程）
    try:
        import torch.distributed as dist
        is_distributed = dist.is_available() and dist.is_initialized()
        if is_distributed:
            current_rank = dist.get_rank()
        else:
            current_rank = 0
        should_print = not is_distributed or current_rank == 0
    except:
        should_print = True
    
    if should_print:
        print(f"🎯 使用损失函数: {loss_config.get('type', 'cross_entropy')}")
        if loss_config.get('type') != 'cross_entropy':
            print(f"  损失函数参数: {loss_config}")
    
    model = Qwen2_5_VLForImageClassification(
        pretrained_model_name=config['model']['pretrained_name'],
        num_labels=config['model']['num_labels'],
        loss_config=loss_config
    )
    
    # 如果有预训练检查点，加载它
    if config.get('pretrained_checkpoint'):
        print(f"加载预训练检查点: {config['pretrained_checkpoint']}")
        checkpoint = torch.load(config['pretrained_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    return model

def print_training_info(config, train_loader, val_loader):
    """打印训练信息"""
    # 获取DeepSpeed配置
    deepspeed_config = config.get('deepspeed', {})
    if isinstance(deepspeed_config, str):
        import json
        with open(deepspeed_config, 'r') as f:
            deepspeed_config = json.load(f)
    
    gradient_accumulation_steps = deepspeed_config.get('gradient_accumulation_steps', 1)
    micro_batch_size = deepspeed_config.get('train_micro_batch_size_per_gpu', 1)
    train_batch_size = deepspeed_config.get('train_batch_size', 32)
    
    # 计算有效步数
    dataloader_steps_per_epoch = len(train_loader)
    effective_steps_per_epoch = dataloader_steps_per_epoch // gradient_accumulation_steps
    total_effective_steps = effective_steps_per_epoch * config['training']['num_epochs']
    
    print("=" * 60)
    print("🚀 Qwen2.5-VL食物分类多GPU训练")
    print("=" * 60)
    print(f"📊 训练数据集大小: {len(train_loader.dataset):,}")
    print(f"📊 验证数据集大小: {len(val_loader.dataset):,}")
    print(f"🎯 类别数量: {config['model']['num_labels']}")
    print(f"🔄 训练轮数: {config['training']['num_epochs']}")
    print()
    print("📦 批次配置:")
    print(f"  • 每GPU微批次大小: {micro_batch_size}")
    print(f"  • 梯度累积步数: {gradient_accumulation_steps}")
    print(f"  • 有效批次大小: {train_batch_size}")
    print()
    print("📈 步数统计:")
    print(f"  • DataLoader步数每epoch: {dataloader_steps_per_epoch:,}")
    print(f"  • 有效训练步数每epoch: {effective_steps_per_epoch:,}")
    print(f"  • 总有效训练步数: {total_effective_steps:,}")
    print()
    print("📝 训练配置:")
    print(f"  • 日志步数: {config['logging_steps']}")
    print(f"  • 保存步数: {config['save_steps']}")
    print(f"  • 评估步数: {config['eval_steps']}")
    print(f"  • 输出目录: {config['output_dir']}")
    print("=" * 60)

def main():
    """主函数"""
    args = parse_args()
    
    # 初始化分布式环境 (DeepSpeed会处理这个)
    deepspeed.init_distributed()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置DeepSpeed配置文件路径
    config['deepspeed'] = args.deepspeed_config
    
    # 创建输出目录
    output_dir = config['training']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # 为了兼容性，将output_dir提升到根层级
    config['output_dir'] = output_dir
    
    # 提前准备配置参数（参数映射等）
    from training.utils.config_utils import prepare_config
    config = prepare_config(config)
    
    # 设置模型
    model = setup_model(config)
    
    # 设置数据加载器（现在分布式环境已经初始化）
    train_loader, val_loader = create_dataloaders(config)
    
    # 创建训练器（这里会调用prepare_config）
    trainer = DeepSpeedTrainer(config)
    
    # 打印训练信息（在prepare_config之后）
    if args.local_rank <= 0:
        print_training_info(config, train_loader, val_loader)
    
    # 设置优化器
    optimizer = create_optimizer(model, config)
    
    # 设置学习率调度器
    lr_scheduler = create_lr_scheduler(optimizer, config, len(train_loader))
    
    # 设置训练器
    trainer.setup_model(model, train_loader, val_loader, optimizer, lr_scheduler)
    
    # 如果需要恢复训练
    if args.resume_from:
        print(f"恢复训练从: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main() 