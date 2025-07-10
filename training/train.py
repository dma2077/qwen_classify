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
    parser.add_argument("--deepspeed_config", type=str, required=True, help="DeepSpeed配置文件路径")
    parser.add_argument("--local_rank", type=int, default=-1, help="本地进程排名")
    parser.add_argument("--resume_from", type=str, help="恢复训练的检查点路径")
    
    # 支持DeepSpeed参数
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def setup_model(config):
    """设置模型"""
    model = Qwen2_5_VLForImageClassification(
        pretrained_model_name=config['model']['pretrained_name'],
        num_labels=config['model']['num_labels']
    )
    
    # 如果有预训练检查点，加载它
    if config.get('pretrained_checkpoint'):
        print(f"加载预训练检查点: {config['pretrained_checkpoint']}")
        checkpoint = torch.load(config['pretrained_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    return model

def print_training_info(config, train_loader, val_loader):
    """打印训练信息"""
    print("=" * 60)
    print("🚀 Qwen2.5-VL食物分类多GPU训练")
    print("=" * 60)
    print(f"📊 训练数据集大小: {len(train_loader.dataset):,}")
    print(f"📊 验证数据集大小: {len(val_loader.dataset):,}")
    print(f"📦 训练批次数: {len(train_loader):,}")
    print(f"📦 验证批次数: {len(val_loader):,}")
    print(f"🎯 类别数量: {config['model']['num_labels']}")
    print(f"🔄 训练轮数: {config['training']['num_epochs']}")
    print(f"📝 日志步数: {config['logging_steps']}")
    print(f"💾 保存步数: {config['save_steps']}")
    print(f"🔍 评估步数: {config['eval_steps']}")
    print(f"📁 输出目录: {config['output_dir']}")
    print("=" * 60)

def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置DeepSpeed配置路径
    config['deepspeed'] = args.deepspeed_config
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 设置模型
    model = setup_model(config)
    
    # 设置数据加载器
    train_loader, val_loader = create_dataloaders(config)
    
    # 打印训练信息
    if args.local_rank <= 0:
        print_training_info(config, train_loader, val_loader)
    
    # 设置优化器
    optimizer = create_optimizer(model, config)
    
    # 设置学习率调度器
    lr_scheduler = create_lr_scheduler(optimizer, config, len(train_loader))
    
    # 创建训练器
    trainer = DeepSpeedTrainer(config)
    
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