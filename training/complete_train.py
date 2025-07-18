#!/usr/bin/env python3
"""
完整的Qwen2.5-VL图像分类训练脚本
包含FlashAttention、DeepSpeed、WandB监控、性能优化
"""

import os
import sys
import argparse
import yaml
import deepspeed
import torch
import numpy as np
from pathlib import Path

# 🔥 设置FlashAttention环境变量
os.environ["FLASH_ATTENTION_FORCE_ENABLE"] = "1"
os.environ["FLASH_ATTENTION_2"] = "1"

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.dataloader import create_dataloaders
from models.qwen2_5_vl_classify import Qwen2_5_VLForImageClassification
from optimizer.optimizer import create_optimizer
from training.deepspeed_trainer import DeepSpeedTrainer
from training.lr_scheduler import create_lr_scheduler
from training.utils.config_utils import prepare_config

def is_main_process():
    """检查是否为主进程"""
    try:
        import torch.distributed as dist
        return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
    except ImportError:
        return True

def set_random_seeds(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Qwen2.5-VL图像分类完整训练")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--local_rank", type=int, default=-1, help="本地进程排名")
    parser.add_argument("--resume_from", type=str, help="恢复训练的检查点路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # 支持DeepSpeed参数（包括--deepspeed_config）
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

def setup_model(config):
    """设置模型"""
    if is_main_process():
        print("🔧 设置模型...")
    
    # 获取损失函数配置
    loss_config = config.get('loss', {'type': 'cross_entropy'})
    
    # 获取多数据集配置
    dataset_configs = config.get('datasets', {}).get('dataset_configs', {})
    enable_logits_masking = config.get('datasets', {}).get('enable_logits_masking', True)
    
    # 只在主进程中打印配置信息
    if is_main_process():
        print(f"🎯 损失函数: {loss_config.get('type', 'cross_entropy')}")
        if loss_config.get('type') != 'cross_entropy':
            print(f"  损失函数参数: {loss_config}")
        
        if dataset_configs:
            print(f"🗂️ 多数据集模式:")
            print(f"  • 数据集数量: {len(dataset_configs)}")
            print(f"  • Logits Masking: {'启用' if enable_logits_masking else '禁用'}")
            for dataset_name, dataset_config in dataset_configs.items():
                num_classes = dataset_config.get('num_classes', 'N/A')
                print(f"  • {dataset_name}: {num_classes} classes")
    
    # 创建模型
    model = Qwen2_5_VLForImageClassification(
        pretrained_model_name=config['model']['pretrained_name'],
        num_labels=config['model']['num_labels'],
        loss_config=loss_config,
        dataset_configs=dataset_configs,
        enable_logits_masking=enable_logits_masking
    )
    
    if is_main_process():
        print(f"✅ 模型创建完成: {config['model']['pretrained_name']}")
    return model

def setup_data(config):
    """设置数据加载器"""
    if is_main_process():
        print("🔧 设置数据加载器...")
    
    # 创建数据加载器 - 只传递config参数
    train_loader, val_loader = create_dataloaders(config)
    
    # 只在主进程中打印信息
    if is_main_process():
        # 获取数据配置用于打印信息
        data_config = config.get('data', {})
        training_config = config.get('training', {})
        
        print(f"✅ 数据加载器创建完成")
        print(f"  • 训练集: {len(train_loader.dataset)} 样本")
        print(f"  • 验证集: {len(val_loader.dataset)} 样本")
        
        # 从DeepSpeed配置中获取批次大小
        if 'deepspeed' in config:
            if isinstance(config['deepspeed'], str):
                import json
                with open(config['deepspeed'], 'r') as f:
                    deepspeed_config = json.load(f)
            else:
                deepspeed_config = config['deepspeed']
            batch_size = deepspeed_config.get('train_micro_batch_size_per_gpu', 1)
        else:
            batch_size = training_config.get('batch_size', 8)
        
        print(f"  • 批次大小: {batch_size}")
        print(f"  • Worker数量: {training_config.get('num_workers', 16)}")
    
    return train_loader, val_loader

def setup_optimizer_and_scheduler(model, config):
    """设置优化器和学习率调度器"""
    if is_main_process():
        print("🔧 设置优化器和学习率调度器...")
    
    # 创建优化器 - 只传递model和config
    optimizer = create_optimizer(model, config)
    
    # 创建学习率调度器 - 需要config和steps_per_epoch
    # 这里先创建一个临时的steps_per_epoch，后续会在trainer中更新
    temp_steps_per_epoch = 1000  # 临时值，会在trainer中更新
    lr_scheduler = create_lr_scheduler(optimizer, config, temp_steps_per_epoch)
    
    # 只在主进程中打印信息
    if is_main_process():
        # 获取配置信息用于打印
        training_config = config.get('training', {})
        lr_config = training_config.get('lr_scheduler', {})
        
        print(f"✅ 优化器和调度器创建完成")
        print(f"  • 学习率: {training_config.get('lr', 1e-5)}")
        print(f"  • 权重衰减: {training_config.get('weight_decay', 0.01)}")
        print(f"  • 预热步数: {training_config.get('warmup_steps', 100)}")
        print(f"  • 调度器类型: {lr_config.get('type', 'cosine')}")
    
    return optimizer, lr_scheduler

def main():
    """主训练函数"""
    args = parse_args()
    
    # 设置随机种子
    set_random_seeds(args.seed)
    
    # 只在主进程中加载和准备配置
    if is_main_process():
        print("📋 加载配置文件...")
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 验证并设置DeepSpeed配置
    if is_main_process():
        print(f"🔧 使用命令行指定的DeepSpeed配置: {args.deepspeed_config}")
    
    # 验证DeepSpeed配置文件是否存在
    if not os.path.exists(args.deepspeed_config):
        raise FileNotFoundError(f"DeepSpeed配置文件不存在: {args.deepspeed_config}")
    
    # 将DeepSpeed配置添加到config中
    config['deepspeed'] = args.deepspeed_config
    
    config = prepare_config(config)
    
    # 创建输出目录
    output_dir = config.get('training', {}).get('output_dir', './outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    if is_main_process():
        print(f"📁 输出目录: {output_dir}")
    
    # 设置模型
    model = setup_model(config)
    
    # 设置数据加载器
    train_loader, val_loader = setup_data(config)
    
    # 设置优化器和调度器
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model, config)
    
    # 创建DeepSpeed训练器
    if is_main_process():
        print("🔧 创建DeepSpeed训练器...")
    trainer = DeepSpeedTrainer(config)
    
    # 设置模型和相关组件
    trainer.setup_model(model, train_loader, val_loader, optimizer, lr_scheduler)
    
    # 如果指定了恢复检查点
    if args.resume_from and is_main_process():
        print(f"🔄 从检查点恢复训练: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
    
    # 开始训练
    if is_main_process():
        print("🚀 开始训练...")
    try:
        trainer.train()
        if is_main_process():
            print("🎉 训练完成!")
    except KeyboardInterrupt:
        if is_main_process():
            print("⚠️ 训练被用户中断")
    except Exception as e:
        if is_main_process():
            print(f"❌ 训练过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
        raise e

if __name__ == "__main__":
    main() 