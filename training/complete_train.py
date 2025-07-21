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

# 🔥 修复：强制设置NCCL_NTHREADS，避免警告
os.environ['NCCL_NTHREADS'] = '64'  # 强制设置为64（32的倍数）
print(f"🔧 在complete_train.py中强制设置 NCCL_NTHREADS={os.environ['NCCL_NTHREADS']}")

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.dataloader import create_dataloaders
from models.qwen2_5_vl_classify import Qwen2_5_VLForImageClassification
from training.deepspeed_trainer import DeepSpeedTrainer
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
    # 获取损失函数配置
    loss_config = config.get('loss', {'type': 'cross_entropy'})
    
    # 获取多数据集配置
    dataset_configs = config.get('datasets', {}).get('dataset_configs', {})
    enable_logits_masking = config.get('datasets', {}).get('enable_logits_masking', True)
    
    # 只在主进程中打印关键信息
    if is_main_process():
        print(f"🎯 损失函数: {loss_config.get('type', 'cross_entropy')}")
        if dataset_configs:
            print(f"🗂️ 数据集: {len(dataset_configs)} 个")
    
    # 创建模型
    model = Qwen2_5_VLForImageClassification(
        pretrained_model_name=config['model']['pretrained_name'],
        num_labels=config['model']['num_labels'],
        loss_config=loss_config,
        dataset_configs=dataset_configs,
        enable_logits_masking=enable_logits_masking
    )
    
    if is_main_process():
        print(f"✅ 模型创建完成")
    return model

def setup_data(config):
    """设置数据加载器"""
    # 🔥 修复：确保在创建DataLoader前分布式已初始化
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()):
        print("⚠️ 警告：DataLoader创建时分布式环境未初始化")
        print("   这可能导致batch size计算不准确")
    
    # 创建数据加载器 - 只传递config参数
    train_loader, val_loader = create_dataloaders(config)
    
    # 只在主进程中打印关键信息
    if is_main_process():
        print("✅ 数据加载器创建完成")
        print(f"  • 训练集: {len(train_loader.dataset):,} 样本")
        print(f"  • 验证集: {len(val_loader.dataset):,} 样本")
    
    return train_loader, val_loader



def main():
    """主训练函数"""
    args = parse_args()
    
    # 设置随机种子
    set_random_seeds(args.seed)
    
    # 🔥 修复：添加分布式初始化，确保分布式环境正确设置
    # 设置端口配置，避免端口冲突
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29501'  # 使用29501端口，避免29500冲突
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    
    # 初始化分布式环境 (DeepSpeed会处理这个)
    deepspeed.init_distributed()
    
    if is_main_process():
        print("✅ 分布式环境初始化完成")
    
    # 只在主进程中加载和准备配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 验证并设置DeepSpeed配置
    if hasattr(args, 'deepspeed_config') and args.deepspeed_config:
        # 验证DeepSpeed配置文件是否存在
        if not os.path.exists(args.deepspeed_config):
            raise FileNotFoundError(f"DeepSpeed配置文件不存在: {args.deepspeed_config}")
        
        # 将DeepSpeed配置添加到config中
        config['deepspeed'] = args.deepspeed_config
    else:
        raise ValueError("DeepSpeed配置文件未指定！请使用--deepspeed_config参数指定配置文件")
    
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
    
    # 创建DeepSpeed训练器
    trainer = DeepSpeedTrainer(config)
    
    # 设置模型和相关组件（优化器和调度器会在DeepSpeed初始化时创建）
    trainer.setup_model(model, train_loader, val_loader, None, None)
    
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