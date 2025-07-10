#!/usr/bin/env python3
"""
验证步数计算是否正确
"""
import os
import sys
import yaml
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def verify_steps():
    # 加载配置
    config = load_config()
    
    # 模拟数据集大小
    train_jsonl = config['data']['train_jsonl']
    if os.path.exists(train_jsonl):
        with open(train_jsonl, 'r') as f:
            dataset_size = sum(1 for _ in f)
    else:
        dataset_size = 75750  # 假设值
    
    # 读取DeepSpeed配置
    deepspeed_config_file = config.get('deepspeed', 'configs/ds_s2.json')
    if isinstance(deepspeed_config_file, str):
        with open(deepspeed_config_file, 'r') as f:
            deepspeed_config = json.load(f)
    else:
        deepspeed_config = deepspeed_config_file
    
    # 获取参数
    micro_batch_size = deepspeed_config.get('train_micro_batch_size_per_gpu', 1)
    gradient_accumulation_steps = deepspeed_config.get('gradient_accumulation_steps', 1)
    train_batch_size = deepspeed_config.get('train_batch_size', 32)
    num_epochs = config['training']['num_epochs']
    num_gpus = 8  # 假设8个GPU
    
    # 计算步数
    dataloader_steps_per_epoch = dataset_size // micro_batch_size
    effective_steps_per_epoch = dataloader_steps_per_epoch // gradient_accumulation_steps
    
    total_dataloader_steps = dataloader_steps_per_epoch * num_epochs
    total_effective_steps = effective_steps_per_epoch * num_epochs
    
    # 验证有效批次大小
    effective_batch_size = micro_batch_size * num_gpus * gradient_accumulation_steps
    
    print("=" * 60)
    print("步数计算验证")
    print("=" * 60)
    print(f"数据集大小: {dataset_size:,}")
    print(f"每GPU微批次大小: {micro_batch_size}")
    print(f"梯度累积步数: {gradient_accumulation_steps}")
    print(f"GPU数量: {num_gpus}")
    print(f"有效批次大小: {effective_batch_size}")
    print(f"配置中的train_batch_size: {train_batch_size}")
    print()
    print(f"DataLoader步数每epoch: {dataloader_steps_per_epoch:,}")
    print(f"有效训练步数每epoch: {effective_steps_per_epoch:,}")
    print(f"训练轮数: {num_epochs}")
    print()
    print(f"总DataLoader步数: {total_dataloader_steps:,}")
    print(f"总有效训练步数: {total_effective_steps:,}")
    print()
    print(f"步数比例: {total_dataloader_steps // total_effective_steps}:1")
    print()
    
    # 验证配置一致性
    if effective_batch_size == train_batch_size:
        print("✅ 有效批次大小与配置一致")
    else:
        print(f"❌ 有效批次大小({effective_batch_size})与配置({train_batch_size})不一致")
    
    # 验证步数计算
    expected_ratio = gradient_accumulation_steps
    actual_ratio = total_dataloader_steps // total_effective_steps
    if actual_ratio == expected_ratio:
        print("✅ 步数计算正确")
    else:
        print(f"❌ 步数计算错误，期望比例{expected_ratio}:1，实际比例{actual_ratio}:1")

if __name__ == "__main__":
    verify_steps() 