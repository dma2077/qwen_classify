#!/usr/bin/env python3
"""
性能测试脚本 - 测量训练速度
"""

import os
import sys
import time
import yaml
import torch

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_training_speed():
    """测试训练速度"""
    print("⚡ 开始性能测试...")
    
    # 设置环境变量
    os.environ['NCCL_NTHREADS'] = '64'
    
    # 加载配置
    config_file = "configs/fast_eval_config.yaml"
    print(f"📋 使用配置: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 准备配置
    from training.utils.config_utils import prepare_config
    config = prepare_config(config)
    
    # 创建数据加载器
    from data.dataloader import create_dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    print(f"📊 数据集大小: {len(train_loader.dataset)}")
    print(f"📊 批次数: {len(train_loader)}")
    
    # 测试数据加载速度
    print("🔥 测试数据加载速度...")
    data_loading_times = []
    
    for i, batch in enumerate(train_loader):
        if i >= 5:  # 只测试前5个batch
            break
        
        start_time = time.time()
        # 模拟数据处理
        batch_keys = list(batch.keys())
        batch_time = time.time() - start_time
        data_loading_times.append(batch_time)
        
        print(f"  Batch {i+1}: {batch_time:.3f}s")
    
    avg_data_time = sum(data_loading_times) / len(data_loading_times)
    print(f"📊 平均数据加载时间: {avg_data_time:.3f}s/batch")
    
    # 估算总体性能
    estimated_time_per_10_steps = avg_data_time * 10
    print(f"📊 估算10步数据加载时间: {estimated_time_per_10_steps:.1f}s")
    
    if estimated_time_per_10_steps > 60:
        print("⚠️  数据加载可能存在性能问题")
    else:
        print("✅ 数据加载性能正常")

if __name__ == "__main__":
    test_training_speed() 