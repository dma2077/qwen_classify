#!/usr/bin/env python3
"""
训练性能基准测试脚本
"""

import os
import sys
import time
import yaml
import torch

# 设置环境变量
os.environ['NCCL_NTHREADS'] = '64'

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def benchmark_dataloader():
    """基准测试数据加载器性能"""
    print("⚡ 开始数据加载器性能测试...")
    
    # 加载配置
    config_file = "configs/fast_eval_config.yaml"
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 准备配置
    from training.utils.config_utils import prepare_config
    config = prepare_config(config)
    
    # 创建数据加载器
    from data.dataloader import create_dataloaders
    
    start_time = time.time()
    train_loader, val_loader = create_dataloaders(config)
    creation_time = time.time() - start_time
    
    print(f"✅ 数据加载器创建: {creation_time:.2f}s")
    print(f"📊 训练数据集大小: {len(train_loader.dataset)}")
    print(f"📊 批次数: {len(train_loader)}")
    
    # 测试数据加载速度
    print("🔥 测试数据加载速度...")
    
    batch_times = []
    total_start = time.time()
    
    for i, batch in enumerate(train_loader):
        if i >= 10:  # 只测试前10个batch
            break
        
        batch_start = time.time()
        # 简单处理batch数据
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                _ = value.shape
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        if i < 5:  # 只打印前5个
            print(f"  Batch {i+1}: {batch_time:.3f}s")
    
    total_time = time.time() - total_start
    avg_batch_time = sum(batch_times) / len(batch_times)
    
    print(f"📊 平均batch时间: {avg_batch_time:.3f}s")
    print(f"📊 总时间 (10 batch): {total_time:.1f}s")
    print(f"📊 预估10步时间: {avg_batch_time * 10:.1f}s")
    
    # 性能评估
    if avg_batch_time > 3.0:
        print("⚠️  数据加载可能存在性能问题")
    elif avg_batch_time > 1.0:
        print("🔶 数据加载性能一般")
    else:
        print("✅ 数据加载性能良好")
    
    return avg_batch_time

def benchmark_simple_forward():
    """基准测试简单前向传播"""
    print("\n⚡ 测试简单模型前向传播...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"🔧 GPU: {torch.cuda.get_device_name()}")
        print(f"🔧 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 创建简单的测试tensor
    batch_size = 8
    seq_length = 512
    hidden_size = 768
    
    # 测试tensor创建和移动时间
    start_time = time.time()
    test_tensor = torch.randn(batch_size, seq_length, hidden_size)
    if torch.cuda.is_available():
        test_tensor = test_tensor.to(device)
        torch.cuda.synchronize()
    creation_time = time.time() - start_time
    
    print(f"📊 Tensor创建和移动时间: {creation_time:.3f}s")
    
    # 测试简单计算
    start_time = time.time()
    for _ in range(10):
        result = torch.matmul(test_tensor, test_tensor.transpose(-1, -2))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    compute_time = (time.time() - start_time) / 10
    
    print(f"📊 平均计算时间: {compute_time:.3f}s")
    
    return creation_time, compute_time

if __name__ == "__main__":
    print("🚀 开始训练性能基准测试")
    print("=" * 50)
    
    # 数据加载器基准测试
    avg_batch_time = benchmark_dataloader()
    
    # 简单计算基准测试
    creation_time, compute_time = benchmark_simple_forward()
    
    print("\n" + "=" * 50)
    print("📊 性能基准测试总结:")
    print(f"  • 数据加载: {avg_batch_time:.3f}s/batch")
    print(f"  • Tensor操作: {creation_time:.3f}s")
    print(f"  • 计算操作: {compute_time:.3f}s")
    
    # 估算总体性能
    estimated_step_time = avg_batch_time + creation_time + compute_time * 5  # 估算
    print(f"  • 估算步骤时间: {estimated_step_time:.1f}s")
    
    if estimated_step_time > 30:
        print("❌ 性能存在严重问题")
    elif estimated_step_time > 15:
        print("⚠️  性能需要优化")
    elif estimated_step_time > 7:
        print("🔶 性能一般")
    else:
        print("✅ 性能良好") 