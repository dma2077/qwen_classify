#!/usr/bin/env python3
"""
分布式初始化调试脚本
"""

import os
import sys
import torch
import deepspeed

def check_environment():
    """检查环境变量"""
    print("🔍 检查环境变量:")
    env_vars = [
        'WORLD_SIZE', 'RANK', 'LOCAL_RANK', 
        'MASTER_ADDR', 'MASTER_PORT',
        'CUDA_VISIBLE_DEVICES', 'NCCL_DEBUG'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'Not Set')
        print(f"  • {var}: {value}")

def check_cuda():
    """检查CUDA环境"""
    print("\n🔍 检查CUDA环境:")
    print(f"  • CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  • CUDA设备数量: {torch.cuda.device_count()}")
        print(f"  • 当前设备: {torch.cuda.current_device()}")
        for i in range(torch.cuda.device_count()):
            print(f"  • GPU {i}: {torch.cuda.get_device_name(i)}")

def test_distributed_init():
    """测试分布式初始化"""
    print("\n🔍 测试分布式初始化:")
    
    try:
        # 设置基本环境变量（如果没有设置）
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29501'
        
        print("  • 尝试调用 deepspeed.init_distributed()...")
        deepspeed.init_distributed()
        print("  ✅ deepspeed.init_distributed() 成功")
        
        # 检查分布式状态
        import torch.distributed as dist
        if dist.is_available():
            print(f"  • torch.distributed 可用: True")
            if dist.is_initialized():
                print(f"  ✅ 分布式已初始化")
                print(f"  • World Size: {dist.get_world_size()}")
                print(f"  • Rank: {dist.get_rank()}")
                print(f"  • Backend: {dist.get_backend()}")
            else:
                print(f"  ❌ 分布式未初始化")
        else:
            print(f"  ❌ torch.distributed 不可用")
            
    except Exception as e:
        print(f"  ❌ 分布式初始化失败: {e}")
        import traceback
        traceback.print_exc()

def test_dataloader_batch_calculation():
    """测试DataLoader batch size计算"""
    print("\n🔍 测试batch size计算:")
    
    # 模拟DeepSpeed配置
    deepspeed_config = {
        "train_batch_size": 256,
        "train_micro_batch_size_per_gpu": 8,
        "gradient_accumulation_steps": 4
    }
    
    micro_batch_size_per_gpu = deepspeed_config.get('train_micro_batch_size_per_gpu', 1)
    total_batch_size = deepspeed_config.get('train_batch_size', 1)
    gradient_accumulation_steps = deepspeed_config.get('gradient_accumulation_steps', 1)
    
    print(f"  • micro_batch_size_per_gpu: {micro_batch_size_per_gpu}")
    print(f"  • total_batch_size: {total_batch_size}")
    print(f"  • gradient_accumulation_steps: {gradient_accumulation_steps}")
    
    # 测试分布式状态
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        num_gpus = dist.get_world_size()
        print(f"  ✅ 从分布式环境获取GPU数量: {num_gpus}")
    else:
        # Fallback计算
        calculated_num_gpus = total_batch_size // (micro_batch_size_per_gpu * gradient_accumulation_steps)
        num_gpus = max(1, calculated_num_gpus)
        print(f"  🔧 从DeepSpeed配置计算GPU数量: {num_gpus}")
        print(f"     计算公式: {total_batch_size} / ({micro_batch_size_per_gpu} × {gradient_accumulation_steps}) = {num_gpus}")
    
    # 计算batch sizes
    train_batch_size = micro_batch_size_per_gpu
    eval_batch_size = micro_batch_size_per_gpu * num_gpus
    
    print(f"  • Training DataLoader batch_size: {train_batch_size}")
    print(f"  • Evaluation DataLoader batch_size: {eval_batch_size}")
    
    # 验证计算
    expected_total = micro_batch_size_per_gpu * num_gpus * gradient_accumulation_steps
    print(f"  • 验证总batch size: {expected_total} (期望: {total_batch_size})")
    if expected_total == total_batch_size:
        print(f"  ✅ batch size计算正确")
    else:
        print(f"  ❌ batch size计算不一致")

if __name__ == "__main__":
    print("🚀 分布式初始化调试")
    print("=" * 60)
    
    check_environment()
    check_cuda()
    test_distributed_init()
    test_dataloader_batch_calculation()
    
    print("\n" + "=" * 60)
    print("🔍 调试完成！请检查上述输出以诊断问题") 