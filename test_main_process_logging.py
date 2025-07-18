#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试主进程日志输出
"""

import sys
import os
import torch
import torch.distributed as dist

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_main_process_check():
    """测试主进程检查函数"""
    print("🧪 测试主进程检查函数...")
    
    # 测试complete_train.py中的is_main_process函数
    from training.complete_train import is_main_process
    
    result = is_main_process()
    print(f"📋 is_main_process() 结果: {result}")
    
    # 测试分布式环境检查
    print(f"📋 dist.is_available(): {dist.is_available()}")
    print(f"📋 dist.is_initialized(): {dist.is_initialized()}")
    
    if dist.is_available() and dist.is_initialized():
        print(f"📋 dist.get_rank(): {dist.get_rank()}")
        print(f"📋 dist.get_world_size(): {dist.get_world_size()}")
    
    return result

def test_logging_functions():
    """测试各个模块的日志函数"""
    print("\n🧪 测试各个模块的日志函数...")
    
    # 测试数据加载器
    try:
        from data.dataloader import create_dataloaders
        print("✅ data.dataloader 导入成功")
    except Exception as e:
        print(f"❌ data.dataloader 导入失败: {e}")
    
    # 测试学习率调度器
    try:
        from training.lr_scheduler import create_lr_scheduler
        print("✅ training.lr_scheduler 导入成功")
    except Exception as e:
        print(f"❌ training.lr_scheduler 导入失败: {e}")
    
    # 测试DeepSpeed训练器
    try:
        from training.deepspeed_trainer import DeepSpeedTrainer
        print("✅ training.deepspeed_trainer 导入成功")
    except Exception as e:
        print(f"❌ training.deepspeed_trainer 导入失败: {e}")
    
    # 测试模型
    try:
        from models.qwen2_5_vl_classify import Qwen2_5_VLForImageClassification
        print("✅ models.qwen2_5_vl_classify 导入成功")
    except Exception as e:
        print(f"❌ models.qwen2_5_vl_classify 导入失败: {e}")

def test_distributed_context():
    """测试分布式上下文"""
    print("\n🧪 测试分布式上下文...")
    
    try:
        from training.utils.distributed import DistributedContext
        dist_ctx = DistributedContext()
        print("✅ DistributedContext 创建成功")
        print(f"📋 is_main_process: {dist_ctx.is_main_process}")
        print(f"📋 rank: {dist_ctx.rank}")
        print(f"📋 world_size: {dist_ctx.world_size}")
        print(f"📋 device: {dist_ctx.device}")
    except Exception as e:
        print(f"❌ DistributedContext 创建失败: {e}")

def main():
    """主函数"""
    print("🚀 测试主进程日志输出")
    print("=" * 50)
    
    # 测试主进程检查
    is_main = test_main_process_check()
    
    # 测试日志函数
    test_logging_functions()
    
    # 测试分布式上下文
    test_distributed_context()
    
    print("\n" + "=" * 50)
    if is_main:
        print("✅ 当前为主进程，所有日志都会正常输出")
    else:
        print("ℹ️ 当前为非主进程，日志输出会被抑制")

if __name__ == "__main__":
    main() 