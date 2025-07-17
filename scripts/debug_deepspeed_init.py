#!/usr/bin/env python3
"""
DeepSpeed初始化调试脚本
用于诊断DeepSpeed卡住的问题
"""

import os
import sys
import time
import torch
import logging
from typing import Dict, Any
import datetime

# 设置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def check_environment():
    """检查环境配置"""
    print("🔍 检查环境配置...")
    
    # 检查CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 检查环境变量
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"NCCL_DEBUG: {os.environ.get('NCCL_DEBUG', 'Not set')}")
    print(f"NCCL_IB_DISABLE: {os.environ.get('NCCL_IB_DISABLE', 'Not set')}")
    
    # 检查网络
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'Not set')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'Not set')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not set')}")
    print(f"RANK: {os.environ.get('RANK', 'Not set')}")

def test_basic_torch():
    """测试基础PyTorch功能"""
    print("\n🔍 测试基础PyTorch功能...")
    
    try:
        # 测试张量操作
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.mm(x, y)
        print("✅ 基础张量操作正常")
        
        # 测试GPU内存
        memory_allocated = torch.cuda.memory_allocated()
        memory_reserved = torch.cuda.memory_reserved()
        print(f"GPU内存使用: {memory_allocated / 1024**2:.2f}MB 分配, {memory_reserved / 1024**2:.2f}MB 保留")
        
    except Exception as e:
        print(f"❌ 基础PyTorch测试失败: {e}")

def test_nccl():
    """测试NCCL通信"""
    print("\n🔍 测试NCCL通信...")
    
    try:
        import torch.distributed as dist
        
        # 初始化进程组
        print("初始化进程组...")
        dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=30))
        print("✅ NCCL初始化成功")
        
        # 测试通信
        tensor = torch.randn(100, 100).cuda()
        dist.all_reduce(tensor)
        print("✅ NCCL通信测试成功")
        
        # 清理
        dist.destroy_process_group()
        
    except Exception as e:
        print(f"❌ NCCL测试失败: {e}")

def test_deepspeed_import():
    """测试DeepSpeed导入"""
    print("\n🔍 测试DeepSpeed导入...")
    
    try:
        import deepspeed
        print(f"✅ DeepSpeed版本: {deepspeed.__version__}")
        
        # 测试DeepSpeed初始化
        print("测试DeepSpeed初始化...")
        engine = deepspeed.init_distributed()
        print("✅ DeepSpeed初始化成功")
        
    except Exception as e:
        print(f"❌ DeepSpeed测试失败: {e}")

def test_network_connectivity():
    """测试网络连通性"""
    print("\n🔍 测试网络连通性...")
    
    try:
        import socket
        
        # 测试本地端口
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('localhost', 29500))
        sock.close()
        
        if result == 0:
            print("❌ 端口29500已被占用")
        else:
            print("✅ 端口29500可用")
            
    except Exception as e:
        print(f"❌ 网络测试失败: {e}")

def main():
    """主函数"""
    print("=" * 80)
    print("🔧 DeepSpeed初始化调试工具")
    print("=" * 80)
    
    # 设置环境变量用于调试
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_IB_DISABLE'] = '1'  # 禁用InfiniBand
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'  # 使用本地回环接口
    
    # 检查环境
    check_environment()
    
    # 测试基础功能
    test_basic_torch()
    
    # 测试网络
    test_network_connectivity()
    
    # 测试NCCL
    test_nccl()
    
    # 测试DeepSpeed
    test_deepspeed_import()
    
    print("\n" + "=" * 80)
    print("✅ 调试完成")
    print("=" * 80)

if __name__ == "__main__":
    main() 