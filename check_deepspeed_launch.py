#!/usr/bin/env python3
"""
检查DeepSpeed启动是否正确分配rank
"""

import os
import sys
import time

def check_deepspeed_launch():
    """检查DeepSpeed启动状态"""
    
    print("=" * 80)
    print("🔍 DeepSpeed启动检查")
    print("=" * 80)
    
    # 检查关键环境变量
    print("\n🌍 关键环境变量:")
    env_vars = ['LOCAL_RANK', 'RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
    for var in env_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"  • {var}: {value}")
    
    # 检查CUDA设备
    print(f"\n🎮 CUDA环境:")
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')
    print(f"  • CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            print(f"  • Available devices: {device_count}")
            print(f"  • Current device: {current_device}")
        else:
            print("  • CUDA not available!")
    except ImportError:
        print("  • PyTorch not available!")
    
    # 检查进程信息
    print(f"\n🔧 进程信息:")
    print(f"  • PID: {os.getpid()}")
    print(f"  • PPID: {os.getppid()}")
    
    # 检查命令行参数
    print(f"\n📋 命令行参数:")
    print(f"  • sys.argv: {sys.argv}")
    
    # 模拟等待，让我们看到所有进程的输出
    local_rank = os.environ.get('LOCAL_RANK', 'UNKNOWN')
    rank = os.environ.get('RANK', 'UNKNOWN')
    
    print(f"\n🎯 当前进程标识:")
    print(f"  • LOCAL_RANK: {local_rank}")
    print(f"  • RANK: {rank}")
    print(f"  • PID: {os.getpid()}")
    
    # 等待一段时间，让所有进程都输出
    print(f"\n⏰ 等待5秒让所有进程输出...")
    time.sleep(5)
    
    print(f"\n🏁 进程 LOCAL_RANK={local_rank}, RANK={rank} 检查完成")

if __name__ == "__main__":
    check_deepspeed_launch() 