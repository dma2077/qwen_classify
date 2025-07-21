#!/usr/bin/env python3
"""
诊断分布式rank问题
"""

import os
import sys
import argparse
import torch
import deepspeed

def debug_distributed_info():
    """诊断分布式环境信息"""
    
    print("=" * 80)
    print("🔍 分布式环境诊断")
    print("=" * 80)
    
    # 1. 检查命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    print(f"📋 命令行参数:")
    print(f"  • --local_rank: {args.local_rank}")
    print(f"  • sys.argv: {sys.argv}")
    
    # 2. 检查环境变量
    print(f"\n🌍 环境变量:")
    important_vars = [
        'LOCAL_RANK', 'RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
        'CUDA_VISIBLE_DEVICES', 'NCCL_DEBUG', 'NCCL_SOCKET_IFNAME'
    ]
    
    for var in important_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"  • {var}: {value}")
    
    # 3. 初始化分布式
    print(f"\n🔧 初始化分布式...")
    try:
        deepspeed.init_distributed()
        print("✅ 分布式初始化成功")
    except Exception as e:
        print(f"❌ 分布式初始化失败: {e}")
        return
    
    # 4. 检查分布式状态
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
        
        print(f"\n📊 分布式状态:")
        print(f"  • Global Rank: {rank}")
        print(f"  • World Size: {world_size}")
        print(f"  • Local Rank: {local_rank}")
        print(f"  • Device: cuda:{local_rank}")
        
        # 5. 检查GPU设备
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            device_count = torch.cuda.device_count()
            print(f"  • Current CUDA Device: {current_device}")
            print(f"  • Available CUDA Devices: {device_count}")
            
            # 设置正确的设备
            torch.cuda.set_device(local_rank)
            print(f"  • Set CUDA Device to: {local_rank}")
        
        # 6. 测试简单的分布式操作
        print(f"\n🧪 测试分布式操作:")
        test_tensor = torch.tensor([rank], dtype=torch.float32, device=f'cuda:{local_rank}')
        print(f"  • Rank {rank} created tensor: {test_tensor}")
        
        # All-reduce测试
        try:
            dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
            print(f"  • Rank {rank} after all_reduce: {test_tensor}")
        except Exception as e:
            print(f"  • Rank {rank} all_reduce failed: {e}")
        
        # Barrier测试
        try:
            dist.barrier()
            print(f"  • Rank {rank} passed barrier")
        except Exception as e:
            print(f"  • Rank {rank} barrier failed: {e}")
            
    else:
        print("❌ 分布式未初始化")
    
    print("\n" + "=" * 80)
    print(f"🏁 诊断完成 - Rank {rank if 'rank' in locals() else 'UNKNOWN'}")
    print("=" * 80)

if __name__ == "__main__":
    debug_distributed_info() 