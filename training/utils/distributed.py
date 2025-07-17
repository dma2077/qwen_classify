import os
import torch
import torch.distributed as dist

class DistributedContext:
    """管理分布式训练的上下文"""
    
    def __init__(self):
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.is_main_process = self.rank == 0
        
        # 设置CUDA设备
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.device = torch.device('cpu')
    
    def print_info(self):
        """打印分布式训练信息"""
        if self.is_main_process:
            print(f"WORLD_SIZE: {self.world_size}")
            print(f"rank: {self.rank}")
            print(f"local_rank: {self.local_rank}")
            print(f"device: {self.device}")
    
    def barrier(self):
        """同步所有进程"""
        if self.world_size > 1:
            dist.barrier()
    
    def print_main(self, message):
        """只有主进程打印消息"""
        if self.is_main_process:
            print(message)

def safe_all_reduce(tensor, op=dist.ReduceOp.SUM):
    """
    简化的all_reduce操作 - 移除复杂的超时处理，因为已修复根本原因
    """
    if not (dist.is_available() and dist.is_initialized()):
        return True  # 非分布式环境，直接返回成功
    
    try:
        dist.all_reduce(tensor, op=op)
        return True
    except Exception as e:
        print(f"❌ all_reduce操作失败: {e}")
        return False

def safe_barrier(timeout=300):
    """
    简化的barrier操作
    """
    if not (dist.is_available() and dist.is_initialized()):
        return True
    
    try:
        dist.barrier()
        return True
    except Exception as e:
        print(f"❌ barrier操作失败: {e}")
        return False

def batch_all_reduce(tensors, op=dist.ReduceOp.SUM):
    """
    简化的批量all_reduce操作 - 移除复杂的分块逻辑
    """
    if not (dist.is_available() and dist.is_initialized()):
        return True  # 非分布式环境，直接返回成功
    
    if not tensors:
        return True
    
    try:
        for tensor in tensors:
            dist.all_reduce(tensor, op=op)
        return True
    except Exception as e:
        print(f"❌ 批量all_reduce操作失败: {e}")
        return False

def setup_nccl_timeout_env():
    """设置基础的NCCL超时环境变量"""
    
    # 设置合理的超时时间（10分钟），现在不需要30分钟了
    if 'NCCL_TIMEOUT' not in os.environ:
        os.environ['NCCL_TIMEOUT'] = '600'  # 10分钟超时
    
    # 启用异步错误处理
    if 'NCCL_ASYNC_ERROR_HANDLING' not in os.environ:
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    
    print("✅ 已设置基础NCCL超时保护（10分钟超时）") 