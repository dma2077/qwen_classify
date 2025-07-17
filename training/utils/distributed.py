import os
import torch
import torch.distributed as dist
import time

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

def safe_all_reduce(tensor, op=dist.ReduceOp.SUM, timeout=None):
    """
    安全的all_reduce操作，简化版本，专注性能
    
    Args:
        tensor: 要聚合的tensor
        op: 聚合操作类型
        timeout: 保留参数但不使用，避免性能开销
        
    Returns:
        bool: 是否成功执行
    """
    if not (dist.is_available() and dist.is_initialized()):
        return True  # 非分布式环境，直接返回成功
    
    try:
        # 直接调用all_reduce，让NCCL自身处理超时
        dist.all_reduce(tensor, op=op)
        return True
    except Exception as e:
        print(f"❌ all_reduce操作失败: {e}")
        return False

def safe_barrier(timeout=None):
    """
    安全的barrier操作，简化版本
    
    Args:
        timeout: 保留参数但不使用
        
    Returns:
        bool: 是否成功执行
    """
    if not (dist.is_available() and dist.is_initialized()):
        return True  # 非分布式环境，直接返回成功
    
    try:
        dist.barrier()
        return True
    except Exception as e:
        print(f"❌ barrier操作失败: {e}")
        return False

def batch_all_reduce(tensors, op=dist.ReduceOp.SUM):
    """
    批量all_reduce操作，减少通信次数
    
    Args:
        tensors: tensor列表
        op: 聚合操作类型
        
    Returns:
        bool: 是否全部成功执行
    """
    if not (dist.is_available() and dist.is_initialized()):
        return True  # 非分布式环境，直接返回成功
    
    if not tensors:
        return True
    
    try:
        # 批量处理所有tensor
        for tensor in tensors:
            dist.all_reduce(tensor, op=op)
        return True
    except Exception as e:
        print(f"❌ 批量all_reduce操作失败: {e}")
        return False

def get_nccl_debug_info():
    """获取NCCL调试信息"""
    info = {
        'NCCL_DEBUG': os.environ.get('NCCL_DEBUG', 'Not set'),
        'NCCL_SOCKET_IFNAME': os.environ.get('NCCL_SOCKET_IFNAME', 'Not set'),
        'NCCL_IB_DISABLE': os.environ.get('NCCL_IB_DISABLE', 'Not set'),
        'NCCL_P2P_DISABLE': os.environ.get('NCCL_P2P_DISABLE', 'Not set'),
        'NCCL_TREE_THRESHOLD': os.environ.get('NCCL_TREE_THRESHOLD', 'Not set'),
    }
    
    print("🔍 NCCL环境变量:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    return info

def setup_nccl_timeout_env():
    """设置轻量级的NCCL超时环境变量，专注性能"""
    # 只设置必要的超时保护，避免过度配置影响性能
    if 'NCCL_TIMEOUT' not in os.environ:
        os.environ['NCCL_TIMEOUT'] = '600'  # 10分钟超时，比较宽松
    
    # 启用异步错误处理，但不强制禁用性能优化
    if 'NCCL_ASYNC_ERROR_HANDLING' not in os.environ:
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    
    print("✅ 已设置轻量级NCCL超时保护")
    
    # 只在调试模式下显示详细信息
    if os.environ.get('NCCL_DEBUG', '').upper() in ['INFO', 'WARN']:
        get_nccl_debug_info() 