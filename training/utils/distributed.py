import os
import torch
import torch.distributed as dist
import time
import signal
from contextlib import contextmanager

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

class TimeoutError(Exception):
    """自定义超时异常"""
    pass

@contextmanager
def timeout_handler(timeout_seconds=300):
    """
    上下文管理器，用于处理操作超时
    
    Args:
        timeout_seconds: 超时秒数，默认5分钟
    """
    def timeout_signal_handler(signum, frame):
        raise TimeoutError(f"操作超时 ({timeout_seconds}秒)")
    
    # 设置信号处理器
    old_handler = signal.signal(signal.SIGALRM, timeout_signal_handler)
    signal.alarm(timeout_seconds)
    
    try:
        yield
    finally:
        # 恢复原来的信号处理器
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def safe_all_reduce(tensor, op=dist.ReduceOp.SUM, timeout=180):
    """
    安全的all_reduce操作，带有超时保护
    
    Args:
        tensor: 要聚合的tensor
        op: 聚合操作类型
        timeout: 超时秒数，默认3分钟
        
    Returns:
        bool: 是否成功执行
    """
    if not (dist.is_available() and dist.is_initialized()):
        return True  # 非分布式环境，直接返回成功
    
    try:
        with timeout_handler(timeout):
            dist.all_reduce(tensor, op=op)
        return True
    except TimeoutError as e:
        print(f"❌ all_reduce操作超时: {e}")
        return False
    except Exception as e:
        print(f"❌ all_reduce操作失败: {e}")
        return False

def safe_barrier(timeout=120):
    """
    安全的barrier操作，带有超时保护
    
    Args:
        timeout: 超时秒数，默认2分钟
        
    Returns:
        bool: 是否成功执行
    """
    if not (dist.is_available() and dist.is_initialized()):
        return True  # 非分布式环境，直接返回成功
    
    try:
        with timeout_handler(timeout):
            dist.barrier()
        return True
    except TimeoutError as e:
        print(f"❌ barrier操作超时: {e}")
        return False
    except Exception as e:
        print(f"❌ barrier操作失败: {e}")
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
    """设置NCCL超时环境变量"""
    # 设置较短的超时时间，避免长时间挂起
    os.environ['NCCL_TIMEOUT'] = '300'  # 5分钟超时
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'  # 启用异步错误处理
    
    # 在某些环境下禁用P2P通信，提高稳定性
    if 'NCCL_P2P_DISABLE' not in os.environ:
        os.environ['NCCL_P2P_DISABLE'] = '1'
    
    # 强制使用树形算法，减少通信复杂度
    if 'NCCL_TREE_THRESHOLD' not in os.environ:
        os.environ['NCCL_TREE_THRESHOLD'] = '0'
    
    print("✅ 已设置NCCL超时保护环境变量")
    get_nccl_debug_info() 