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
        # 简化输出，只在需要时显示
        pass
    
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
    改进的barrier操作，添加更好的超时和错误处理
    """
    if not (dist.is_available() and dist.is_initialized()):
        return True
    
    try:
        # 🔥 改进：使用更智能的barrier策略
        import threading
        import time
        
        # 创建一个标志来跟踪barrier是否完成
        barrier_completed = threading.Event()
        barrier_error = None
        
        def barrier_worker():
            nonlocal barrier_error
            try:
                dist.barrier()
                barrier_completed.set()
            except Exception as e:
                barrier_error = e
                barrier_completed.set()
        
        # 在单独线程中执行barrier
        worker_thread = threading.Thread(target=barrier_worker)
        worker_thread.daemon = True
        worker_thread.start()
        
        # 等待barrier完成或超时
        if barrier_completed.wait(timeout=timeout):
            if barrier_error is None:
                return True
            else:
                print(f"❌ barrier操作失败: {barrier_error}")
                return False
        else:
            print(f"❌ barrier操作超时 ({timeout}秒)")
            return False
            
    except Exception as e:
        print(f"❌ barrier操作异常: {e}")
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
    """设置NCCL环境变量以提高稳定性"""
    
    # 🔥 改进：设置更全面的NCCL配置
    nccl_configs = {
        # 基础超时设置
        'NCCL_TIMEOUT': '300',  # 5分钟超时，从10分钟减少
        'NCCL_ASYNC_ERROR_HANDLING': '1',  # 启用异步错误处理
        
        # 🔥 新增：连接稳定性设置
        'NCCL_SOCKET_TIMEOUT': '30000',  # 30秒socket超时
        'NCCL_CONNECT_TIMEOUT': '30000',  # 30秒连接超时
        'NCCL_HEARTBEAT_TIMEOUT': '30000',  # 30秒心跳超时
        
        # 🔥 新增：网络重试设置
        'NCCL_RETRY_COUNT': '3',  # 重试3次
        'NCCL_RETRY_TIMEOUT': '10000',  # 重试超时10秒
        
        # 🔥 新增：评估时的特殊设置
        'NCCL_BUFFSIZE': '8388608',  # 8MB缓冲区，减少内存使用
        'NCCL_NTHREADS': '16',  # 减少线程数
        
        # 🔥 新增：调试和稳定性
        'NCCL_DEBUG': 'WARN',  # 只显示警告和错误
        'NCCL_IB_DISABLE': '1',  # 禁用InfiniBand，使用TCP
        'NCCL_P2P_DISABLE': '1',  # 禁用P2P，提高稳定性
    }
    
    # 只设置尚未设置的环境变量
    for key, value in nccl_configs.items():
        if key not in os.environ:
            os.environ[key] = value
    
    # 静默设置，不输出信息
    pass 