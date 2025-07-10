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
            print(f"世界大小: {self.world_size}")
            print(f"当前进程: {self.rank}")
            print(f"本地进程: {self.local_rank}")
            print(f"设备: {self.device}")
    
    def barrier(self):
        """同步所有进程"""
        if self.world_size > 1:
            dist.barrier()
    
    def print_main(self, message):
        """只有主进程打印消息"""
        if self.is_main_process:
            print(message) 