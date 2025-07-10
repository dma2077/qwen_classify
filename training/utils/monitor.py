import time
import json
import os
import torch
from typing import Dict, List, Optional

def make_json_serializable(obj):
    """确保对象可以JSON序列化"""
    if isinstance(obj, torch.Tensor):
        return float(obj.item()) if obj.numel() == 1 else obj.tolist()
    elif torch.is_tensor(obj):
        # 处理其他torch tensor类型
        return float(obj.item()) if obj.numel() == 1 else obj.tolist()
    elif hasattr(obj, 'dtype') and 'torch' in str(type(obj)):
        # 处理torch的标量类型
        if 'float' in str(obj.dtype):
            return float(obj)
        elif 'int' in str(obj.dtype):
            return int(obj)
        else:
            return float(obj)
    elif isinstance(obj, (float, int)):
        return obj
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, output_dir: str, log_file: str = "training_log.json"):
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, log_file)
        self.step_logs = []
        self.epoch_logs = []
        self.start_time = None
        self.step_start_time = None
        
        # 创建日志目录
        os.makedirs(output_dir, exist_ok=True)
    
    def start_training(self):
        """开始训练"""
        self.start_time = time.time()
        self.step_start_time = time.time()
    
    def log_step(self, step: int, epoch: int, loss: float, grad_norm: float, learning_rate: float):
        """记录训练步骤"""
        current_time = time.time()
        step_time = current_time - self.step_start_time
        
        log_entry = {
            'step': int(step),
            'epoch': int(epoch),
            'loss': float(loss),
            'grad_norm': float(grad_norm),
            'learning_rate': float(learning_rate),
            'step_time': float(step_time),
            'timestamp': float(current_time)
        }
        
        self.step_logs.append(log_entry)
        self.step_start_time = current_time
        
        # 定期保存日志
        if step % 100 == 0:
            self.save_logs()
    
    def log_epoch(self, epoch: int, avg_loss: float, elapsed_time: float):
        """记录epoch统计"""
        log_entry = {
            'epoch': int(epoch),
            'avg_loss': float(avg_loss),
            'elapsed_time': float(elapsed_time),
            'timestamp': float(time.time())
        }
        
        self.epoch_logs.append(log_entry)
        self.save_logs()
    
    def save_logs(self):
        """保存日志到文件"""
        try:
            logs = {
                'step_logs': self.step_logs,
                'epoch_logs': self.epoch_logs,
                'total_training_time': time.time() - self.start_time if self.start_time else 0
            }
            
            # 确保所有数据都可以JSON序列化
            serializable_logs = make_json_serializable(logs)
            
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_logs, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存日志失败: {e}")
    
    def get_latest_metrics(self) -> Optional[Dict]:
        """获取最新的训练指标"""
        if self.step_logs:
            return self.step_logs[-1]
        return None
    
    def get_avg_metrics(self, last_n_steps: int = 100) -> Dict:
        """获取最近N步的平均指标"""
        if not self.step_logs:
            return {}
        
        recent_logs = self.step_logs[-last_n_steps:]
        
        if not recent_logs:
            return {}
        
        avg_loss = sum(log['loss'] for log in recent_logs) / len(recent_logs)
        avg_grad_norm = sum(log['grad_norm'] for log in recent_logs) / len(recent_logs)
        avg_step_time = sum(log['step_time'] for log in recent_logs) / len(recent_logs)
        
        return {
            'avg_loss': avg_loss,
            'avg_grad_norm': avg_grad_norm,
            'avg_step_time': avg_step_time,
            'num_steps': len(recent_logs)
        } 