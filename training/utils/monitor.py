import time
import json
import os
from typing import Dict, List, Optional

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
            'step': step,
            'epoch': epoch,
            'loss': loss,
            'grad_norm': grad_norm,
            'learning_rate': learning_rate,
            'step_time': step_time,
            'timestamp': current_time
        }
        
        self.step_logs.append(log_entry)
        self.step_start_time = current_time
        
        # 定期保存日志
        if step % 100 == 0:
            self.save_logs()
    
    def log_epoch(self, epoch: int, avg_loss: float, elapsed_time: float):
        """记录epoch统计"""
        log_entry = {
            'epoch': epoch,
            'avg_loss': avg_loss,
            'elapsed_time': elapsed_time,
            'timestamp': time.time()
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
            
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
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