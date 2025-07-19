#!/usr/bin/env python3
"""
快速WandB测试脚本
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from training.utils.monitor import TrainingMonitor

def quick_test():
    """快速测试WandB功能"""
    print("🚀 快速WandB测试...")
    
    # 创建测试配置
    test_config = {
        'model': {
            'pretrained_name': 'test_model',
            'num_labels': 10
        },
        'training': {
            'num_epochs': 1,
            'learning_rate': 1e-4
        },
        'wandb': {
            'enabled': True,
            'project': 'quick_test',
            'run_name': 'test_run'
        },
        'monitor': {
            'freq': {
                'training_log_freq': 1,
                'perf_log_freq': 1,
                'eval_log_freq': 1
            }
        }
    }
    
    # 创建输出目录
    output_dir = "./test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建监控器
    monitor = TrainingMonitor(output_dir, test_config)
    
    if not monitor.use_wandb:
        print("❌ WandB未启用")
        return
    
    print("✅ WandB已启用")
    
    # 测试记录
    for step in range(1, 4):
        data = {
            "training/loss": 1.0 - step * 0.2,
            "training/lr": 1e-4,
            "training/epoch": step * 0.1,
            "training/grad_norm": 1.0 + step * 0.1
        }
        
        monitor.log_metrics(data, step=step, commit=True)
        print(f"✅ Step {step} 记录完成")
    
    print("🎉 测试完成！")

if __name__ == "__main__":
    quick_test() 