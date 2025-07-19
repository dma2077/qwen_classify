#!/usr/bin/env python3
"""
测试eval指标记录到WandB的功能
"""

import os
import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from training.utils.monitor import TrainingMonitor

def test_eval_metrics():
    """测试eval指标记录功能"""
    print("🧪 开始测试eval指标记录...")
    
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
            'project': 'test_eval_metrics',
            'run_name': 'eval_test_run'
        },
        'monitor': {
            'freq': {
                'training_log_freq': 1,
                'eval_log_freq': 1,
                'perf_log_freq': 1
            }
        },
        'datasets': {
            'dataset_configs': {
                'test_dataset': {
                    'num_classes': 10,
                    'description': 'Test dataset'
                }
            }
        }
    }
    
    # 创建输出目录
    output_dir = "./test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建监控器
    monitor = TrainingMonitor(output_dir, test_config)
    
    if not monitor.use_wandb:
        print("❌ WandB未启用，跳过测试")
        return
    
    print("✅ WandB已启用，开始记录测试指标...")
    
    # 测试training指标记录
    for step in range(1, 6):
        training_data = {
            "training/loss": 0.5 - step * 0.01,
            "training/lr": 1e-4,
            "training/epoch": 0.1 * step,
            "training/grad_norm": 1.0 + step * 0.1
        }
        
        monitor.log_metrics(training_data, step=step, commit=True)
        print(f"  ✅ Step {step}: training指标记录成功")
        
        # 每3步记录一次eval指标
        if step % 3 == 0:
            eval_data = {
                "eval/overall_loss": 0.3 - step * 0.02,
                "eval/overall_accuracy": 0.8 + step * 0.02,
                "eval/overall_samples": 1000,
                "eval/overall_correct": 800 + step * 20,
                "eval/test_dataset_loss": 0.3 - step * 0.02,
                "eval/test_dataset_accuracy": 0.8 + step * 0.02,
                "eval/test_dataset_samples": 1000
            }
            
            monitor.log_metrics(eval_data, step=step, commit=True)
            print(f"  📊 Step {step}: eval指标记录成功")
            
            # 显示eval指标详情
            eval_metrics_list = [k for k in eval_data.keys() if k.startswith('eval/')]
            print(f"     📈 Eval指标: {eval_metrics_list}")
    
    print("\n🎉 测试完成！")
    print("📊 请检查WandB界面，应该能看到:")
    print("   • Training指标: loss, lr, epoch, grad_norm")
    print("   • Eval指标: overall_loss, overall_accuracy, overall_samples, overall_correct")
    print("   • 数据集特定指标: test_dataset_loss, test_dataset_accuracy, test_dataset_samples")
    print("   • Eval指标在step 3和6时记录")

if __name__ == "__main__":
    test_eval_metrics() 