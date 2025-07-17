#!/usr/bin/env python3
"""
测试training和eval指标同时记录到WandB
"""

import os
import sys
import time
import json
import yaml

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_combined_metrics_logging():
    """测试合并指标记录"""
    
    print("🧪 开始测试合并指标记录...")
    
    # 模拟配置
    config = {
        'output_dir': './test_output',
        'wandb': {
            'project': 'test_metrics',
            'name': 'test_combined_metrics',
            'enabled': True
        },
        'monitor': {
            'freq': {
                'log_freq': 1,
                'eval_log_freq': 1,
                'perf_log_freq': 1,
                'flops_profile_freq': 10
            }
        }
    }
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 初始化monitor
    from training.utils.monitor import TrainingMonitor
    monitor = TrainingMonitor(config['output_dir'], config)
    
    # 模拟训练和eval数据
    for step in range(1, 11):
        print(f"\n📊 步骤 {step}:")
        
        # 模拟training数据
        training_data = {
            "training/loss": 0.1 + step * 0.01,
            "training/lr": 1e-4,
            "training/epoch": 0.1,
            "training/grad_norm": 1.0 + step * 0.1,
        }
        
        # 模拟eval数据（每5步评估一次）
        eval_data = {}
        if step % 5 == 0:
            eval_data = {
                "eval/overall_loss": 0.2 + step * 0.01,
                "eval/overall_accuracy": 0.8 - step * 0.01,
                "eval/overall_samples": 1000,
                "eval/overall_correct": 800 - step * 10,
            }
            print(f"   📈 包含eval指标: {list(eval_data.keys())}")
        else:
            print(f"   📈 仅包含training指标")
        
        # 合并数据
        combined_data = {**training_data, **eval_data}
        combined_data["step"] = step
        
        # 记录到WandB
        monitor.log_metrics(combined_data, step, commit=True)
        
        print(f"   ✅ 已记录 {len(combined_data)} 个指标")
        print(f"   📊 指标keys: {list(combined_data.keys())}")
        
        time.sleep(1)  # 避免WandB API限制
    
    print("\n🎉 测试完成！")
    print("请检查WandB界面，应该能看到:")
    print("  - training/loss, training/lr, training/epoch, training/grad_norm")
    print("  - eval/overall_loss, eval/overall_accuracy (每5步)")
    print("  - 所有指标都使用相同的step轴")

if __name__ == "__main__":
    test_combined_metrics_logging() 