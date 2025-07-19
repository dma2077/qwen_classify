#!/usr/bin/env python3
"""
测试training指标记录到WandB的功能
"""

import os
import sys
import time
import yaml
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from training.utils.monitor import TrainingMonitor

def test_training_metrics():
    """测试training指标记录功能"""
    print("🧪 开始测试training指标记录...")
    
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
            'project': 'test_training_metrics',
            'run_name': 'test_run'
        },
        'monitor': {
            'freq': {
                'training_log_freq': 1,  # 每步都记录
                'perf_log_freq': 2,      # 每2步记录性能
                'eval_log_freq': 1       # 每步都记录eval
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
    for step in range(1, 11):
        training_data = {
            "training/loss": 0.5 - step * 0.01,
            "training/lr": 1e-4,
            "training/epoch": 0.1 * step,
            "training/grad_norm": 1.0 + step * 0.1
        }
        
        # 每2步添加性能指标
        if step % 2 == 0:
            training_data.update({
                "perf/step_time": 0.1 + step * 0.01,
                "perf/steps_per_second": 10.0 - step * 0.1,
                "perf/mfu": 0.3 + step * 0.02
            })
        
        try:
            monitor.log_metrics(training_data, step=step, commit=True)
            print(f"  ✅ Step {step}: training指标记录成功")
            
            # 检查WandB的当前step
            import wandb
            if wandb.run is not None:
                current_wandb_step = getattr(wandb.run, 'step', 0)
                print(f"     📊 WandB当前step: {current_wandb_step}")
                
                # 检查step是否一致
                if current_wandb_step == step:
                    print(f"     ✅ Step一致")
                else:
                    print(f"     ⚠️  Step不一致: 期望{step}, 实际{current_wandb_step}")
            
            time.sleep(0.1)  # 短暂延迟
            
        except Exception as e:
            print(f"  ❌ Step {step}: 记录失败 - {e}")
    
    print("\n📊 测试eval指标记录...")
    eval_data = {
        "eval/overall_loss": 0.3,
        "eval/overall_accuracy": 0.85,
        "eval/overall_samples": 1000,
        "eval/overall_correct": 850
    }
    
    try:
        monitor.log_metrics(eval_data, step=10, commit=True)
        print("✅ Eval指标记录成功")
    except Exception as e:
        print(f"❌ Eval指标记录失败: {e}")
    
    print("\n🎉 测试完成！")
    print("📊 请检查WandB界面，应该能看到:")
    print("   • Training指标: loss, lr, epoch, grad_norm")
    print("   • Perf指标: step_time, steps_per_second, mfu")
    print("   • Eval指标: overall_loss, overall_accuracy")
    print("   • 所有指标都应该有正确的step值")

if __name__ == "__main__":
    test_training_metrics() 