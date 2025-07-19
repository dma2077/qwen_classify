#!/usr/bin/env python3
"""
测试eval指标在WandB界面上的可见性
"""

import os
import sys
import time
import random
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from training.utils.monitor import TrainingMonitor

def test_eval_wandb_visibility():
    """测试eval指标在WandB界面上的可见性"""
    print("🧪 开始测试eval指标在WandB界面上的可见性...")
    
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
            'project': 'test_eval_visibility',
            'run_name': f'eval_visibility_test_{int(time.time())}'
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
    output_dir = "./test_eval_visibility_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建监控器
    monitor = TrainingMonitor(output_dir, test_config)
    
    if not monitor.use_wandb:
        print("❌ WandB未启用，跳过测试")
        return
    
    print("✅ WandB已启用，开始记录测试指标...")
    
    # 模拟训练过程，记录多个数据点
    for step in range(1, 21):  # 记录20个数据点
        # 记录training指标
        training_data = {
            "training/loss": 0.5 - step * 0.01,
            "training/lr": 1e-4,
            "training/epoch": 0.1 * step,
            "training/grad_norm": 1.0 + step * 0.1
        }
        
        monitor.log_metrics(training_data, step=step, commit=True)
        print(f"  ✅ Step {step}: training指标记录成功")
        
        # 每5步记录一次eval指标（模拟评估频率）
        if step % 5 == 0:
            # 模拟eval指标（使用随机值模拟真实训练过程）
            eval_loss = 0.3 - step * 0.005 + random.uniform(-0.02, 0.02)
            eval_accuracy = 0.8 + step * 0.01 + random.uniform(-0.05, 0.05)
            
            eval_data = {
                "eval/overall_loss": max(0.1, eval_loss),  # 确保损失为正
                "eval/overall_accuracy": min(1.0, max(0.0, eval_accuracy)),  # 确保准确率在0-1之间
                "eval/overall_samples": 1000,
                "eval/overall_correct": int(1000 * min(1.0, max(0.0, eval_accuracy))),
                "eval/test_dataset_loss": max(0.1, eval_loss),
                "eval/test_dataset_accuracy": min(1.0, max(0.0, eval_accuracy)),
                "eval/test_dataset_samples": 1000
            }
            
            monitor.log_metrics(eval_data, step=step, commit=True)
            print(f"  📊 Step {step}: eval指标记录成功")
            print(f"     📈 Eval Loss: {eval_data['eval/overall_loss']:.4f}")
            print(f"     📈 Eval Accuracy: {eval_data['eval/overall_accuracy']:.4f}")
            
            # 显示eval指标详情
            eval_metrics_list = [k for k in eval_data.keys() if k.startswith('eval/')]
            print(f"     📊 Eval指标: {eval_metrics_list}")
        
        # 每10步记录一次性能指标
        if step % 10 == 0:
            perf_data = {
                "perf/step_time": 0.1 + random.uniform(-0.02, 0.02),
                "perf/mfu": 0.3 + random.uniform(-0.05, 0.05),
                "perf/tokens_per_second": 1000 + random.uniform(-100, 100)
            }
            
            monitor.log_metrics(perf_data, step=step, commit=True)
            print(f"  ⚡ Step {step}: perf指标记录成功")
        
        # 短暂延迟，模拟真实训练
        time.sleep(0.1)
    
    print("\n🎉 测试完成！")
    print("📊 请检查WandB界面，应该能看到:")
    print("   • Training指标图表: loss, lr, epoch, grad_norm")
    print("   • Eval指标图表: overall_loss, overall_accuracy, overall_samples, overall_correct")
    print("   • 数据集特定指标: test_dataset_loss, test_dataset_accuracy, test_dataset_samples")
    print("   • 性能指标图表: step_time, mfu, tokens_per_second")
    print("   • Eval指标在step 5, 10, 15, 20时记录")
    print("   • 总共记录了4个eval数据点")
    
    # 显示WandB URL
    try:
        import wandb
        if wandb.run is not None:
            print(f"\n🔗 WandB URL: {wandb.run.url}")
            print(f"📊 项目: {wandb.run.project}")
            print(f"🏃 运行名称: {wandb.run.name}")
    except Exception as e:
        print(f"⚠️ 获取WandB URL失败: {e}")

if __name__ == "__main__":
    test_eval_wandb_visibility() 