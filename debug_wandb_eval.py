#!/usr/bin/env python3
"""
调试WandB eval指标记录问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.utils.monitor import TrainingMonitor

def test_wandb_eval_logging():
    """测试WandB eval指标记录"""
    
    print("=" * 60)
    print("测试WandB eval指标记录")
    print("=" * 60)
    
    # 模拟训练配置
    config = {
        'wandb': {
            'enabled': True,  # 启用WandB
            'project': 'test_eval_debug',
            'run_name': 'eval_debug_test'
        },
        'datasets': {
            'dataset_configs': {
                'food101': {'num_classes': 101},
                'test_dataset': {'num_classes': 50}
            }
        }
    }
    
    print("\n1. 创建TrainingMonitor...")
    monitor = TrainingMonitor("./test_output", config=config)
    
    if not monitor.use_wandb:
        print("❌ WandB未启用，跳过测试")
        return
    
    print("\n2. 测试eval指标记录...")
    
    # 测试基础eval指标
    eval_data_1 = {
        "eval/overall_loss": 0.5,
        "eval/overall_accuracy": 0.85,
        "eval/overall_samples": 1000,
        "eval/overall_correct": 850,
    }
    
    print("\n   测试基础eval指标...")
    monitor.log_metrics(eval_data_1, step=10, commit=True)
    
    # 测试多数据集eval指标
    eval_data_2 = {
        "eval/overall_loss": 0.4,
        "eval/overall_accuracy": 0.88,
        "eval/food101_loss": 0.42,
        "eval/food101_accuracy": 0.86,
        "eval/test_dataset_loss": 0.38,
        "eval/test_dataset_accuracy": 0.90,
    }
    
    print("\n   测试多数据集eval指标...")
    monitor.log_metrics(eval_data_2, step=20, commit=True)
    
    # 测试最终eval指标
    eval_data_3 = {
        "eval/final_overall_loss": 0.35,
        "eval/final_overall_accuracy": 0.92,
        "eval/final_evaluation": 1.0,
    }
    
    print("\n   测试最终eval指标...")
    monitor.log_metrics(eval_data_3, step=30, commit=True)
    
    print("\n3. 完成测试")
    
    try:
        import wandb
        if wandb.run is not None:
            print(f"\n🔗 查看结果: {wandb.run.url}")
            print("📊 请检查WandB界面中是否显示eval组指标")
        
        # 完成wandb记录
        monitor.finish_training()
        
    except Exception as e:
        print(f"❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("调试测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_wandb_eval_logging() 