#!/usr/bin/env python3
"""
测试每次eval都能正确记录
"""

import os
import sys
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_eval_steps():
    """测试每次eval都能正确记录"""
    
    # 模拟配置
    config = {
        'output_dir': '/tmp/test_eval_steps',
        'wandb': {
            'enabled': True,
            'project': 'qwen_classification_test',
            'run_name': 'test_eval_steps',
            'tags': ['test', 'eval', 'steps'],
            'notes': 'Testing eval steps recording'
        }
    }
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    try:
        from training.utils.monitor import TrainingMonitor
        
        # 创建monitor
        monitor = TrainingMonitor(config['output_dir'], config)
        
        print("✅ Monitor创建成功")
        
        # 等待WandB初始化
        time.sleep(2)
        
        # 模拟每30步eval一次
        eval_steps = [30, 60, 90, 120]
        
        for step in eval_steps:
            print(f"\n{'='*50}")
            print(f"📊 模拟第{step}步的eval")
            print(f"{'='*50}")
            
            # 模拟training指标
            training_metrics = {
                "training/loss": 0.5 - step * 0.001,  # 模拟loss下降
                "training/lr": 1e-5,
                "training/epoch": step // 100,
                "training/grad_norm": 1.0 + step * 0.01
            }
            
            # 模拟eval指标
            eval_metrics = {
                "eval/overall_loss": 0.3 - step * 0.0005,  # 模拟eval loss下降
                "eval/overall_accuracy": 0.7 + step * 0.002,  # 模拟accuracy上升
                "eval/overall_samples": 1000,
                "eval/overall_correct": int(700 + step * 2)
            }
            
            # 合并指标
            combined_metrics = training_metrics.copy()
            combined_metrics.update(eval_metrics)
            
            print(f"📊 准备记录指标 (step={step}):")
            print(f"   🏃 training指标: {list(training_metrics.keys())}")
            print(f"   📊 eval指标: {list(eval_metrics.keys())}")
            print(f"   🔢 总指标数量: {len(combined_metrics)}")
            
            # 记录到WandB
            monitor.log_metrics(combined_metrics, step=step, commit=True)
            
            print(f"✅ 第{step}步指标记录完成")
            
            # 等待一下让WandB同步
            time.sleep(2)
        
        # 结束WandB
        monitor.finish_training()
        
        print(f"\n{'='*50}")
        print("✅ 测试完成")
        print("🔗 请检查WandB界面，应该能看到以下eval步骤:")
        for step in eval_steps:
            print(f"   📊 Step {step}: eval/overall_loss, eval/overall_accuracy")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_eval_steps() 