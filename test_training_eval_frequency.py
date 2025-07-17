#!/usr/bin/env python3
"""
测试training和eval频率不同的情况
"""

import os
import sys
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_training_eval_frequency():
    """测试training和eval频率不同的情况"""
    
    # 模拟配置
    config = {
        'output_dir': '/tmp/test_frequency',
        'wandb': {
            'enabled': True,
            'project': 'qwen_classification_test',
            'run_name': 'test_frequency',
            'tags': ['test', 'frequency'],
            'notes': 'Testing training and eval frequency'
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
        
        # 模拟训练过程：每5步记录training，每30步eval
        total_steps = 100
        eval_interval = 30
        
        for step in range(1, total_steps + 1):
            print(f"\n{'='*40}")
            print(f"📊 Step {step}")
            print(f"{'='*40}")
            
            # 模拟training指标（每个步骤都有）
            training_metrics = {
                "training/loss": 0.5 - step * 0.001,  # 模拟loss下降
                "training/lr": 1e-5,
                "training/epoch": step // 50,
                "training/grad_norm": 1.0 + step * 0.01
            }
            
            # 判断是否是eval步骤
            is_eval_step = (step % eval_interval == 0)
            
            if is_eval_step:
                print(f"🎯 这是eval步骤 (step={step})")
                
                # 模拟eval指标
                eval_metrics = {
                    "eval/overall_loss": 0.3 - step * 0.0005,  # 模拟eval loss下降
                    "eval/overall_accuracy": 0.7 + step * 0.002,  # 模拟accuracy上升
                    "eval/overall_samples": 1000,
                    "eval/overall_correct": int(700 + step * 2)
                }
                
                # 先记录training指标（commit=False）
                monitor.log_metrics(training_metrics, step=step, commit=False)
                print(f"   ✅ 已记录training指标 (commit=False)")
                
                # 再记录eval指标（commit=True）
                monitor.log_metrics(eval_metrics, step=step, commit=True)
                print(f"   ✅ 已记录eval指标 (commit=True)")
                print(f"   📊 eval指标: {list(eval_metrics.keys())}")
                
            else:
                print(f"🏃 这是普通training步骤 (step={step})")
                
                # 只记录training指标
                monitor.log_metrics(training_metrics, step=step, commit=True)
                print(f"   ✅ 已记录training指标 (commit=True)")
            
            print(f"   📈 training指标: {list(training_metrics.keys())}")
            
            # 等待一下让WandB同步
            time.sleep(0.5)
        
        # 结束WandB
        monitor.finish_training()
        
        print(f"\n{'='*50}")
        print("✅ 测试完成")
        print("🔗 请检查WandB界面:")
        print("   📊 training指标应该在每个步骤都有记录")
        print("   🎯 eval指标应该只在步骤 30, 60, 90 有记录")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_training_eval_frequency() 