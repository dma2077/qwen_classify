#!/usr/bin/env python3
"""
测试eval指标修复
"""

import os
import sys
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_eval_metrics_fix():
    """测试eval指标修复"""
    
    # 模拟配置
    config = {
        'output_dir': '/tmp/test_eval_fix',
        'wandb': {
            'enabled': True,
            'project': 'qwen_classification_test',
            'run_name': 'test_eval_fix',
            'tags': ['test', 'eval', 'fix'],
            'notes': 'Testing eval metrics fix'
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
        
        # 测试记录eval指标
        eval_metrics = {
            "eval/overall_loss": 0.1234,
            "eval/overall_accuracy": 0.8567,
            "eval/overall_samples": 1000,
            "eval/overall_correct": 857
        }
        
        print(f"📊 准备记录eval指标: {list(eval_metrics.keys())}")
        
        # 记录到WandB
        monitor.log_metrics(eval_metrics, step=100, commit=True)
        
        print("✅ eval指标记录完成")
        
        # 等待一下让WandB同步
        time.sleep(3)
        
        # 再次记录一些指标
        eval_metrics_2 = {
            "eval/overall_loss": 0.0987,
            "eval/overall_accuracy": 0.9012,
            "eval/overall_samples": 1000,
            "eval/overall_correct": 901
        }
        
        monitor.log_metrics(eval_metrics_2, step=200, commit=True)
        
        print("✅ 第二次eval指标记录完成")
        
        # 等待同步
        time.sleep(3)
        
        # 结束WandB
        monitor.finish_training()
        
        print("✅ 测试完成")
        print("🔗 请检查WandB界面，应该能看到eval指标")
        print("📊 应该包含: eval/overall_loss, eval/overall_accuracy, eval/overall_samples, eval/overall_correct")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_eval_metrics_fix() 