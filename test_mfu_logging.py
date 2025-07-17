#!/usr/bin/env python3
"""
测试MFU记录
"""

import os
import sys
import time
import torch
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_mfu_logging():
    """测试MFU记录"""
    
    # 模拟配置
    config = {
        'output_dir': '/tmp/test_mfu',
        'wandb': {
            'enabled': True,
            'project': 'qwen_classification_test',
            'run_name': 'test_mfu_logging',
            'tags': ['test', 'mfu'],
            'notes': 'Testing MFU logging'
        },
        'monitor': {
            'freq': {
                'perf_log_freq': 5,  # 每5步记录一次性能指标
                'training_log_freq': 5
            }
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
        
        # 模拟一个简单的模型用于测试
        class MockModel:
            def __init__(self):
                self.processor = None
                
            def parameters(self):
                return [torch.randn(1000, 1000)]
                
            def __call__(self, **kwargs):
                return type('MockOutput', (), {'loss': torch.tensor(0.5)})()
        
        # 设置模型引用
        mock_model = MockModel()
        monitor.set_model_ref(mock_model)
        
        # 模拟batch数据
        batch_example = {
            "input_ids": torch.randint(0, 1000, (4, 512)),
            "attention_mask": torch.ones(4, 512),
            "pixel_values": torch.randn(4, 3, 224, 224),
            "labels": torch.randint(0, 10, (4,))
        }
        
        # 进行FLOPs profiling
        print("🔍 开始FLOPs profiling...")
        monitor.profile_model_flops(batch_example)
        
        # 模拟训练步骤
        for step in range(1, 21):
            print(f"\n{'='*40}")
            print(f"📊 Step {step}")
            print(f"{'='*40}")
            
            # 模拟训练指标
            training_metrics = {
                "training/loss": 0.5 - step * 0.01,
                "training/lr": 1e-5,
                "training/epoch": step // 10,
                "training/grad_norm": 1.0 + step * 0.01
            }
            
            # 模拟性能指标
            step_time = 0.1 + step * 0.01  # 模拟步骤时间
            perf_metrics = {
                "perf/step_time": step_time,
                "perf/steps_per_second": 1.0 / step_time,
                "perf/mfu": 0.3 + step * 0.02,  # 模拟MFU
                "perf/mfu_percent": (0.3 + step * 0.02) * 100,
                "perf/tokens_per_second": 1000 + step * 100,
                "perf/samples_per_second": 10 + step,
                "perf/actual_flops": 1e12 + step * 1e10,
                "perf/actual_seq_length": 512,
                "perf/flops_per_second": (1e12 + step * 1e10) / step_time
            }
            
            # 合并指标
            combined_metrics = training_metrics.copy()
            combined_metrics.update(perf_metrics)
            
            print(f"📊 准备记录指标 (step={step}):")
            print(f"   🏃 training指标: {list(training_metrics.keys())}")
            print(f"   ⚡ perf指标: {list(perf_metrics.keys())}")
            print(f"   🔢 总指标数量: {len(combined_metrics)}")
            
            # 记录到WandB
            monitor.log_metrics(combined_metrics, step=step, commit=True)
            
            print(f"✅ 第{step}步指标记录完成")
            
            # 等待一下让WandB同步
            time.sleep(1)
        
        # 结束WandB
        monitor.finish_training()
        
        print(f"\n{'='*50}")
        print("✅ 测试完成")
        print("🔗 请检查WandB界面，应该能看到:")
        print("   📊 training指标: training/loss, training/lr, training/epoch, training/grad_norm")
        print("   ⚡ perf指标: perf/mfu, perf/mfu_percent, perf/step_time, perf/steps_per_second")
        print("   🎯 MFU应该在perf组中显示")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mfu_logging() 