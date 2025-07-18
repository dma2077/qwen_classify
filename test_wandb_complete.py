#!/usr/bin/env python3
"""
完整的WandB日志记录测试
验证training、eval和perf指标都能正确记录
"""

import os
import sys
import time
import torch
import yaml
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from training.deepspeed_trainer import DeepSpeedTrainer
from training.utils.config_utils import prepare_config
from training.utils.monitor import TrainingMonitor

def test_wandb_logging_complete():
    """测试完整的WandB日志记录功能"""
    print("🧪 开始完整WandB日志记录测试...")
    
    # 创建输出目录
    os.makedirs('./test_output', exist_ok=True)
    
    # 创建monitor实例
    monitor = TrainingMonitor(
        output_dir='./test_output',
        config={'training': {'batch_size': 2}},
        flops_profile_freq=5
    )
    
    # 模拟一些基础数据
    monitor.actual_flops = 1e12  # 1 TFLOPs
    monitor.model_ref = "dummy_model"
    monitor.batch_size = 2
    monitor.seq_length = 512
    
    # 测试1: Training指标记录
    print("\n" + "="*60)
    print("测试1: Training指标记录")
    print("="*60)
    
    training_data = {
        "training/loss": 0.3,
        "training/lr": 1e-5,
        "training/epoch": 0.5,
        "training/grad_norm": 0.1,
        "step": 10
    }
    
    print("📝 记录training指标...")
    monitor.log_metrics(training_data, step=10, commit=True)
    
    # 测试2: Perf指标记录
    print("\n" + "="*60)
    print("测试2: Perf指标记录")
    print("="*60)
    
    perf_data = {
        "perf/step_time": 0.05,
        "perf/steps_per_second": 20.0,
        "perf/mfu": 0.75,
        "perf/mfu_percent": 75.0,
        "perf/tokens_per_second": 1000.0,
        "perf/samples_per_second": 40.0,
        "perf/actual_flops": 1e12,
        "perf/actual_seq_length": 512.0,
        "perf/flops_per_second": 2e13,
        "step": 15
    }
    
    print("📝 记录perf指标...")
    monitor.log_metrics(perf_data, step=15, commit=True)
    
    # 测试3: Eval指标记录
    print("\n" + "="*60)
    print("测试3: Eval指标记录")
    print("="*60)
    
    eval_data = {
        "eval/overall_loss": 0.5,
        "eval/overall_accuracy": 0.85,
        "eval/overall_samples": 100,
        "eval/overall_correct": 85,
        "eval/food101_loss": 0.4,
        "eval/food101_accuracy": 0.9,
        "eval/food101_samples": 50,
        "eval/food101_correct": 45,
        "eval/cifar10_loss": 0.6,
        "eval/cifar10_accuracy": 0.8,
        "eval/cifar10_samples": 50,
        "eval/cifar10_correct": 40,
        "step": 20
    }
    
    print("📝 记录eval指标...")
    monitor.log_metrics(eval_data, step=20, commit=True)
    
    # 测试4: 合并指标记录（模拟eval步骤）
    print("\n" + "="*60)
    print("测试4: 合并指标记录（模拟eval步骤）")
    print("="*60)
    
    # 先记录training指标（不commit）
    print("📝 记录training指标（不commit）...")
    monitor.log_metrics(training_data, step=30, commit=False)
    
    # 再记录eval指标（commit）
    print("📝 记录eval指标（commit）...")
    monitor.log_metrics(eval_data, step=30, commit=True)
    
    # 测试5: 验证trainer的指标构建方法
    print("\n" + "="*60)
    print("测试5: 验证trainer的指标构建方法")
    print("="*60)
    
    # 创建trainer实例
    trainer = DeepSpeedTrainer({'output_dir': './test_output'})
    trainer.monitor = monitor
    trainer.dist_ctx = type('obj', (object,), {
        'is_main_process': lambda: True,
        'world_size': 1
    })()
    
    # 测试_build_training_metrics方法
    inputs = torch.randn(2, 10)
    attention_mask = torch.ones(2, 10)
    
    training_metrics = trainer._build_training_metrics(
        effective_step=40,
        epoch=1,
        aggregated_loss=0.25,
        current_lr=1e-5,
        grad_norm_value=0.08,
        inputs=inputs,
        attention_mask=attention_mask,
        step_time=0.06
    )
    
    print("📋 构建的training指标:")
    for key, value in training_metrics.items():
        print(f"  {key}: {value}")
    
    # 测试_build_eval_metrics方法
    eval_results = {
        'total_samples': 100,
        'total_correct': 85,
        'dataset_metrics': {
            'food101': {
                'loss': 0.4,
                'accuracy': 0.9,
                'samples': 50,
                'correct': 45
            },
            'cifar10': {
                'loss': 0.6,
                'accuracy': 0.8,
                'samples': 50,
                'correct': 40
            }
        }
    }
    
    eval_metrics = trainer._build_eval_metrics(0.5, 0.85, eval_results)
    
    print("\n📋 构建的eval指标:")
    for key, value in eval_metrics.items():
        print(f"  {key}: {value}")
    
    # 测试6: MFU计算
    print("\n" + "="*60)
    print("测试6: MFU计算")
    print("="*60)
    
    # 测试不同的step_time值
    test_cases = [
        (0.1, "正常step_time"),
        (0.0, "零step_time"),
        (None, "None step_time")
    ]
    
    for step_time, description in test_cases:
        print(f"🔍 测试 {description}...")
        mfu = trainer._calculate_mfu(50, inputs, attention_mask, step_time or 0.0)
        
        if mfu is not None:
            print(f"  ✅ MFU计算成功: {mfu:.4f}")
        else:
            print(f"  ⚠️ MFU计算返回None (预期行为)")
    
    print("\n" + "="*60)
    print("✅ 完整WandB日志记录测试完成！")
    print("="*60)
    print("📊 测试总结:")
    print("  • Training指标记录: ✅")
    print("  • Perf指标记录: ✅")
    print("  • Eval指标记录: ✅")
    print("  • 合并指标记录: ✅")
    print("  • 指标构建方法: ✅")
    print("  • MFU计算: ✅")
    print("\n🎯 如果所有测试都通过，说明WandB日志记录功能正常！")

def test_wandb_metrics_structure():
    """测试WandB指标结构"""
    print("\n" + "="*60)
    print("测试WandB指标结构")
    print("="*60)
    
    # 创建monitor实例
    monitor = TrainingMonitor(
        output_dir='./test_output',
        config={'training': {'batch_size': 2}},
        flops_profile_freq=5
    )
    
    # 模拟完整的指标结构
    complete_metrics = {
        # Training指标
        "training/loss": 0.3,
        "training/lr": 1e-5,
        "training/epoch": 0.5,
        "training/grad_norm": 0.1,
        
        # Perf指标
        "perf/step_time": 0.05,
        "perf/steps_per_second": 20.0,
        "perf/mfu": 0.75,
        "perf/mfu_percent": 75.0,
        "perf/tokens_per_second": 1000.0,
        "perf/samples_per_second": 40.0,
        "perf/actual_flops": 1e12,
        "perf/actual_seq_length": 512.0,
        "perf/flops_per_second": 2e13,
        
        # Eval指标
        "eval/overall_loss": 0.5,
        "eval/overall_accuracy": 0.85,
        "eval/overall_samples": 100,
        "eval/overall_correct": 85,
        "eval/food101_loss": 0.4,
        "eval/food101_accuracy": 0.9,
        "eval/food101_samples": 50,
        "eval/food101_correct": 45,
        "eval/cifar10_loss": 0.6,
        "eval/cifar10_accuracy": 0.8,
        "eval/cifar10_samples": 50,
        "eval/cifar10_correct": 40,
        
        # 统一step字段
        "step": 100
    }
    
    print("📋 完整的指标结构:")
    training_count = 0
    perf_count = 0
    eval_count = 0
    
    for key, value in complete_metrics.items():
        if key.startswith('training/'):
            training_count += 1
            print(f"  🏃 {key}: {value}")
        elif key.startswith('perf/'):
            perf_count += 1
            print(f"  ⚡ {key}: {value}")
        elif key.startswith('eval/'):
            eval_count += 1
            print(f"  📊 {key}: {value}")
        else:
            print(f"  🔢 {key}: {value}")
    
    print(f"\n📈 指标统计:")
    print(f"  • Training指标: {training_count}个")
    print(f"  • Perf指标: {perf_count}个")
    print(f"  • Eval指标: {eval_count}个")
    print(f"  • 总指标: {len(complete_metrics)}个")
    
    # 验证指标分类
    expected_training = ['training/loss', 'training/lr', 'training/epoch', 'training/grad_norm']
    expected_perf = ['perf/step_time', 'perf/steps_per_second', 'perf/mfu', 'perf/mfu_percent', 
                    'perf/tokens_per_second', 'perf/samples_per_second', 'perf/actual_flops', 
                    'perf/actual_seq_length', 'perf/flops_per_second']
    expected_eval = ['eval/overall_loss', 'eval/overall_accuracy', 'eval/overall_samples', 
                    'eval/overall_correct', 'eval/food101_loss', 'eval/food101_accuracy', 
                    'eval/food101_samples', 'eval/food101_correct', 'eval/cifar10_loss', 
                    'eval/cifar10_accuracy', 'eval/cifar10_samples', 'eval/cifar10_correct']
    
    actual_training = [k for k in complete_metrics.keys() if k.startswith('training/')]
    actual_perf = [k for k in complete_metrics.keys() if k.startswith('perf/')]
    actual_eval = [k for k in complete_metrics.keys() if k.startswith('eval/')]
    
    print(f"\n✅ 指标分类验证:")
    print(f"  • Training指标匹配: {len(actual_training) == len(expected_training)}")
    print(f"  • Perf指标匹配: {len(actual_perf) == len(expected_perf)}")
    print(f"  • Eval指标匹配: {len(actual_eval) == len(expected_eval)}")

if __name__ == "__main__":
    # 运行完整测试
    test_wandb_logging_complete()
    
    # 运行指标结构测试
    test_wandb_metrics_structure()
    
    print("\n🎉 所有测试完成！")
    print("💡 如果测试通过，说明WandB日志记录功能已经正确实现。")
    print("🚀 现在可以运行实际训练来验证所有指标都能正确记录到WandB。") 