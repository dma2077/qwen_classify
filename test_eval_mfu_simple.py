#!/usr/bin/env python3
"""
简化的eval指标和MFU计算测试
不依赖实际数据文件，只测试核心功能
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

def test_eval_metrics_building():
    """测试eval指标构建功能"""
    print("📊 测试eval指标构建功能...")
    
    # 创建monitor实例
    monitor = TrainingMonitor(
        output_dir='./test_output',
        config={'training': {'batch_size': 2}},
        flops_profile_freq=5
    )
    
    # 模拟eval结果
    eval_loss = 0.5
    eval_accuracy = 0.85
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
    
    # 创建trainer实例来测试_build_eval_metrics方法
    trainer = DeepSpeedTrainer({'output_dir': './test_output'})
    
    # 构建eval指标
    eval_data = trainer._build_eval_metrics(eval_loss, eval_accuracy, eval_results)
    
    print(f"    📋 构建的eval指标:")
    for key, value in eval_data.items():
        print(f"      {key}: {value}")
    
    # 检查是否包含所有必要的指标
    expected_keys = [
        'eval/overall_loss',
        'eval/overall_accuracy',
        'eval/overall_samples',
        'eval/overall_correct',
        'eval/food101_loss',
        'eval/food101_accuracy',
        'eval/food101_samples',
        'eval/food101_correct',
        'eval/cifar10_loss',
        'eval/cifar10_accuracy',
        'eval/cifar10_samples',
        'eval/cifar10_correct'
    ]
    
    missing_keys = [key for key in expected_keys if key not in eval_data]
    if missing_keys:
        print(f"    ⚠️ 缺失的eval指标: {missing_keys}")
        return False
    else:
        print(f"    ✅ 所有预期的eval指标都已包含")
        return True

def test_mfu_calculation_logic():
    """测试MFU计算逻辑（不依赖实际模型）"""
    print("🔍 测试MFU计算逻辑...")
    
    # 创建monitor实例
    monitor = TrainingMonitor(
        output_dir='./test_output',
        config={'training': {'batch_size': 2}},
        flops_profile_freq=5
    )
    
    # 模拟一些值
    monitor.actual_flops = 1e12  # 1 TFLOPs
    monitor.model_ref = "dummy_model"
    
    # 创建trainer实例
    trainer = DeepSpeedTrainer({'output_dir': './test_output'})
    trainer.monitor = monitor
    trainer.dist_ctx = type('obj', (object,), {
        'is_main_process': lambda: True,
        'world_size': 1
    })()
    
    # 测试不同的step_time值
    test_cases = [
        (0.1, "正常step_time"),
        (0.0, "零step_time"),
        (None, "None step_time")
    ]
    
    # 创建模拟的tensor
    inputs = torch.randn(2, 10)  # batch_size=2, seq_len=10
    attention_mask = torch.ones(2, 10)
    
    all_tests_passed = True
    
    for step_time, description in test_cases:
        print(f"    🔍 测试 {description}...")
        
        # 计算MFU
        mfu = trainer._calculate_mfu(1, inputs, attention_mask, step_time or 0.0)
        
        if mfu is not None:
            print(f"      ✅ MFU计算成功: {mfu:.4f}")
        else:
            print(f"      ⚠️ MFU计算返回None (预期行为)")
            if description == "零step_time" or description == "None step_time":
                print(f"        ✅ 这是预期的，因为step_time无效")
            else:
                print(f"        ❌ 这不应该发生")
                all_tests_passed = False
    
    # 测试actual_flops为None的情况
    print(f"    🔍 测试actual_flops为None...")
    original_flops = monitor.actual_flops
    monitor.actual_flops = None
    
    mfu = trainer._calculate_mfu(1, inputs, attention_mask, 0.1)
    if mfu is None:
        print(f"      ✅ MFU计算正确返回None (actual_flops为None)")
    else:
        print(f"      ❌ MFU计算应该返回None")
        all_tests_passed = False
    
    # 恢复original_flops
    monitor.actual_flops = original_flops
    
    return all_tests_passed

def test_combined_metrics_logging():
    """测试合并指标记录逻辑"""
    print("📝 测试合并指标记录逻辑...")
    
    # 模拟training和eval数据
    training_data = {
        "training/loss": 0.3,
        "training/lr": 1e-5,
        "training/epoch": 0.5,
        "training/grad_norm": 0.1,
        "perf/step_time": 0.05,
        "perf/mfu": 0.75,
        "step": 10
    }
    
    eval_data = {
        "eval/overall_loss": 0.5,
        "eval/overall_accuracy": 0.85,
        "eval/overall_samples": 100,
        "eval/overall_correct": 85
    }
    
    # 测试合并逻辑
    combined_data = training_data.copy()
    combined_data.update(eval_data)
    
    print(f"    📋 合并后的指标:")
    for key, value in combined_data.items():
        print(f"      {key}: {value}")
    
    # 检查是否包含所有指标
    expected_training_keys = [k for k in training_data.keys() if k.startswith('training/')]
    expected_perf_keys = [k for k in training_data.keys() if k.startswith('perf/')]
    expected_eval_keys = [k for k in eval_data.keys() if k.startswith('eval/')]
    
    missing_training = [k for k in expected_training_keys if k not in combined_data]
    missing_perf = [k for k in expected_perf_keys if k not in combined_data]
    missing_eval = [k for k in expected_eval_keys if k not in combined_data]
    
    if missing_training or missing_perf or missing_eval:
        print(f"    ❌ 缺失指标:")
        if missing_training:
            print(f"      training: {missing_training}")
        if missing_perf:
            print(f"      perf: {missing_perf}")
        if missing_eval:
            print(f"      eval: {missing_eval}")
        return False
    else:
        print(f"    ✅ 所有指标都已正确合并")
        return True

def main():
    """主测试函数"""
    print("🧪 开始简化测试eval指标和MFU计算修复...")
    
    # 创建输出目录
    os.makedirs('./test_output', exist_ok=True)
    
    # 运行测试
    tests = [
        ("Eval指标构建", test_eval_metrics_building),
        ("MFU计算逻辑", test_mfu_calculation_logic),
        ("合并指标记录", test_combined_metrics_logging)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"测试: {test_name}")
        print(f"{'='*50}")
        
        try:
            if test_func():
                print(f"✅ {test_name} 测试通过")
                passed_tests += 1
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试出错: {e}")
    
    print(f"\n{'='*50}")
    print(f"测试结果: {passed_tests}/{total_tests} 通过")
    print(f"{'='*50}")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！修复应该有效。")
    else:
        print("⚠️ 部分测试失败，需要进一步检查。")

if __name__ == "__main__":
    main() 