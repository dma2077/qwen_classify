#!/usr/bin/env python3
"""
测试eval指标和MFU计算的修复
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
from models.qwen2_5_vl_classify import Qwen2_5_VLForImageClassification
from data.dataloader import create_dataloaders
from optimizer.optimizer import create_optimizer_and_scheduler

def test_eval_mfu_fix():
    """测试eval指标和MFU计算的修复"""
    
    # 使用简单的测试配置
    test_config = {
        'model': {
            'pretrained_name': 'Qwen/Qwen2.5-VL-7B-Instruct',
            'num_classes': 10,
            'use_flash_attention': False,
            'use_cache': False
        },
        'training': {
            'num_epochs': 1,
            'batch_size': 2,
            'gradient_accumulation_steps': 1,
            'learning_rate': 1e-5,
            'weight_decay': 0.01,
            'warmup_steps': 10,
            'max_grad_norm': 1.0,
            'logging_steps': 5,
            'eval_steps': 10,
            'save_steps': 20
        },
        'data': {
            'train_jsonl': 'data/sample_data/train.jsonl',
            'val_jsonl': 'data/sample_data/val.jsonl',
            'max_length': 512,
            'image_size': 224
        },
        'output_dir': './test_output',
        'save_best_only': False,
        'best_metric': 'overall_accuracy',
        'full_eval_at_end': False,
        'enable_dataset_metrics': True,
        'monitor': {
            'use_wandb': False,  # 禁用wandb进行测试
            'all_freq': {
                'training_log_freq': 5,
                'eval_log_freq': 10,
                'perf_log_freq': 5,
                'gpu_log_freq': 10
            },
            'flops_profile_freq': 5
        }
    }
    
    print("🧪 开始测试eval指标和MFU计算修复...")
    
    # 创建输出目录
    os.makedirs(test_config['output_dir'], exist_ok=True)
    
    # 准备配置
    config = prepare_config(test_config)
    
    # 创建模型
    print("📦 创建模型...")
    model = Qwen2_5_VLForImageClassification(
        pretrained_model_name=config['model']['name'],
        num_labels=config['model']['num_classes'],
        loss_config={'type': 'cross_entropy'},
        dataset_configs={},
        enable_logits_masking=False
    )
    
    # 创建数据加载器
    print("📊 创建数据加载器...")
    train_loader, val_loader = create_dataloaders(config)
    
    # 创建优化器和调度器
    print("⚙️ 创建优化器和调度器...")
    optimizer, lr_scheduler = create_optimizer_and_scheduler(model, config)
    
    # 创建训练器
    print("🚀 创建训练器...")
    trainer = DeepSpeedTrainer(config)
    trainer.setup_model(model, train_loader, val_loader, optimizer, lr_scheduler)
    
    # 测试MFU计算
    print("\n🔍 测试MFU计算...")
    test_mfu_calculation(trainer)
    
    # 测试eval指标构建
    print("\n📊 测试eval指标构建...")
    test_eval_metrics_building(trainer)
    
    print("\n✅ 测试完成！")

def test_mfu_calculation(trainer):
    """测试MFU计算"""
    print("  📈 测试MFU计算功能...")
    
    # 获取一个batch进行测试
    batch = next(iter(trainer.train_loader))
    forward_kwargs, inputs, attention_mask, labels = trainer._prepare_batch_data(batch)
    
    # 测试不同的step_time值
    test_cases = [
        (0.1, "正常step_time"),
        (0.0, "零step_time"),
        (None, "None step_time")
    ]
    
    for step_time, description in test_cases:
        print(f"    🔍 测试 {description}...")
        
        # 模拟step_start_time
        if step_time is not None:
            trainer.monitor.step_start_time = time.time() - step_time
        else:
            trainer.monitor.step_start_time = None
        
        # 计算MFU
        mfu = trainer._calculate_mfu(1, inputs, attention_mask, step_time or 0.0)
        
        if mfu is not None:
            print(f"      ✅ MFU计算成功: {mfu:.4f}")
        else:
            print(f"      ⚠️ MFU计算返回None (预期行为)")
    
    # 检查actual_flops是否已设置
    if trainer.monitor.actual_flops is not None:
        print(f"      ✅ actual_flops已设置: {trainer.monitor.actual_flops:.2e}")
    else:
        print(f"      ⚠️ actual_flops未设置")

def test_eval_metrics_building(trainer):
    """测试eval指标构建"""
    print("  📊 测试eval指标构建功能...")
    
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
    else:
        print(f"    ✅ 所有预期的eval指标都已包含")

if __name__ == "__main__":
    test_eval_mfu_fix() 