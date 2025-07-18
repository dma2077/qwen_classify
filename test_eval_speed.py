#!/usr/bin/env python3
"""
测试评估速度和配置传递
"""

import os
import sys
import json
import yaml
import torch
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.dataloader import create_dataloaders
from models.qwen2_5_vl_classify import Qwen2_5_VLForImageClassification

def test_eval_speed():
    """测试评估速度"""
    print("🔍 测试评估速度和配置传递")
    print("="*60)
    
    # 加载配置
    config_path = "configs/food101_cosine_hold.yaml"
    deepspeed_config_path = "configs/ds_minimal.json"
    
    print(f"📋 加载配置文件:")
    print(f"  • YAML配置: {config_path}")
    print(f"  • DeepSpeed配置: {deepspeed_config_path}")
    
    # 加载YAML配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载DeepSpeed配置
    with open(deepspeed_config_path, 'r') as f:
        deepspeed_config = json.load(f)
    
    # 将DeepSpeed配置添加到config中
    config['deepspeed'] = deepspeed_config_path
    
    print(f"\n📊 DeepSpeed配置:")
    print(f"  • train_batch_size: {deepspeed_config.get('train_batch_size')}")
    print(f"  • train_micro_batch_size_per_gpu: {deepspeed_config.get('train_micro_batch_size_per_gpu')}")
    print(f"  • gradient_accumulation_steps: {deepspeed_config.get('gradient_accumulation_steps')}")
    
    print(f"\n📊 评估配置:")
    dataset_configs = config.get('datasets', {}).get('dataset_configs', {})
    for dataset_name, dataset_config in dataset_configs.items():
        eval_ratio = dataset_config.get('eval_ratio', 1.0)
        print(f"  • {dataset_name}: eval_ratio={eval_ratio}")
    
    # 创建数据加载器
    print(f"\n🔧 创建数据加载器...")
    start_time = time.time()
    
    try:
        train_loader, val_loader = create_dataloaders(config)
        load_time = time.time() - start_time
        
        print(f"✅ 数据加载器创建成功 (耗时: {load_time:.2f}s)")
        print(f"  • 训练集: {len(train_loader.dataset):,} 样本")
        print(f"  • 验证集: {len(val_loader.dataset):,} 样本")
        print(f"  • 验证批次数: {len(val_loader)}")
        print(f"  • 批次大小: {val_loader.batch_size}")
        
        # 计算预期的评估时间
        estimated_batches = len(val_loader)
        estimated_time_per_batch = 2.0  # 假设每批次2秒（保守估计）
        estimated_total_time = estimated_batches * estimated_time_per_batch
        
        print(f"\n⏱️ 评估时间估算:")
        print(f"  • 预计批次数: {estimated_batches}")
        print(f"  • 预计每批次时间: {estimated_time_per_batch}s")
        print(f"  • 预计总评估时间: {estimated_total_time:.1f}s ({estimated_total_time/60:.1f}分钟)")
        
        # 测试一个批次的处理时间
        print(f"\n🧪 测试单批次处理时间...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建模型（简化版本，只用于测试）
        model = Qwen2_5_VLForImageClassification(
            pretrained_model_name=config['model']['pretrained_name'],
            num_labels=config['model']['num_labels']
        )
        model.to(device)
        model.eval()
        
        # 测试几个批次
        batch_times = []
        for i, batch in enumerate(val_loader):
            if i >= 3:  # 只测试前3个批次
                break
                
            start_batch = time.time()
            
            # 移动数据到设备
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            # 前向传播
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels
                )
            
            batch_time = time.time() - start_batch
            batch_times.append(batch_time)
            
            print(f"  • 批次 {i+1}: {batch_time:.2f}s")
        
        if batch_times:
            avg_batch_time = sum(batch_times) / len(batch_times)
            print(f"  • 平均批次时间: {avg_batch_time:.2f}s")
            
            # 重新计算总评估时间
            real_estimated_time = len(val_loader) * avg_batch_time
            print(f"  • 实际预计总评估时间: {real_estimated_time:.1f}s ({real_estimated_time/60:.1f}分钟)")
        
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_eval_speed() 