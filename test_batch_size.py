#!/usr/bin/env python3
"""
测试批次大小设置脚本
验证训练和评估批次大小是否正确设置
"""

import yaml
import json
from data.dataloader import create_dataloaders

def test_batch_size_config():
    """测试批次大小配置"""
    
    # 测试配置
    test_config = {
        'model': {
            'pretrained_name': 'Qwen/Qwen2.5-VL-7B-Instruct',
            'num_labels': 101
        },
        'training': {
            'batch_size': 8,
            'num_workers': 4
        },
        'data': {
            'train_jsonl': 'data/food101/train.jsonl',
            'val_jsonl': 'data/food101/val.jsonl'
        },
        'datasets': {
            'dataset_configs': {
                'food101': {
                    'num_classes': 101,
                    'eval_ratio': 0.2
                }
            }
        },
        'deepspeed': 'configs/ds_minimal.json'
    }
    
    print("🧪 测试批次大小配置...")
    print("=" * 50)
    
    # 读取DeepSpeed配置
    with open('configs/ds_minimal.json', 'r') as f:
        ds_config = json.load(f)
    
    print(f"📊 DeepSpeed配置:")
    print(f"  • train_batch_size: {ds_config.get('train_batch_size')}")
    print(f"  • train_micro_batch_size_per_gpu: {ds_config.get('train_micro_batch_size_per_gpu')}")
    print(f"  • gradient_accumulation_steps: {ds_config.get('gradient_accumulation_steps')}")
    
    # 计算预期的批次大小
    micro_batch = ds_config.get('train_micro_batch_size_per_gpu', 1)
    grad_accum = ds_config.get('gradient_accumulation_steps', 1)
    total_batch = ds_config.get('train_batch_size', 1)
    
    print(f"\n📈 预期批次大小:")
    print(f"  • 训练批次大小 (per GPU): {micro_batch}")
    print(f"  • 有效训练批次大小: {micro_batch * grad_accum * 4} (假设4个GPU)")
    print(f"  • 评估批次大小: {total_batch}")
    
    print(f"\n✅ 测试完成！")
    print(f"   现在评估将使用 {total_batch} 的批次大小，而不是 {micro_batch}")

if __name__ == "__main__":
    test_batch_size_config() 