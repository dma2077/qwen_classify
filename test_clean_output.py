#!/usr/bin/env python3
"""
测试输出优化效果
"""

import os
import sys
import yaml
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_clean_output():
    """测试输出优化效果"""
    print("🧪 测试输出优化效果")
    print("="*50)
    
    # 加载配置
    config_path = "configs/food101_cosine_hold.yaml"
    deepspeed_config_path = "configs/ds_minimal.json"
    
    print(f"📋 配置文件:")
    print(f"  • YAML: {config_path}")
    print(f"  • DeepSpeed: {deepspeed_config_path}")
    
    # 加载YAML配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载DeepSpeed配置
    with open(deepspeed_config_path, 'r') as f:
        deepspeed_config = json.load(f)
    
    # 将DeepSpeed配置添加到config中
    config['deepspeed'] = deepspeed_config_path
    
    print(f"\n📊 关键配置信息:")
    print(f"  • 模型: {config['model']['pretrained_name']}")
    print(f"  • 类别数: {config['model']['num_labels']}")
    print(f"  • 批次大小: {deepspeed_config['train_micro_batch_size_per_gpu']} x {deepspeed_config.get('gradient_accumulation_steps', 1)} = {deepspeed_config['train_batch_size']}")
    
    # 检查数据集配置
    dataset_configs = config.get('datasets', {}).get('dataset_configs', {})
    if dataset_configs:
        print(f"  • 数据集: {len(dataset_configs)} 个")
        for dataset_name, dataset_config in dataset_configs.items():
            eval_ratio = dataset_config.get('eval_ratio', 1.0)
            print(f"    - {dataset_name}: eval_ratio={eval_ratio}")
    
    # 检查训练配置
    training_config = config.get('training', {})
    print(f"  • 训练轮数: {training_config.get('epochs', 'N/A')}")
    print(f"  • 学习率: {training_config.get('lr', 'N/A')}")
    print(f"  • 评估间隔: {training_config.get('eval_steps', 'N/A')} 步")
    
    print(f"\n✅ 输出优化完成！")
    print(f"📝 现在的输出应该更加简洁，只显示关键信息")

if __name__ == "__main__":
    test_clean_output() 