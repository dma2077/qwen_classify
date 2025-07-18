#!/usr/bin/env python3
"""
测试简化后的配置流程
"""

import os
import sys
import yaml
import json

def test_simplified_config():
    """测试简化后的配置流程"""
    print("🔍 测试简化后的配置流程")
    print("="*50)
    
    # 模拟命令行参数
    config_file = "configs/food101_cosine_hold.yaml"
    deepspeed_config = "configs/ds_s2.json"
    
    print(f"📋 模拟命令行参数:")
    print(f"  • config: {config_file}")
    print(f"  • deepspeed_config: {deepspeed_config}")
    print()
    
    # 1. 加载YAML配置
    print("📋 1. 加载YAML配置")
    if not os.path.exists(config_file):
        print(f"❌ YAML配置文件不存在: {config_file}")
        return
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ YAML配置加载成功")
    print(f"  • 是否包含deepspeed配置: {'deepspeed' in config}")
    print()
    
    # 2. 验证DeepSpeed配置文件
    print("📋 2. 验证DeepSpeed配置文件")
    if not os.path.exists(deepspeed_config):
        print(f"❌ DeepSpeed配置文件不存在: {deepspeed_config}")
        return
    
    with open(deepspeed_config, 'r') as f:
        ds_config = json.load(f)
    
    print(f"✅ DeepSpeed配置文件加载成功")
    print(f"  • train_batch_size: {ds_config.get('train_batch_size', 'NOT_FOUND')}")
    print(f"  • train_micro_batch_size_per_gpu: {ds_config.get('train_micro_batch_size_per_gpu', 'NOT_FOUND')}")
    print(f"  • gradient_accumulation_steps: {ds_config.get('gradient_accumulation_steps', 'NOT_FOUND')}")
    print()
    
    # 3. 将DeepSpeed配置添加到config中
    print("📋 3. 将DeepSpeed配置添加到config中")
    config['deepspeed'] = deepspeed_config
    print(f"✅ DeepSpeed配置已添加")
    print(f"  • config['deepspeed']: {config['deepspeed']}")
    print()
    
    # 4. 验证配置
    print("📋 4. 验证配置")
    required_fields = ['train_batch_size', 'train_micro_batch_size_per_gpu']
    missing_fields = [field for field in required_fields if field not in ds_config]
    
    if missing_fields:
        print(f"❌ DeepSpeed配置文件缺少必要字段: {missing_fields}")
        return
    else:
        print(f"✅ DeepSpeed配置文件包含所有必要字段")
    
    print()
    print("✅ 简化后的配置流程测试完成!")

if __name__ == "__main__":
    test_simplified_config() 