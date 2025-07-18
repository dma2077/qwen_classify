#!/usr/bin/env python3
"""
简单的DeepSpeed配置测试
"""

import os
import sys
import json
import yaml

def test_config():
    """测试配置加载"""
    print("🔍 测试DeepSpeed配置加载")
    print("="*50)
    
    # 测试1: 直接加载ds_s2.json
    print("📋 测试1: 直接加载ds_s2.json")
    ds_config_path = "configs/ds_s2.json"
    
    if os.path.exists(ds_config_path):
        with open(ds_config_path, 'r') as f:
            ds_config = json.load(f)
        
        print(f"  ✅ 配置文件加载成功")
        print(f"  • train_batch_size: {ds_config.get('train_batch_size', 'NOT_FOUND')}")
        print(f"  • train_micro_batch_size_per_gpu: {ds_config.get('train_micro_batch_size_per_gpu', 'NOT_FOUND')}")
        print(f"  • gradient_accumulation_steps: {ds_config.get('gradient_accumulation_steps', 'NOT_FOUND')}")
    else:
        print(f"  ❌ 配置文件不存在: {ds_config_path}")
    
    print()
    
    # 测试2: 从YAML加载
    print("📋 测试2: 从YAML加载配置")
    yaml_config_path = "configs/food101_cosine_hold.yaml"
    
    if os.path.exists(yaml_config_path):
        with open(yaml_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        deepspeed_config = config.get('deepspeed', {})
        print(f"  • YAML中的DeepSpeed配置: {deepspeed_config}")
        
        if isinstance(deepspeed_config, str):
            print(f"  • 配置类型: 文件路径")
            if os.path.exists(deepspeed_config):
                with open(deepspeed_config, 'r') as f:
                    ds_config = json.load(f)
                print(f"  ✅ 从YAML路径加载成功")
                print(f"  • train_batch_size: {ds_config.get('train_batch_size', 'NOT_FOUND')}")
                print(f"  • train_micro_batch_size_per_gpu: {ds_config.get('train_micro_batch_size_per_gpu', 'NOT_FOUND')}")
            else:
                print(f"  ❌ 配置文件不存在: {deepspeed_config}")
    else:
        print(f"  ❌ YAML配置文件不存在: {yaml_config_path}")
    
    print()
    
    # 测试3: 模拟命令行参数
    print("📋 测试3: 模拟命令行参数")
    cmd_deepspeed_config = "configs/ds_s2.json"
    
    if os.path.exists(cmd_deepspeed_config):
        with open(cmd_deepspeed_config, 'r') as f:
            ds_config = json.load(f)
        
        print(f"  ✅ 命令行配置加载成功")
        print(f"  • train_batch_size: {ds_config.get('train_batch_size', 'NOT_FOUND')}")
        print(f"  • train_micro_batch_size_per_gpu: {ds_config.get('train_micro_batch_size_per_gpu', 'NOT_FOUND')}")
    else:
        print(f"  ❌ 命令行配置文件不存在: {cmd_deepspeed_config}")

if __name__ == "__main__":
    test_config() 