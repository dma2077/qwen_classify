#!/usr/bin/env python3
"""
测试DeepSpeed配置加载逻辑
"""

import os
import sys
import yaml
import json
import argparse

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_config_loading():
    """测试配置加载逻辑"""
    print("🔍 测试DeepSpeed配置加载逻辑")
    print("="*50)
    
    # 测试1: 从YAML文件加载
    print("📋 测试1: 从YAML文件加载配置")
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
                print(f"  • 解析后的配置: {ds_config}")
            else:
                print(f"  ❌ 配置文件不存在: {deepspeed_config}")
        else:
            print(f"  • 配置类型: 字典")
            print(f"  • 配置内容: {deepspeed_config}")
    else:
        print(f"  ❌ YAML配置文件不存在: {yaml_config_path}")
    
    print()
    
    # 测试2: 模拟命令行参数覆盖
    print("📋 测试2: 命令行参数覆盖")
    cmd_deepspeed_config = "configs/ds_s2.json"
    
    if os.path.exists(cmd_deepspeed_config):
        print(f"  • 命令行指定的配置: {cmd_deepspeed_config}")
        with open(cmd_deepspeed_config, 'r') as f:
            ds_config = json.load(f)
        print(f"  • 解析后的配置: {ds_config}")
        
        # 检查必要的字段
        required_fields = ['train_batch_size', 'train_micro_batch_size_per_gpu']
        for field in required_fields:
            if field in ds_config:
                print(f"  ✅ {field}: {ds_config[field]}")
            else:
                print(f"  ❌ {field}: 缺失")
    else:
        print(f"  ❌ 命令行指定的配置文件不存在: {cmd_deepspeed_config}")
    
    print()
    
    # 测试3: 测试所有可用的DeepSpeed配置文件
    print("📋 测试3: 检查所有可用的DeepSpeed配置文件")
    ds_config_dir = "configs"
    ds_configs = [f for f in os.listdir(ds_config_dir) if f.endswith('.json') and 'ds' in f]
    
    for config_file in ds_configs:
        config_path = os.path.join(ds_config_dir, config_file)
        print(f"  📄 {config_file}:")
        
        try:
            with open(config_path, 'r') as f:
                ds_config = json.load(f)
            
            # 检查必要字段
            train_batch_size = ds_config.get('train_batch_size', 'NOT_FOUND')
            train_micro_batch_size_per_gpu = ds_config.get('train_micro_batch_size_per_gpu', 'NOT_FOUND')
            gradient_accumulation_steps = ds_config.get('gradient_accumulation_steps', 'NOT_FOUND')
            
            print(f"    • train_batch_size: {train_batch_size}")
            print(f"    • train_micro_batch_size_per_gpu: {train_micro_batch_size_per_gpu}")
            print(f"    • gradient_accumulation_steps: {gradient_accumulation_steps}")
            
            # 验证配置
            if train_batch_size != 'NOT_FOUND' and train_micro_batch_size_per_gpu != 'NOT_FOUND':
                print(f"    ✅ 配置有效")
            else:
                print(f"    ❌ 配置无效 - 缺少必要字段")
                
        except Exception as e:
            print(f"    ❌ 解析失败: {e}")
        
        print()

if __name__ == "__main__":
    test_config_loading() 