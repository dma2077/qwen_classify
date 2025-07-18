#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试配置文件加载
"""

import json
import yaml
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_yaml_config():
    """测试YAML配置文件加载"""
    print("🧪 测试YAML配置文件加载...")
    
    try:
        config_path = "configs/food101_cosine_hold.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("✅ YAML配置文件加载成功")
        print(f"📋 DeepSpeed配置: {config.get('deepspeed')}")
        
        return config
        
    except Exception as e:
        print(f"❌ YAML配置文件加载失败: {e}")
        return None

def test_json_config():
    """测试JSON配置文件加载"""
    print("\n🧪 测试JSON配置文件加载...")
    
    try:
        config_path = "configs/ds_minimal.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("✅ JSON配置文件加载成功")
        print(f"📋 配置内容: {config}")
        
        # 检查必要字段
        required_fields = [
            'train_batch_size',
            'train_micro_batch_size_per_gpu',
            'gradient_accumulation_steps'
        ]
        
        for field in required_fields:
            if field in config:
                print(f"✅ {field}: {config[field]}")
            else:
                print(f"❌ 缺少字段: {field}")
                return None
        
        return config
        
    except Exception as e:
        print(f"❌ JSON配置文件加载失败: {e}")
        return None

def test_deepspeed_config_loading():
    """测试DeepSpeed配置加载逻辑"""
    print("\n🧪 测试DeepSpeed配置加载逻辑...")
    
    try:
        # 模拟DeepSpeed训练器的配置加载逻辑
        yaml_config = test_yaml_config()
        if yaml_config is None:
            return False
        
        deepspeed_config_path = yaml_config.get('deepspeed', {}).get('config_file')
        if not deepspeed_config_path:
            print("❌ 未找到DeepSpeed配置文件路径")
            return False
        
        print(f"📋 DeepSpeed配置文件路径: {deepspeed_config_path}")
        
        # 加载DeepSpeed配置文件
        with open(deepspeed_config_path, 'r') as f:
            deepspeed_config = json.load(f)
        
        print("✅ DeepSpeed配置文件加载成功")
        print(f"📋 DeepSpeed配置内容: {deepspeed_config}")
        
        # 验证配置
        if (deepspeed_config.get('train_batch_size') and 
            deepspeed_config.get('train_micro_batch_size_per_gpu')):
            print("✅ DeepSpeed配置验证通过")
            return True
        else:
            print("❌ DeepSpeed配置验证失败")
            return False
        
    except Exception as e:
        print(f"❌ DeepSpeed配置加载测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 测试配置文件加载")
    print("=" * 50)
    
    # 测试YAML配置
    yaml_config = test_yaml_config()
    if yaml_config is None:
        return
    
    # 测试JSON配置
    json_config = test_json_config()
    if json_config is None:
        return
    
    # 测试DeepSpeed配置加载
    if test_deepspeed_config_loading():
        print("\n" + "=" * 50)
        print("✅ 所有配置文件加载测试通过！")
    else:
        print("\n" + "=" * 50)
        print("❌ 配置文件加载测试失败")

if __name__ == "__main__":
    main() 