#!/usr/bin/env python3
"""
测试DeepSpeed配置的完整流程
"""

import os
import sys
import yaml
import json

def test_deepspeed_config_flow():
    """测试DeepSpeed配置的完整流程"""
    print("🔍 测试DeepSpeed配置的完整流程")
    print("="*60)
    
    # 1. 加载YAML配置
    print("📋 1. 加载YAML配置")
    yaml_path = "configs/food101_cosine_hold.yaml"
    
    if not os.path.exists(yaml_path):
        print(f"❌ YAML配置文件不存在: {yaml_path}")
        return
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ YAML配置加载成功")
    print(f"  • 原始DeepSpeed配置: {config.get('deepspeed', 'NOT_FOUND')}")
    print()
    
    # 2. 模拟命令行参数覆盖
    print("📋 2. 模拟命令行参数覆盖")
    cmd_deepspeed_config = "configs/ds_s2.json"
    config['deepspeed'] = cmd_deepspeed_config
    print(f"  • 覆盖后的DeepSpeed配置: {config['deepspeed']}")
    print()
    
    # 3. 验证配置文件存在
    print("📋 3. 验证配置文件存在")
    if os.path.exists(cmd_deepspeed_config):
        print(f"✅ 配置文件存在: {cmd_deepspeed_config}")
        with open(cmd_deepspeed_config, 'r') as f:
            ds_config = json.load(f)
        print(f"  • train_batch_size: {ds_config.get('train_batch_size', 'NOT_FOUND')}")
        print(f"  • train_micro_batch_size_per_gpu: {ds_config.get('train_micro_batch_size_per_gpu', 'NOT_FOUND')}")
        print(f"  • gradient_accumulation_steps: {ds_config.get('gradient_accumulation_steps', 'NOT_FOUND')}")
    else:
        print(f"❌ 配置文件不存在: {cmd_deepspeed_config}")
        return
    print()
    
    # 4. 模拟DeepSpeedTrainer的配置获取
    print("📋 4. 模拟DeepSpeedTrainer的配置获取")
    deepspeed_config = config.get('deepspeed', {})
    
    if isinstance(deepspeed_config, str):
        print(f"  • 配置类型: 文件路径")
        print(f"  • 配置文件路径: {deepspeed_config}")
        print(f"  • 文件是否存在: {os.path.exists(deepspeed_config)}")
        
        if os.path.exists(deepspeed_config):
            with open(deepspeed_config, 'r') as f:
                parsed_config = json.load(f)
            print(f"  ✅ 配置文件解析成功")
            print(f"  • 解析后配置: {parsed_config}")
            
            # 检查必要字段
            required_fields = ['train_batch_size', 'train_micro_batch_size_per_gpu']
            for field in required_fields:
                if field in parsed_config:
                    print(f"  ✅ {field}: {parsed_config[field]}")
                else:
                    print(f"  ❌ {field}: 缺失")
        else:
            print(f"  ❌ 配置文件不存在")
            return
    else:
        print(f"  • 配置类型: 字典")
        print(f"  • 配置内容: {deepspeed_config}")
    
    print()
    print("✅ DeepSpeed配置流程测试完成!")

if __name__ == "__main__":
    test_deepspeed_config_flow() 