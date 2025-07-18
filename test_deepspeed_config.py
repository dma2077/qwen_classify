#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试DeepSpeed配置
"""

import json
import sys
import os

def test_deepspeed_config(config_path):
    """测试DeepSpeed配置文件"""
    print(f"🧪 测试DeepSpeed配置文件: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("✅ 配置文件加载成功")
        
        # 检查bf16配置
        if 'bf16' in config:
            bf16_config = config['bf16']
            print(f"📋 bf16配置: {bf16_config}")
            
            if isinstance(bf16_config, dict):
                print("✅ bf16配置格式正确（字典格式）")
                
                # 检查必要的字段
                required_fields = ['enabled']
                for field in required_fields:
                    if field in bf16_config:
                        print(f"  ✅ {field}: {bf16_config[field]}")
                    else:
                        print(f"  ⚠️ 缺少字段: {field}")
            else:
                print(f"❌ bf16配置格式错误，期望字典，实际: {type(bf16_config)}")
                return False
        else:
            print("⚠️ 未找到bf16配置")
        
        # 检查其他重要配置
        print(f"📋 训练批次大小: {config.get('train_batch_size', 'N/A')}")
        print(f"📋 每GPU微批次大小: {config.get('train_micro_batch_size_per_gpu', 'N/A')}")
        print(f"📋 梯度累积步数: {config.get('gradient_accumulation_steps', 'N/A')}")
        
        # 检查ZeRO配置
        if 'zero_optimization' in config:
            zero_config = config['zero_optimization']
            print(f"📋 ZeRO阶段: {zero_config.get('stage', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 测试DeepSpeed配置文件")
    print("=" * 50)
    
    # 测试两个配置文件
    configs = [
        "configs/ds_s2.json",
        "configs/ds_config_zero2.json"
    ]
    
    all_passed = True
    for config_path in configs:
        if os.path.exists(config_path):
            if not test_deepspeed_config(config_path):
                all_passed = False
        else:
            print(f"❌ 配置文件不存在: {config_path}")
            all_passed = False
        print()
    
    if all_passed:
        print("✅ 所有DeepSpeed配置文件测试通过！")
    else:
        print("❌ 部分配置文件测试失败")

if __name__ == "__main__":
    main() 