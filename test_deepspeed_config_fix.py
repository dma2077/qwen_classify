#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试DeepSpeed配置修复
"""

import json
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_deepspeed_config_loading():
    """测试DeepSpeed配置加载"""
    print("🧪 测试DeepSpeed配置加载...")
    
    try:
        # 测试配置文件加载
        config_path = "configs/ds_s2.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("✅ 配置文件加载成功")
        
        # 检查必要的字段
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
                return False
        
        # 检查bf16配置
        if 'bf16' in config:
            bf16_config = config['bf16']
            if isinstance(bf16_config, dict) and 'enabled' in bf16_config:
                print(f"✅ bf16配置: {bf16_config}")
            else:
                print(f"❌ bf16配置格式错误: {bf16_config}")
                return False
        else:
            print("❌ 缺少bf16配置")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")
        return False

def test_deepspeed_trainer_config():
    """测试DeepSpeed训练器配置"""
    print("\n🧪 测试DeepSpeed训练器配置...")
    
    try:
        from training.deepspeed_trainer import DeepSpeedTrainer
        
        # 创建测试配置
        test_config = {
            'deepspeed': 'configs/ds_s2.json',
            'training': {
                'num_epochs': 5
            }
        }
        
        # 创建训练器实例
        trainer = DeepSpeedTrainer(test_config)
        
        # 测试配置获取
        deepspeed_config = trainer._get_deepspeed_config()
        
        print("✅ DeepSpeed训练器配置获取成功")
        print(f"📋 配置类型: {type(deepspeed_config)}")
        print(f"📋 配置内容: {deepspeed_config}")
        
        # 检查配置内容
        if isinstance(deepspeed_config, dict):
            required_fields = [
                'train_batch_size',
                'train_micro_batch_size_per_gpu',
                'gradient_accumulation_steps'
            ]
            
            for field in required_fields:
                if field in deepspeed_config:
                    print(f"✅ {field}: {deepspeed_config[field]}")
                else:
                    print(f"❌ 缺少字段: {field}")
                    return False
            
            return True
        else:
            print(f"❌ 配置类型错误: {type(deepspeed_config)}")
            return False
        
    except Exception as e:
        print(f"❌ DeepSpeed训练器配置测试失败: {e}")
        return False

def test_deepspeed_initialize():
    """测试DeepSpeed初始化（模拟）"""
    print("\n🧪 测试DeepSpeed初始化...")
    
    try:
        # 加载配置
        config_path = "configs/ds_s2.json"
        with open(config_path, 'r') as f:
            deepspeed_config = json.load(f)
        
        print("✅ DeepSpeed配置验证通过")
        print(f"📋 train_batch_size: {deepspeed_config.get('train_batch_size')}")
        print(f"📋 train_micro_batch_size_per_gpu: {deepspeed_config.get('train_micro_batch_size_per_gpu')}")
        print(f"📋 gradient_accumulation_steps: {deepspeed_config.get('gradient_accumulation_steps')}")
        
        # 验证配置完整性
        if (deepspeed_config.get('train_batch_size') and 
            deepspeed_config.get('train_micro_batch_size_per_gpu')):
            print("✅ 批次大小配置完整")
            return True
        else:
            print("❌ 批次大小配置不完整")
            return False
        
    except Exception as e:
        print(f"❌ DeepSpeed初始化测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 测试DeepSpeed配置修复")
    print("=" * 50)
    
    # 测试配置文件加载
    if not test_deepspeed_config_loading():
        print("❌ 配置文件加载测试失败")
        return
    
    # 测试DeepSpeed训练器配置
    if not test_deepspeed_trainer_config():
        print("❌ DeepSpeed训练器配置测试失败")
        return
    
    # 测试DeepSpeed初始化
    if not test_deepspeed_initialize():
        print("❌ DeepSpeed初始化测试失败")
        return
    
    print("\n" + "=" * 50)
    print("✅ 所有DeepSpeed配置测试通过！")

if __name__ == "__main__":
    main() 