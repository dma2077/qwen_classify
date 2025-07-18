#!/usr/bin/env python3
"""
测试complete_train.py的配置加载流程
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

from training.utils.config_utils import prepare_config

def test_config_loading():
    """测试配置加载流程"""
    print("🔍 测试complete_train.py配置加载流程")
    print("="*60)
    
    # 模拟命令行参数
    class MockArgs:
        def __init__(self):
            self.config = "configs/food101_cosine_hold.yaml"
            self.deepspeed_config = "configs/ds_s2.json"
            self.local_rank = -1
            self.resume_from = None
            self.seed = 42
    
    args = MockArgs()
    
    print(f"📋 模拟命令行参数:")
    print(f"  • config: {args.config}")
    print(f"  • deepspeed_config: {args.deepspeed_config}")
    print(f"  • seed: {args.seed}")
    print()
    
    # 步骤1: 加载YAML配置
    print("📋 步骤1: 加载YAML配置")
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"  ✅ YAML配置加载成功")
        print(f"  • 原始DeepSpeed配置: {config.get('deepspeed', 'NOT_FOUND')}")
    else:
        print(f"  ❌ YAML配置文件不存在: {args.config}")
        return
    print()
    
    # 步骤2: 命令行参数覆盖
    print("📋 步骤2: 命令行参数覆盖")
    if args.deepspeed_config:
        print(f"  🔧 使用命令行指定的DeepSpeed配置: {args.deepspeed_config}")
        config['deepspeed'] = args.deepspeed_config
    else:
        print(f"  🔧 使用YAML中的DeepSpeed配置: {config.get('deepspeed', 'NOT_FOUND')}")
    print()
    
    # 步骤3: 验证DeepSpeed配置
    print("📋 步骤3: 验证DeepSpeed配置")
    if 'deepspeed' not in config:
        print(f"  ❌ DeepSpeed配置未找到！")
        return
    else:
        print(f"  ✅ DeepSpeed配置存在")
        deepspeed_config = config['deepspeed']
        print(f"  • 配置类型: {type(deepspeed_config)}")
        print(f"  • 配置内容: {deepspeed_config}")
        
        if isinstance(deepspeed_config, str):
            if os.path.exists(deepspeed_config):
                print(f"  ✅ 配置文件存在: {deepspeed_config}")
                with open(deepspeed_config, 'r') as f:
                    ds_config = json.load(f)
                print(f"  • train_batch_size: {ds_config.get('train_batch_size', 'NOT_FOUND')}")
                print(f"  • train_micro_batch_size_per_gpu: {ds_config.get('train_micro_batch_size_per_gpu', 'NOT_FOUND')}")
            else:
                print(f"  ❌ 配置文件不存在: {deepspeed_config}")
    print()
    
    # 步骤4: 调用prepare_config
    print("📋 步骤4: 调用prepare_config")
    try:
        config = prepare_config(config)
        print(f"  ✅ prepare_config成功")
    except Exception as e:
        print(f"  ❌ prepare_config失败: {e}")
        return
    print()
    
    # 步骤5: 最终验证
    print("📋 步骤5: 最终验证")
    print(f"  • DeepSpeed配置: {config.get('deepspeed', 'NOT_FOUND')}")
    print(f"  • 输出目录: {config.get('output_dir', 'NOT_FOUND')}")
    print(f"  • 训练轮数: {config.get('training', {}).get('num_epochs', 'NOT_FOUND')}")
    print(f"  • 学习率: {config.get('training', {}).get('learning_rate', 'NOT_FOUND')}")
    
    print()
    print("✅ 配置加载流程测试完成!")

if __name__ == "__main__":
    test_config_loading() 