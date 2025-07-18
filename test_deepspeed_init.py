#!/usr/bin/env python3
"""
测试DeepSpeed初始化
"""

import os
import sys
import json
import torch
import deepspeed

def test_deepspeed_init():
    """测试DeepSpeed初始化"""
    print("🔍 测试DeepSpeed初始化")
    print("="*50)
    
    # 加载配置
    config_path = "configs/ds_s2.json"
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        deepspeed_config = json.load(f)
    
    print(f"📋 加载的配置:")
    print(f"  • train_batch_size: {deepspeed_config.get('train_batch_size', 'NOT_FOUND')}")
    print(f"  • train_micro_batch_size_per_gpu: {deepspeed_config.get('train_micro_batch_size_per_gpu', 'NOT_FOUND')}")
    print(f"  • gradient_accumulation_steps: {deepspeed_config.get('gradient_accumulation_steps', 'NOT_FOUND')}")
    
    # 创建一个简单的模型
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters())
    
    print(f"🔧 准备初始化DeepSpeed...")
    
    try:
        # 初始化DeepSpeed
        model, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=deepspeed_config
        )
        print(f"✅ DeepSpeed初始化成功!")
        
    except Exception as e:
        print(f"❌ DeepSpeed初始化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_deepspeed_init() 