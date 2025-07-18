#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试简化后的模型代码
"""

import torch
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.qwen2_5_vl_classify import Qwen2_5_VLForImageClassification

def test_model_initialization():
    """测试模型初始化"""
    print("🧪 测试模型初始化...")
    
    try:
        # 创建模型实例
        model = Qwen2_5_VLForImageClassification(
            pretrained_model_name="/llm_reco/dehua/model/Qwen2.5-VL-7B-Instruct",
            num_labels=101,
            loss_config={'type': 'label_smoothing', 'smoothing': 0.1},
            enable_logits_masking=True
        )
        
        print("✅ 模型初始化成功")
        
        # 检查模型配置
        if hasattr(model.model, 'config') and hasattr(model.model.config, '_attn_implementation'):
            attn_impl = model.model.config._attn_implementation
            print(f"📋 Attention实现: {attn_impl}")
        else:
            print("📋 无法检测attention实现")
        
        # 检查数据类型
        print(f"📋 模型数据类型: {next(model.parameters()).dtype}")
        
        return model
        
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return None

def test_forward_pass(model):
    """测试前向传播"""
    print("\n🧪 测试前向传播...")
    
    try:
        # 创建测试数据
        batch_size = 2
        seq_len = 512
        hidden_size = 4096  # Qwen2.5-VL的hidden_size
        
        # 模拟输入数据
        pixel_values = torch.randn(batch_size, 3, 224, 224, dtype=torch.bfloat16)
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, 101, (batch_size,))
        
        # 前向传播
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        print("✅ 前向传播成功")
        print(f"📋 Loss: {outputs.loss}")
        print(f"📋 Logits shape: {outputs.logits.shape}")
        print(f"📋 Logits dtype: {outputs.logits.dtype}")
        
        return True
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始测试简化后的模型代码")
    print("=" * 50)
    
    # 测试模型初始化
    model = test_model_initialization()
    
    if model is not None:
        # 测试前向传播
        test_forward_pass(model)
    
    print("\n" + "=" * 50)
    print("✅ 测试完成")

if __name__ == "__main__":
    main() 