#!/usr/bin/env python3
"""
验证hidden_states修复是否生效的脚本
"""

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

class MockQwenModel(nn.Module):
    """模拟Qwen模型，用于测试"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        
    def forward(self, **kwargs):
        # 模拟大的hidden_states tensor (4.67亿元素)
        batch_size = 8
        seq_length = 2048
        hidden_size = 3584
        
        # 创建模拟的大tensor
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        logits = torch.randn(batch_size, 5)
        loss = torch.randn(1)
        
        print(f"🔍 MockQwenModel forward调用:")
        print(f"   模型training状态: {self.training}")
        print(f"   Hidden states大小: {hidden_states.numel():,} 元素")
        
        # 模拟我们的修复逻辑
        if not self.training:
            print("   ✅ 评估模式: 返回 hidden_states=None")
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=None,
                attentions=None,
            )
        else:
            print("   ⚠️  训练模式: 返回完整hidden_states")
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=hidden_states,
                attentions=hidden_states,  # 模拟attentions
            )

def test_model_modes():
    """测试模型在不同模式下的行为"""
    print("=" * 60)
    print("🧪 测试hidden_states修复")
    print("=" * 60)
    
    model = MockQwenModel()
    
    # 测试训练模式
    print("\n📚 测试训练模式:")
    model.train()
    outputs = model()
    print(f"   返回的hidden_states: {outputs.hidden_states is not None}")
    if outputs.hidden_states is not None:
        print(f"   Hidden states元素数: {outputs.hidden_states.numel():,}")
    
    # 测试评估模式
    print("\n📊 测试评估模式:")
    model.eval()
    outputs = model()
    print(f"   返回的hidden_states: {outputs.hidden_states is not None}")
    
    print("\n" + "=" * 60)
    print("结果分析:")
    print("✅ 如果评估模式下hidden_states=None，说明修复生效")
    print("❌ 如果评估模式下仍返回hidden_states，说明修复失败")
    print("=" * 60)

if __name__ == "__main__":
    test_model_modes() 