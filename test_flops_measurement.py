#!/usr/bin/env python3
"""
测试FLOPs测量功能
验证profiler和估算方法是否正常工作
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.utils.monitor import (
    profile_model_flops, 
    _measure_flops_with_profiler,
    _estimate_flops_fallback,
    _create_dummy_batch_for_profiling
)

def test_flops_measurement():
    """测试FLOPs测量功能"""
    
    print("🧪 测试FLOPs测量功能...")
    print("=" * 50)
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return
    
    device = torch.device('cuda:0')
    
    # 创建一个简单的测试模型
    class SimpleTestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(512, 256)
            self.linear2 = torch.nn.Linear(256, 101)
            self.relu = torch.nn.ReLU()
            
        def forward(self, input_ids, attention_mask, pixel_values, labels):
            # 模拟多模态输入处理
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)
            
            # 处理文本输入
            text_features = torch.randn(batch_size, seq_len, 512, device=input_ids.device)
            text_output = self.linear1(text_features)
            text_output = self.relu(text_output)
            
            # 处理图像输入
            image_features = torch.randn(batch_size, 3, 224, 224, device=pixel_values.device)
            image_output = torch.mean(image_features.view(batch_size, -1), dim=1, keepdim=True)
            image_output = image_output.expand(-1, seq_len, -1)
            
            # 融合特征
            combined = text_output + image_output
            logits = self.linear2(combined)
            
            # 计算损失
            loss = torch.nn.functional.cross_entropy(logits.view(-1, 101), labels.view(-1))
            
            return type('Outputs', (), {'loss': loss, 'logits': logits})()
    
    # 创建模型
    model = SimpleTestModel().to(device)
    print(f"✅ 创建测试模型: {sum(p.numel() for p in model.parameters()):,} 参数")
    
    # 创建测试batch
    batch_size = 8
    seq_length = 512
    
    test_batch = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_length), device=device),
        'attention_mask': torch.ones(batch_size, seq_length, device=device),
        'pixel_values': torch.randn(batch_size, 3, 224, 224, device=device),
        'labels': torch.randint(0, 101, (batch_size,), device=device)
    }
    
    print(f"✅ 创建测试batch: batch_size={batch_size}, seq_length={seq_length}")
    
    # 测试1: 虚拟batch创建
    print("\n📊 测试1: 虚拟batch创建")
    try:
        dummy_batch = _create_dummy_batch_for_profiling(batch_size, seq_length, device)
        if dummy_batch:
            print(f"✅ 虚拟batch创建成功: {list(dummy_batch.keys())}")
        else:
            print("❌ 虚拟batch创建失败")
    except Exception as e:
        print(f"❌ 虚拟batch创建错误: {e}")
    
    # 测试2: Profiler FLOPs测量
    print("\n📊 测试2: Profiler FLOPs测量")
    try:
        flops = _measure_flops_with_profiler(model, batch_size, seq_length)
        print(f"✅ Profiler FLOPs测量: {flops:.2e}")
    except Exception as e:
        print(f"❌ Profiler FLOPs测量错误: {e}")
    
    # 测试3: 估算FLOPs
    print("\n📊 测试3: 估算FLOPs")
    try:
        estimated_flops = _estimate_flops_fallback(model, test_batch, seq_length)
        print(f"✅ 估算FLOPs: {estimated_flops:.2e}")
    except Exception as e:
        print(f"❌ 估算FLOPs错误: {e}")
    
    # 测试4: 完整FLOPs测量
    print("\n📊 测试4: 完整FLOPs测量")
    try:
        total_flops = profile_model_flops(model, test_batch)
        print(f"✅ 完整FLOPs测量: {total_flops:.2e}")
    except Exception as e:
        print(f"❌ 完整FLOPs测量错误: {e}")
    
    print("\n✅ FLOPs测量测试完成")

if __name__ == "__main__":
    test_flops_measurement() 