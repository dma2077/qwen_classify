#!/usr/bin/env python3
"""
简单测试FLOPs测量修复
验证所有profiler错误是否已解决
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_flops_simple():
    """简单测试FLOPs测量"""
    
    print("🧪 简单测试FLOPs测量修复...")
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
            
            # 处理图像输入 - 修复维度问题
            image_features = torch.randn(batch_size, 3, 224, 224, device=pixel_values.device)
            # 将图像特征展平并投影到正确的维度
            image_flat = image_features.view(batch_size, -1)  # [batch_size, 3*224*224]
            image_projected = torch.nn.functional.linear(image_flat, torch.randn(256, image_flat.size(1), device=image_flat.device))  # [batch_size, 256]
            image_output = image_projected.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, 256]
            
            # 融合特征
            combined = text_output + image_output
            logits = self.linear2(combined)
            
            # 计算损失 - 修复batch size不匹配问题
            # logits形状: [batch_size, seq_len, 101] -> [batch_size * seq_len, 101]
            # labels形状: [batch_size] -> [batch_size * seq_len]
            logits_flat = logits.view(-1, 101)  # [batch_size * seq_len, 101]
            labels_flat = labels.unsqueeze(1).expand(-1, seq_len).contiguous().view(-1)  # [batch_size * seq_len]
            loss = torch.nn.functional.cross_entropy(logits_flat, labels_flat)
            
            # 返回一个类似transformers输出的对象（不包含last_hidden_state）
            class Outputs:
                def __init__(self, loss, logits):
                    self.loss = loss
                    self.logits = logits
                    # 不包含last_hidden_state，模拟你的模型输出
            
            return Outputs(loss, logits)
    
    # 创建模型
    model = SimpleTestModel().to(device)
    print(f"✅ 创建测试模型: {sum(p.numel() for p in model.parameters()):,} 参数")
    
    # 创建测试batch（包含attention_mask）
    batch_size = 8
    seq_length = 512
    
    test_batch = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_length), device=device),
        'attention_mask': torch.ones(batch_size, seq_length, device=device),  # 全1的attention_mask
        'pixel_values': torch.randn(batch_size, 3, 224, 224, device=device),
        'labels': torch.randint(0, 101, (batch_size,), device=device)
    }
    
    print(f"✅ 创建测试batch: batch_size={batch_size}, seq_length={seq_length}")
    
    # 测试序列长度获取
    print("\n📊 测试序列长度获取...")
    try:
        from training.utils.monitor import _get_actual_sequence_length
        seq_len = _get_actual_sequence_length(model, test_batch)
        print(f"✅ 序列长度获取成功: {seq_len}")
    except Exception as e:
        print(f"❌ 序列长度获取失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试估算方法
    print("\n📊 测试估算方法...")
    try:
        from training.utils.monitor import _estimate_flops_fallback
        estimated_flops = _estimate_flops_fallback(model, test_batch, seq_length)
        print(f"✅ 估算方法成功: {estimated_flops:.2e}")
    except Exception as e:
        print(f"❌ 估算方法失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试完整FLOPs测量
    print("\n📊 测试完整FLOPs测量...")
    try:
        from training.utils.monitor import profile_model_flops
        total_flops = profile_model_flops(model, test_batch)
        print(f"✅ 完整FLOPs测量成功: {total_flops:.2e}")
    except Exception as e:
        print(f"❌ 完整FLOPs测量失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ FLOPs测量修复测试完成")

if __name__ == "__main__":
    test_flops_simple() 