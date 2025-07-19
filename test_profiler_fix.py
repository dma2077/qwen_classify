#!/usr/bin/env python3
"""
测试PyTorch Profiler修复
验证NoneType错误是否已解决
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_profiler_fix():
    """测试profiler修复"""
    
    print("🧪 测试PyTorch Profiler修复...")
    print("=" * 50)
    
    # 检查PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    
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
    
    # 测试1: 基础profiler功能
    print("\n📊 测试1: 基础profiler功能")
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=False
        ) as prof:
            with torch.no_grad():
                _ = model(**test_batch)
        
        events = prof.events()
        if events is not None:
            print(f"✅ 基础profiler正常工作，获取到 {len(events)} 个事件")
            
            # 检查是否有FLOPs信息
            flops_events = [e for e in events if hasattr(e, 'flops') and e.flops > 0]
            print(f"  包含FLOPs信息的事件: {len(flops_events)}")
            
            if flops_events:
                total_flops = sum(e.flops for e in flops_events)
                print(f"  总FLOPs: {total_flops:.2e}")
            else:
                print("  未检测到FLOPs信息")
        else:
            print("❌ Profiler events为None")
            
    except Exception as e:
        print(f"❌ 基础profiler测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试2: 序列长度获取
    print("\n📊 测试2: 序列长度获取")
    try:
        from training.utils.monitor import _get_actual_sequence_length
        seq_len = _get_actual_sequence_length(model, test_batch)
        print(f"✅ 序列长度获取成功: {seq_len}")
    except Exception as e:
        print(f"❌ 序列长度获取失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试3: 前向传播FLOPs测量
    print("\n📊 测试3: 前向传播FLOPs测量")
    try:
        from training.utils.monitor import _profile_forward_flops
        forward_flops = _profile_forward_flops(model, test_batch)
        print(f"✅ 前向传播FLOPs测量: {forward_flops:.2e}")
    except Exception as e:
        print(f"❌ 前向传播FLOPs测量失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试4: 反向传播FLOPs测量
    print("\n📊 测试4: 反向传播FLOPs测量")
    try:
        from training.utils.monitor import _profile_backward_flops
        backward_flops = _profile_backward_flops(model, test_batch)
        print(f"✅ 反向传播FLOPs测量: {backward_flops:.2e}")
    except Exception as e:
        print(f"❌ 反向传播FLOPs测量失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试5: 完整FLOPs测量
    print("\n📊 测试5: 完整FLOPs测量")
    try:
        from training.utils.monitor import profile_model_flops
        total_flops = profile_model_flops(model, test_batch)
        print(f"✅ 完整FLOPs测量: {total_flops:.2e}")
    except Exception as e:
        print(f"❌ 完整FLOPs测量失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试6: 估算方法
    print("\n📊 测试6: 估算方法")
    try:
        from training.utils.monitor import _estimate_flops_fallback
        estimated_flops = _estimate_flops_fallback(model, test_batch, seq_length)
        print(f"✅ 估算方法: {estimated_flops:.2e}")
    except Exception as e:
        print(f"❌ 估算方法失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Profiler修复测试完成")

if __name__ == "__main__":
    test_profiler_fix() 