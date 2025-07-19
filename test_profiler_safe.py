#!/usr/bin/env python3
"""
安全测试PyTorch Profiler
专门处理NoneType错误和events()问题
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_profiler_safe():
    """安全测试profiler"""
    
    print("🧪 安全测试PyTorch Profiler...")
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
    
    # 测试1: 最基础的profiler测试
    print("\n📊 测试1: 最基础profiler测试")
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=False
        ) as prof:
            with torch.no_grad():
                _ = model(**test_batch)
        
        print("✅ Profiler上下文管理器正常工作")
        
        # 安全地获取events
        try:
            events = prof.events()
            if events is not None:
                print(f"✅ prof.events()成功，获取到 {len(events)} 个事件")
                
                # 检查事件类型
                event_types = set()
                for event in events:
                    if hasattr(event, 'name'):
                        event_types.add(type(event).__name__)
                
                print(f"  事件类型: {event_types}")
                
                # 检查是否有FLOPs属性
                flops_events = [e for e in events if hasattr(e, 'flops')]
                print(f"  包含flops属性的事件: {len(flops_events)}")
                
                if flops_events:
                    flops_with_values = [e for e in flops_events if e.flops > 0]
                    print(f"  有FLOPs值的事件: {len(flops_with_values)}")
                    
                    if flops_with_values:
                        total_flops = sum(e.flops for e in flops_with_values)
                        print(f"  总FLOPs: {total_flops:.2e}")
                    else:
                        print("  所有FLOPs事件的值都为0")
                else:
                    print("  没有找到包含flops属性的事件")
                    
            else:
                print("❌ prof.events()返回None")
                
        except Exception as events_error:
            print(f"❌ 获取prof.events()失败: {events_error}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ 基础profiler测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试2: 安全的events获取函数
    print("\n📊 测试2: 安全的events获取函数")
    def safe_get_events(prof):
        """安全地获取profiler events"""
        try:
            events = prof.events()
            if events is not None:
                return events, len(events)
            else:
                return None, 0
        except Exception as e:
            print(f"  获取events异常: {e}")
            return None, 0
    
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=False
        ) as prof:
            with torch.no_grad():
                _ = model(**test_batch)
        
        events, count = safe_get_events(prof)
        if events is not None:
            print(f"✅ 安全获取events成功: {count} 个事件")
        else:
            print("❌ 安全获取events失败")
            
    except Exception as e:
        print(f"❌ 安全events获取测试失败: {e}")
    
    # 测试3: 测试我们的修复函数
    print("\n📊 测试3: 测试修复后的函数")
    try:
        from training.utils.monitor import profile_model_flops
        total_flops = profile_model_flops(model, test_batch)
        print(f"✅ 修复后的FLOPs测量: {total_flops:.2e}")
    except Exception as e:
        print(f"❌ 修复后的FLOPs测量失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试4: 测试估算方法
    print("\n📊 测试4: 测试估算方法")
    try:
        from training.utils.monitor import _estimate_flops_fallback
        estimated_flops = _estimate_flops_fallback(model, test_batch, seq_length)
        print(f"✅ 估算方法: {estimated_flops:.2e}")
    except Exception as e:
        print(f"❌ 估算方法失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ 安全Profiler测试完成")

if __name__ == "__main__":
    test_profiler_safe() 