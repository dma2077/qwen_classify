#!/usr/bin/env python3
"""
PyTorch Profiler诊断脚本
专门诊断'NoneType' object is not iterable错误
"""

import torch
import sys
import os
import traceback

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_profiler_diagnosis():
    """诊断profiler问题"""
    
    print("🔍 PyTorch Profiler诊断...")
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
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(512, 256)
            self.linear2 = torch.nn.Linear(256, 101)
            self.relu = torch.nn.ReLU()
            
        def forward(self, input_ids, attention_mask, pixel_values, labels):
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)
            
            # 简单的文本处理
            text_features = torch.randn(batch_size, seq_len, 512, device=input_ids.device)
            text_output = self.linear1(text_features)
            text_output = self.relu(text_output)
            
            # 简单的图像处理
            image_features = torch.randn(batch_size, 256, device=pixel_values.device)
            image_output = image_features.unsqueeze(1).expand(-1, seq_len, -1)
            
            # 融合特征
            combined = text_output + image_output
            logits = self.linear2(combined)
            
            # 简化的损失计算
            first_token_logits = logits[:, 0, :]
            loss = torch.nn.functional.cross_entropy(first_token_logits, labels)
            
            class Outputs:
                def __init__(self, loss, logits):
                    self.loss = loss
                    self.logits = logits
            
            return Outputs(loss, logits)
    
    # 创建模型
    model = SimpleModel().to(device)
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
        
        print("✅ Profiler上下文管理器正常工作")
        
        # 详细检查events
        print("🔍 检查prof.events()...")
        try:
            events = prof.events()
            print(f"  events类型: {type(events)}")
            print(f"  events是否为None: {events is None}")
            
            if events is not None:
                try:
                    events_length = len(events)
                    print(f"  events长度: {events_length}")
                    
                    if events_length > 0:
                        print("  ✅ events不为空，尝试迭代...")
                        
                        # 尝试转换为list
                        try:
                            events_list = list(events)
                            print(f"  ✅ 成功转换为list，长度: {len(events_list)}")
                            
                            # 尝试迭代
                            event_count = 0
                            for event in events_list:
                                event_count += 1
                                if event_count <= 3:  # 只检查前3个事件
                                    print(f"    Event {event_count}: {type(event)}")
                                    if hasattr(event, 'name'):
                                        print(f"      name: {event.name}")
                                    if hasattr(event, 'flops'):
                                        print(f"      flops: {event.flops}")
                                if event_count >= 10:  # 只检查前10个事件
                                    break
                            
                            print(f"  ✅ 成功迭代 {event_count} 个事件")
                            
                        except Exception as list_error:
                            print(f"  ❌ 转换为list失败: {list_error}")
                            traceback.print_exc()
                    else:
                        print("  ⚠️  events为空")
                except Exception as len_error:
                    print(f"  ❌ 获取events长度失败: {len_error}")
                    traceback.print_exc()
            else:
                print("  ❌ events为None")
                
        except Exception as events_error:
            print(f"  ❌ 获取prof.events()失败: {events_error}")
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ 基础profiler测试失败: {e}")
        traceback.print_exc()
    
    # 测试2: 测试我们的修复函数
    print("\n📊 测试2: 测试修复后的函数")
    try:
        from training.utils.monitor import _profile_forward_flops
        forward_flops = _profile_forward_flops(model, test_batch)
        print(f"✅ 前向传播FLOPs测量: {forward_flops:.2e}")
    except Exception as e:
        print(f"❌ 前向传播FLOPs测量失败: {e}")
        traceback.print_exc()
    
    # 测试3: 测试反向传播
    print("\n📊 测试3: 测试反向传播")
    try:
        from training.utils.monitor import _profile_backward_flops
        backward_flops = _profile_backward_flops(model, test_batch)
        print(f"✅ 反向传播FLOPs测量: {backward_flops:.2e}")
    except Exception as e:
        print(f"❌ 反向传播FLOPs测量失败: {e}")
        traceback.print_exc()
    
    # 测试4: 测试完整FLOPs测量
    print("\n📊 测试4: 测试完整FLOPs测量")
    try:
        from training.utils.monitor import profile_model_flops
        total_flops = profile_model_flops(model, test_batch)
        print(f"✅ 完整FLOPs测量: {total_flops:.2e}")
    except Exception as e:
        print(f"❌ 完整FLOPs测量失败: {e}")
        traceback.print_exc()
    
    # 测试5: 测试估算方法
    print("\n📊 测试5: 测试估算方法")
    try:
        from training.utils.monitor import _estimate_flops_fallback
        estimated_flops = _estimate_flops_fallback(model, test_batch, seq_length)
        print(f"✅ 估算方法: {estimated_flops:.2e}")
    except Exception as e:
        print(f"❌ 估算方法失败: {e}")
        traceback.print_exc()
    
    print("\n✅ Profiler诊断完成")
    print("\n📋 诊断总结:")
    print("  1. 检查了prof.events()的类型和内容")
    print("  2. 测试了events的迭代和转换")
    print("  3. 验证了修复后的FLOPs测量函数")
    print("  4. 测试了备选的估算方法")

if __name__ == "__main__":
    test_profiler_diagnosis() 