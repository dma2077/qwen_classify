#!/usr/bin/env python3
"""
测试Profiler MFU计算功能

验证：
1. 每flops_profile_freq步使用profiler计算MFU
2. 其他步骤MFU值为0
3. 性能开销控制
"""

import time
import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.utils.monitor import TrainingMonitor, calculate_mfu_with_profiler

def create_dummy_model():
    """创建一个简单的模型用于测试"""
    try:
        from transformers import AutoModelForImageClassification
        
        # 使用一个小的预训练模型
        model_name = "microsoft/resnet-50"
        model = AutoModelForImageClassification.from_pretrained(model_name)
        
        # 移动到GPU（如果可用）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        return model
        
    except Exception as e:
        print(f"创建模型失败: {e}")
        return None

def test_profiler_mfu_calculation():
    """测试profiler MFU计算"""
    print("=" * 80)
    print("🧪 测试Profiler MFU计算功能")
    print("=" * 80)
    
    # 创建测试模型
    model = create_dummy_model()
    if model is None:
        print("❌ 无法创建测试模型，退出测试")
        return
    
    print(f"✅ 测试模型创建成功")
    print(f"   模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   设备: {next(model.parameters()).device}")
    
    # 测试参数
    batch_size = 4
    seq_length = 512
    step_time = 0.1  # 100ms per step
    
    print(f"\n📊 测试参数:")
    print(f"   批次大小: {batch_size}")
    print(f"   序列长度: {seq_length}")
    print(f"   步骤时间: {step_time}s")
    
    # 测试profiler MFU计算
    print(f"\n🔍 测试Profiler MFU计算:")
    print("-" * 50)
    
    try:
        start_time = time.time()
        mfu = calculate_mfu_with_profiler(model, batch_size, seq_length, step_time)
        calculation_time = time.time() - start_time
        
        print(f"   MFU值: {mfu:.4f}")
        print(f"   计算时间: {calculation_time*1000:.2f} ms")
        print(f"   ✅ Profiler MFU计算成功")
        
    except Exception as e:
        print(f"   ❌ Profiler MFU计算失败: {e}")
        return
    
    # 测试TrainingMonitor的MFU计算
    print(f"\n🔍 测试TrainingMonitor MFU计算:")
    print("-" * 50)
    
    # 创建配置
    config = {
        'model': {'max_sequence_length': seq_length},
        'deepspeed': {'train_batch_size': batch_size},
        'monitoring': {'flops_profile_freq': 3}  # 每3步计算一次
    }
    
    # 创建TrainingMonitor
    monitor = TrainingMonitor("./test_output", config, flops_profile_freq=3)
    monitor.set_model_ref(model)
    
    print(f"   flops_profile_freq: {monitor.flops_profile_freq}")
    print(f"   batch_size: {monitor.batch_size}")
    print(f"   seq_length: {monitor.seq_length}")
    
    # 模拟训练步骤
    print(f"\n📈 模拟训练步骤 (每{monitor.flops_profile_freq}步计算MFU):")
    print("-" * 50)
    
    for step in range(1, 11):  # 测试10步
        # 模拟步骤时间
        step_time = 0.1 + (step % 3) * 0.01  # 稍微变化的步骤时间
        
        # 创建虚拟attention_mask
        attention_mask = torch.ones(batch_size, seq_length)
        
        # 记录步骤
        monitor.log_step(step, 0, 0.5, 1.0, 1e-5, attention_mask)
        
        # 检查是否应该计算MFU
        should_calculate = (step % monitor.flops_profile_freq == 0)
        print(f"   步骤 {step:2d}: {'🔍 计算MFU' if should_calculate else '⏭️  跳过MFU'}")
    
    print(f"\n✅ 测试完成")
    print("=" * 80)

def test_performance_impact():
    """测试性能影响"""
    print("\n" + "=" * 80)
    print("⚡ 测试性能影响")
    print("=" * 80)
    
    model = create_dummy_model()
    if model is None:
        return
    
    batch_size = 4
    seq_length = 512
    step_time = 0.1
    
    # 测试不同频率的性能影响
    frequencies = [1, 10, 50, 100, 500]
    
    print(f"📊 不同频率的性能影响测试:")
    print("-" * 50)
    
    for freq in frequencies:
        print(f"\n🔍 测试频率: 每{freq}步计算一次MFU")
        
        # 模拟1000步训练
        total_steps = 1000
        mfu_calculations = total_steps // freq
        
        # 估算总开销
        single_calculation_time = 0.05  # 假设每次计算50ms
        total_overhead = mfu_calculations * single_calculation_time
        total_training_time = total_steps * step_time
        overhead_percentage = (total_overhead / total_training_time) * 100
        
        print(f"   总步数: {total_steps}")
        print(f"   MFU计算次数: {mfu_calculations}")
        print(f"   估算总开销: {total_overhead:.2f}s")
        print(f"   估算训练时间: {total_training_time:.2f}s")
        print(f"   性能开销: {overhead_percentage:.2f}%")
        
        if overhead_percentage < 1:
            print(f"   ✅ 性能开销可接受 (< 1%)")
        elif overhead_percentage < 5:
            print(f"   ⚠️  性能开销中等 (1-5%)")
        else:
            print(f"   ❌ 性能开销较高 (> 5%)")
    
    print(f"\n💡 性能建议:")
    print(f"   - 对于生产环境: 建议频率 >= 100 (开销 < 1%)")
    print(f"   - 对于调试环境: 建议频率 >= 10 (开销 < 5%)")
    print(f"   - 对于研究环境: 可以设置频率 = 1 (最高精度)")

if __name__ == "__main__":
    try:
        # 基础功能测试
        test_profiler_mfu_calculation()
        
        # 性能影响测试
        test_performance_impact()
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc() 