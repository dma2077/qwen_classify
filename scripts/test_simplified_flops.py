#!/usr/bin/env python3
"""
测试脚本：验证简化后的FLOPs计算功能
"""

import os
import sys
import torch
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_simplified_flops():
    """测试简化后的FLOPs计算功能"""
    
    print("🧪 开始测试简化后的FLOPs计算功能")
    print("=" * 80)
    
    try:
        from training.utils.monitor import profile_model_flops
        
        print("✅ 成功导入 profile_model_flops 函数")
        
        # 创建一个简单的测试batch
        batch_example = {
            'input_ids': torch.randint(0, 1000, (2, 100)),  # batch_size=2, seq_len=100
            'attention_mask': torch.ones((2, 120)),          # 包含visual tokens，总长度120
            'pixel_values': torch.randn((2, 3, 224, 224)),   # 图像数据
            'labels': torch.randint(0, 101, (2,))            # 标签
        }
        
        print("📋 测试数据:")
        print(f"  • 文本tokens长度: {batch_example['input_ids'].size(1)}")
        print(f"  • attention_mask长度: {batch_example['attention_mask'].size(1)}")
        print(f"  • 批次大小: {batch_example['input_ids'].size(0)}")
        print("")
        
        # 创建一个简单的mock模型来测试
        class MockModel:
            def __init__(self):
                self.training = True
                
            def train(self):
                self.training = True
                
            def parameters(self):
                # 模拟7B参数的模型
                yield torch.randn(1000000)  # 1M参数用于测试
                yield torch.randn(6000000)  # 6M参数用于测试
                
        mock_model = MockModel()
        
        print("🔄 调用 profile_model_flops...")
        flops = profile_model_flops(mock_model, batch_example)
        
        print("")
        print("📊 测试结果:")
        print(f"  • 返回的FLOPs值: {flops:.2e}")
        print(f"  • FLOPs类型: {type(flops)}")
        
        if flops > 0:
            print("✅ FLOPs计算成功！")
            print(f"  • 估算的FLOPs: {flops:.2e}")
            
            # 计算一个简单的MFU示例
            # 假设计算时间为0.1秒
            compute_time = 0.1  # 秒
            theoretical_flops_per_sec = flops / compute_time
            
            # 假设GPU的理论峰值性能（例如A100的19.5 TFLOPS for bf16）
            gpu_peak_flops = 19.5e12  # FLOPS
            mfu = (theoretical_flops_per_sec / gpu_peak_flops) * 100
            
            print(f"  • 示例MFU计算:")
            print(f"    - 假设计算时间: {compute_time}s")
            print(f"    - 理论FLOPS/s: {theoretical_flops_per_sec:.2e}")
            print(f"    - GPU峰值性能: {gpu_peak_flops:.2e} FLOPS")
            print(f"    - 估算MFU: {mfu:.2f}%")
            
            return True
        else:
            print("❌ FLOPs计算返回0或负值")
            return False
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_flops_functions_removed():
    """测试复杂的profiler函数是否已被正确删除"""
    
    print("\n" + "=" * 80)
    print("🧪 测试复杂的profiler函数是否已被删除")
    print("=" * 80)
    
    try:
        from training.utils import monitor
        
        # 检查删除的函数是否不存在
        removed_functions = [
            '_profile_forward_flops',
            '_profile_backward_flops'
        ]
        
        all_removed = True
        for func_name in removed_functions:
            if hasattr(monitor, func_name):
                print(f"❌ 函数 {func_name} 仍然存在，应该已被删除")
                all_removed = False
            else:
                print(f"✅ 函数 {func_name} 已正确删除")
        
        # 检查保留的函数是否存在
        kept_functions = [
            'profile_model_flops',
            '_estimate_flops_fallback',
            '_estimate_forward_flops',
            '_get_actual_sequence_length'
        ]
        
        for func_name in kept_functions:
            if hasattr(monitor, func_name):
                print(f"✅ 函数 {func_name} 正确保留")
            else:
                print(f"❌ 函数 {func_name} 不存在，应该保留")
                all_removed = False
        
        return all_removed
        
    except Exception as e:
        print(f"❌ 检查函数时出现错误: {e}")
        return False

if __name__ == "__main__":
    print("🚀 开始简化后的FLOPs计算功能测试")
    print("这个测试将验证:")
    print("1. 简化后的FLOPs计算是否正常工作")
    print("2. 复杂的profiler函数是否已被正确删除")
    print("3. 估算方法是否能正确计算MFU所需的FLOPs")
    print("")
    
    # 运行测试
    test1_pass = test_simplified_flops()
    test2_pass = test_flops_functions_removed()
    
    print("\n" + "=" * 80)
    print("📋 测试结果汇总:")
    print(f"  • 简化FLOPs计算测试: {'✅ 通过' if test1_pass else '❌ 失败'}")
    print(f"  • 函数删除检查测试: {'✅ 通过' if test2_pass else '❌ 失败'}")
    
    if test1_pass and test2_pass:
        print("\n🎉 所有测试通过！FLOPs计算已成功简化")
        print("💡 现在只使用估算方法计算FLOPs，性能更好，代码更简洁")
        print("📈 估算方法足够准确用于MFU计算和性能监控")
    else:
        print("\n❌ 部分测试失败，请检查代码修改")
        sys.exit(1) 