#!/usr/bin/env python3
"""
MFU计算测试 - 分析当前MFU计算方式的问题
"""

import os
import time
import json
import torch
import wandb
from typing import Dict

# 模拟配置
config = {
    'output_dir': './test_output',
    'wandb': {
        'enabled': True,
        'project': 'qwen_classify_test',
        'run_name': 'mfu_calculation_test',
        'tags': ['test', 'mfu']
    },
    'monitor': {
        'freq': {
            'all_freq': 1
        }
    },
    'model': {
        'max_sequence_length': 512
    },
    'deepspeed': {
        'train_batch_size': 32
    }
}

def get_gpu_peak_flops():
    """获取GPU峰值FLOPs性能"""
    try:
        if not torch.cuda.is_available():
            return 312e12  # 默认值
        
        # 获取GPU名称
        gpu_name = torch.cuda.get_device_name(0).upper()
        print(f"🔍 检测到GPU: {gpu_name}")
        
        # 不同GPU的峰值性能 (TFLOPs for FP16/BF16)
        gpu_peak_flops = {
            'A100': 312e12,    # A100 80GB
            'A100-SXM': 312e12,
            'A100-PCIE': 312e12,
            'A800': 280e12,    # A800 80GB
            'H100': 989e12,    # H100 80GB
            'H100-SXM': 989e12,
            'H100-PCIE': 756e12,
            'H800': 850e12,    # H800 80GB
            'V100': 112e12,    # V100 32GB
            'RTX 4090': 165e12,
            'RTX 4080': 112e12,
            'RTX 3090': 71e12,
            'RTX 3080': 58e12,
            'T4': 65e12,
            'L4': 121e12,
        }
        
        # 查找匹配的GPU
        for gpu_model, peak_flops in gpu_peak_flops.items():
            if gpu_model in gpu_name:
                print(f"✅ 识别GPU: {gpu_name} -> {gpu_model} ({peak_flops/1e12:.0f} TFLOPs)")
                return peak_flops
        
        # 如果没有找到匹配的GPU，使用默认值
        print(f"⚠️  未识别的GPU类型: {gpu_name}，使用默认峰值性能 (A100: 312 TFLOPs)")
        return 312e12
        
    except Exception as e:
        print(f"获取GPU峰值性能错误: {e}")
        return 312e12

def calculate_mfu_simple(actual_flops: float, step_time: float) -> float:
    """简单的MFU计算"""
    try:
        if actual_flops is None or actual_flops <= 0:
            return 0.0
        
        # 计算实际FLOPs/s
        actual_flops_per_second = actual_flops / step_time
        
        # 获取GPU峰值性能
        peak_flops_per_second = get_gpu_peak_flops()
        
        # 计算MFU
        mfu = actual_flops_per_second / peak_flops_per_second
        return min(mfu, 1.0)  # 限制在100%以内
        
    except Exception as e:
        print(f"MFU计算错误: {e}")
        return 0.0

def estimate_flops_for_qwen2_5_vl(batch_size: int, seq_length: int, num_classes: int = 101):
    """为Qwen2.5-VL模型估算FLOPs"""
    
    # Qwen2.5-VL-7B的参数数量（近似）
    model_params = 7.2e9  # 7.2B参数
    
    # 更准确的FLOPs估算（基于Transformer架构）
    # 对于7B模型，每个token的前向传播大约需要：
    # - 注意力机制: ~4 * hidden_size * seq_length
    # - 前馈网络: ~8 * hidden_size * seq_length  
    # - 其他操作: ~2 * hidden_size * seq_length
    # 总计: ~14 * hidden_size * seq_length per token
    
    hidden_size = 4096  # Qwen2.5-VL-7B的hidden size
    flops_per_token = 14 * hidden_size * seq_length
    
    # 前向传播FLOPs
    forward_flops = flops_per_token * batch_size
    
    # 反向传播FLOPs（通常是前向传播的2倍）
    backward_flops = 2 * forward_flops
    
    # 分类头FLOPs
    classification_flops = batch_size * hidden_size * num_classes
    
    total_flops = forward_flops + backward_flops + classification_flops
    
    return total_flops

def test_mfu_calculation():
    """测试MFU计算"""
    print("🚀 开始MFU计算测试")
    print("=" * 60)
    
    # 测试参数
    batch_sizes = [8, 16, 32, 64]
    seq_lengths = [256, 512, 1024, 2048]
    step_times = [0.1, 0.2, 0.5, 1.0]  # 秒
    
    print("📊 测试参数:")
    print(f"   • 批次大小: {batch_sizes}")
    print(f"   • 序列长度: {seq_lengths}")
    print(f"   • 步骤时间: {step_times}")
    print("=" * 60)
    
    # 获取GPU峰值性能
    peak_flops = get_gpu_peak_flops()
    print(f"📈 GPU峰值性能: {peak_flops/1e12:.0f} TFLOPs")
    print("=" * 60)
    
    results = []
    
    for batch_size in batch_sizes:
        for seq_length in seq_lengths:
            for step_time in step_times:
                # 估算FLOPs
                estimated_flops = estimate_flops_for_qwen2_5_vl(batch_size, seq_length)
                
                # 计算MFU
                mfu = calculate_mfu_simple(estimated_flops, step_time)
                
                # 计算实际FLOPs/s
                actual_flops_per_second = estimated_flops / step_time
                
                # 计算吞吐量
                tokens_per_second = batch_size * seq_length / step_time
                samples_per_second = batch_size / step_time
                
                result = {
                    'batch_size': batch_size,
                    'seq_length': seq_length,
                    'step_time': step_time,
                    'estimated_flops': estimated_flops,
                    'actual_flops_per_second': actual_flops_per_second,
                    'mfu': mfu,
                    'tokens_per_second': tokens_per_second,
                    'samples_per_second': samples_per_second
                }
                results.append(result)
                
                print(f"📊 Batch={batch_size}, Seq={seq_length}, Time={step_time}s:")
                print(f"   • 估算FLOPs: {estimated_flops:.2e}")
                print(f"   • FLOPs/s: {actual_flops_per_second:.2e}")
                print(f"   • MFU: {mfu:.4f} ({mfu*100:.2f}%)")
                print(f"   • Tokens/s: {tokens_per_second:.0f}")
                print(f"   • Samples/s: {samples_per_second:.1f}")
                print()
    
    # 分析结果
    print("=" * 60)
    print("📈 MFU分析结果:")
    
    # 找出最高MFU
    max_mfu_result = max(results, key=lambda x: x['mfu'])
    print(f"   • 最高MFU: {max_mfu_result['mfu']:.4f} ({max_mfu_result['mfu']*100:.2f}%)")
    print(f"     参数: Batch={max_mfu_result['batch_size']}, Seq={max_mfu_result['seq_length']}, Time={max_mfu_result['step_time']}s")
    
    # 找出最低MFU
    min_mfu_result = min(results, key=lambda x: x['mfu'])
    print(f"   • 最低MFU: {min_mfu_result['mfu']:.4f} ({min_mfu_result['mfu']*100:.2f}%)")
    print(f"     参数: Batch={min_mfu_result['batch_size']}, Seq={min_mfu_result['seq_length']}, Time={min_mfu_result['step_time']}s")
    
    # 平均MFU
    avg_mfu = sum(r['mfu'] for r in results) / len(results)
    print(f"   • 平均MFU: {avg_mfu:.4f} ({avg_mfu*100:.2f}%)")
    
    print("=" * 60)
    print("💡 MFU优化建议:")
    
    if avg_mfu < 0.1:
        print("   • MFU过低，可能的原因:")
        print("     - FLOPs估算不准确")
        print("     - GPU峰值性能设置过高")
        print("     - 实际FLOPs测量失败")
        print("     - 模型架构与估算不匹配")
    elif avg_mfu < 0.3:
        print("   • MFU偏低，建议:")
        print("     - 增加batch_size")
        print("     - 优化序列长度")
        print("     - 检查是否有性能瓶颈")
    else:
        print("   • MFU正常，性能良好")
    
    print("=" * 60)
    print("🔧 调试建议:")
    print("   1. 检查实际FLOPs测量是否成功")
    print("   2. 验证GPU峰值性能设置是否正确")
    print("   3. 对比不同batch_size和序列长度的MFU")
    print("   4. 检查是否有内存带宽限制")
    
    return results

if __name__ == "__main__":
    test_mfu_calculation() 