#!/usr/bin/env python3
"""
测试MFU和FLOPs指标在WandB中的显示
验证性能指标是否能正确记录和显示
"""

import os
import sys
import time
import torch
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from training.utils.monitor import TrainingMonitor

def test_mfu_flops_display():
    """测试MFU和FLOPs指标在WandB中的显示"""
    
    print("🧪 开始测试MFU和FLOPs指标显示...")
    
    # 创建测试配置
    test_config = {
        'model': {
            'pretrained_name': 'Qwen/Qwen2.5-VL-7B-Instruct',
            'num_labels': 10
        },
        'training': {
            'epochs': 1,
            'lr': 1e-5,
            'output_dir': './test_outputs/mfu_flops_test',
            'logging_steps': 1,
            'eval_steps': 5,
            'save_steps': 10
        },
        'monitor': {
            'freq': {
                'training_log_freq': 1,
                'perf_log_freq': 2,  # 每2步记录性能指标
                'gpu_log_freq': 3,
                'flops_profile_freq': 4,  # 每4步使用profiler计算MFU
                'local_save_freq': 5
            }
        },
        'wandb': {
            'enabled': True,
            'project': 'qwen-classify-test',
            'run_name': 'mfu_flops_display_test',
            'tags': ['test', 'mfu', 'flops', 'performance'],
            'notes': '测试MFU和FLOPs指标显示功能'
        }
    }
    
    # 创建输出目录
    output_dir = test_config['training']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化监控器
    monitor = TrainingMonitor(
        output_dir=output_dir,
        config=test_config,
        flops_profile_freq=4
    )
    
    # 创建虚拟模型引用（用于MFU计算）
    class DummyModel:
        def __init__(self):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        def parameters(self):
            return [torch.randn(1000, 1000, device=self.device)]
        
        def eval(self):
            pass
        
        def __call__(self, **kwargs):
            return torch.randn(2, 10)  # 模拟输出
    
    dummy_model = DummyModel()
    monitor.set_model_ref(dummy_model)
    
    # 设置一些模拟的FLOPs数据
    monitor.set_actual_flops(1e12, 512)  # 1 TFLOPs
    
    print("✅ 监控器初始化完成")
    
    # 开始训练监控
    monitor.start_training()
    
    # 模拟训练步骤
    print("\n📊 模拟训练步骤...")
    for step in range(1, 21):  # 20步
        epoch = step // 10
        loss = 2.0 - (step * 0.05)  # 模拟损失下降
        grad_norm = 1.0 + (step * 0.01)  # 模拟梯度范数
        lr = 1e-5 * (0.9 ** (step // 5))  # 模拟学习率衰减
        
        # 创建虚拟attention_mask
        attention_mask = torch.ones(2, 512)  # batch_size=2, seq_len=512
        
        # 模拟实时FLOPs测量（变化的值）
        real_time_flops = 1e12 + (step * 1e10)  # 模拟FLOPs变化
        
        # 记录训练步骤
        monitor.log_step(
            step=step,
            epoch=epoch,
            loss=loss,
            grad_norm=grad_norm,
            learning_rate=lr,
            attention_mask=attention_mask,
            real_time_flops=real_time_flops,
            skip_wandb=False  # 正常记录到wandb
        )
        
        time.sleep(0.1)  # 短暂延迟，避免WandB API限制
    
    # 结束训练
    monitor.finish_training()
    
    print("\n🎉 测试完成！")
    print("📊 请检查WandB界面，确认以下性能指标是否正常显示：")
    print("1. 项目: qwen-classify-test")
    print("2. 运行: mfu_flops_display_test")
    print("3. 确认以下性能指标是否正常显示：")
    print("   • perf/mfu - Model FLOPs Utilization")
    print("   • perf/mfu_percent - MFU百分比")
    print("   • perf/actual_flops - 实际FLOPs")
    print("   • perf/flops_per_second - 每秒FLOPs")
    print("   • perf/tokens_per_second - 每秒token数")
    print("   • perf/samples_per_second - 每秒样本数")
    print("   • perf/step_time - 步骤时间")
    print("   • perf/steps_per_second - 每秒步数")
    print("   • perf/actual_seq_length - 实际序列长度")
    print("4. 确认MFU值不是0，而是有实际的计算值")
    print("5. 确认FLOPs相关指标都有合理的数值")
    
    return True

if __name__ == "__main__":
    try:
        test_mfu_flops_display()
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 