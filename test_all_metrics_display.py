#!/usr/bin/env python3
"""
全面测试所有指标在WandB中的显示
验证training、eval、perf等所有指标组是否正常显示
"""

import os
import sys
import time
import torch
import yaml
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from training.utils.monitor import TrainingMonitor
from training.utils.config_utils import prepare_config

def test_all_metrics_display():
    """测试所有指标在WandB中的显示"""
    
    print("🧪 开始全面测试所有指标显示...")
    
    # 创建测试配置
    test_config = {
        'model': {
            'pretrained_name': 'Qwen/Qwen2.5-VL-7B-Instruct',
            'num_labels': 10
        },
        'training': {
            'epochs': 1,
            'lr': 1e-5,
            'output_dir': './test_outputs/all_metrics_test',
            'logging_steps': 1,
            'eval_steps': 5,
            'save_steps': 10
        },
        'monitor': {
            'freq': {
                'training_log_freq': 1,
                'perf_log_freq': 2,
                'gpu_log_freq': 3,
                'flops_profile_freq': 4,
                'local_save_freq': 5
            }
        },
        'wandb': {
            'enabled': True,
            'project': 'qwen-classify-test',
            'run_name': 'all_metrics_display_test',
            'tags': ['test', 'metrics', 'display'],
            'notes': '全面测试所有指标显示功能'
        }
    }
    
    # 准备配置
    config = prepare_config(test_config)
    
    # 创建输出目录
    output_dir = config['training']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化监控器
    monitor = TrainingMonitor(
        output_dir=output_dir,
        config=config,
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
        
        # 模拟实时FLOPs测量
        real_time_flops = 1e12 + (step * 1e10)  # 模拟FLOPs变化
        
        # 记录训练步骤
        is_eval_step = (step % 5 == 0)
        monitor.log_step(
            step=step,
            epoch=epoch,
            loss=loss,
            grad_norm=grad_norm,
            learning_rate=lr,
            attention_mask=attention_mask,
            real_time_flops=real_time_flops,
            skip_wandb=is_eval_step  # eval步骤跳过wandb记录
        )
        
        # 如果是评估步骤，记录评估指标
        if is_eval_step:
            eval_loss = loss * 0.8  # 评估损失通常比训练损失低
            eval_accuracy = 0.5 + (step * 0.02)  # 模拟准确率提升
            
            # 记录评估指标
            eval_data = {
                "eval/overall_loss": eval_loss,
                "eval/overall_accuracy": eval_accuracy,
                "eval/step": step
            }
            
            # 合并训练和评估指标
            training_data = {
                "training/loss": loss,
                "training/lr": lr,
                "training/epoch": epoch,
                "training/grad_norm": grad_norm,
            }
            
            # 添加性能指标（如果应该记录）
            if step % monitor.freq['perf_log_freq'] == 0:
                training_data.update({
                    "perf/step_time": 0.1 + (step * 0.01),
                    "perf/steps_per_second": 10.0 - (step * 0.1),
                    "perf/mfu": 0.3 + (step * 0.01),
                    "perf/mfu_percent": (0.3 + (step * 0.01)) * 100,
                    "perf/tokens_per_second": 1000 + (step * 50),
                    "perf/samples_per_second": 20 + (step * 1),
                    "perf/actual_flops": real_time_flops,
                    "perf/actual_seq_length": 512,
                    "perf/flops_per_second": real_time_flops / 0.1,
                })
            
            # 添加GPU指标（如果应该记录）
            if step % monitor.freq['gpu_log_freq'] == 0:
                training_data.update({
                    "perf/gpu_memory_allocated_gb": 8.0 + (step * 0.1),
                    "perf/gpu_memory_reserved_gb": 10.0 + (step * 0.1),
                    "perf/gpu_memory_utilization_percent": 60.0 + (step * 1.0),
                })
            
            # 合并所有指标
            combined_data = {**training_data, **eval_data}
            combined_data["step"] = step
            
            # 记录到WandB
            monitor.log_metrics(combined_data, step, commit=True)
            
            print(f"✅ 步骤 {step}: 已记录 {len(combined_data)} 个指标")
            print(f"   训练指标: {list(training_data.keys())}")
            print(f"   评估指标: {list(eval_data.keys())}")
        
        time.sleep(0.1)  # 短暂延迟，避免WandB API限制
    
    # 记录epoch统计
    monitor.log_epoch(epoch=1, avg_loss=1.5, elapsed_time=10.0, current_step=20)
    
    # 记录最终评估
    final_eval_data = {
        "eval/final_overall_loss": 1.2,
        "eval/final_overall_accuracy": 0.85,
        "eval/final_evaluation": 1.0
    }
    monitor.log_metrics(final_eval_data, 20, commit=True)
    
    # 结束训练
    monitor.finish_training()
    
    print("\n🎉 测试完成！")
    print("📊 请检查WandB界面，确认以下指标组是否正常显示：")
    print("   • training/* - 训练指标")
    print("   • eval/* - 评估指标") 
    print("   • perf/* - 性能指标")
    print("   • 所有指标都应该有统一的'step'作为x轴")
    
    return True

if __name__ == "__main__":
    try:
        test_all_metrics_display()
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 