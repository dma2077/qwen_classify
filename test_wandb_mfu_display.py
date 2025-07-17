#!/usr/bin/env python3
"""
测试WandB中MFU和FLOPs指标显示

验证：
1. MFU指标是否正确记录到WandB
2. FLOPs指标是否正确记录到WandB
3. 指标是否在正确的分组中显示
"""

import sys
import os
import time
import torch
import wandb

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.utils.monitor import TrainingMonitor

def test_wandb_mfu_display():
    """测试WandB中MFU和FLOPs指标显示"""
    print("=" * 80)
    print("🧪 测试WandB中MFU和FLOPs指标显示")
    print("=" * 80)
    
    # 配置
    config = {
        'monitor': {
            'freq': {
                'training_log_freq': 1,       # 每步记录训练指标
                'eval_log_freq': 1,           # 每步记录评估指标
                'perf_log_freq': 1,           # 每步记录性能指标
                'gpu_log_freq': 1,            # 每步记录GPU指标
                'flops_profile_freq': 2,      # 每2步计算MFU
                'local_save_freq': 10,        # 每10步保存本地日志
                'progress_update_freq': 1,    # 每步更新进度
            }
        },
        'deepspeed': {'train_batch_size': 64},
        'model': {'max_sequence_length': 512},
        'wandb': {
            'enabled': True,
            'project': 'qwen-classify-mfu-test',
            'name': 'mfu-display-test'
        }
    }
    
    # 创建监控器
    monitor = TrainingMonitor("./test_output", config, flops_profile_freq=2)
    
    # 模拟模型引用
    class MockModel:
        def __init__(self):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        def eval(self):
            pass
        
        def __call__(self, **kwargs):
            # 模拟前向传播
            return {'logits': torch.randn(4, 101).to(self.device)}
    
    mock_model = MockModel()
    monitor.set_model_ref(mock_model)
    
    # 模拟FLOPs测量
    monitor.set_actual_flops(1e12, 512)  # 1T FLOPs
    
    print("📊 开始模拟训练步骤...")
    
    # 模拟多个训练步骤
    for step in range(1, 11):
        print(f"\n🔍 步骤 {step}:")
        
        # 模拟attention_mask
        attention_mask = torch.ones(4, 512).to(mock_model.device)
        
        # 记录步骤
        monitor.log_step(
            step=step,
            epoch=1,
            loss=0.5 + step * 0.01,
            grad_norm=1.0 + step * 0.1,
            learning_rate=1e-5,
            attention_mask=attention_mask,
            skip_wandb=False
        )
        
        # 每5步进行一次评估
        if step % 5 == 0:
            monitor.log_evaluation(
                step=step,
                eval_loss=0.4 + step * 0.005,
                eval_accuracy=0.8 + step * 0.01
            )
        
        time.sleep(0.1)  # 模拟训练时间
    
    print(f"\n✅ 测试完成")
    print("=" * 80)
    print("📋 检查WandB界面中的指标:")
    print("  1. 在 'perf' 组中查找:")
    print("     - perf/mfu")
    print("     - perf/mfu_percent")
    print("     - perf/actual_flops")
    print("     - perf/flops_per_second")
    print("     - perf/tokens_per_second")
    print("     - perf/samples_per_second")
    print("  2. 在 'training' 组中查找:")
    print("     - training/loss")
    print("     - training/lr")
    print("     - training/grad_norm")
    print("  3. 在 'eval' 组中查找:")
    print("     - eval/overall_loss")
    print("     - eval/overall_accuracy")
    print("=" * 80)

def test_wandb_metric_groups():
    """测试WandB指标分组"""
    print("\n" + "=" * 80)
    print("🧪 测试WandB指标分组")
    print("=" * 80)
    
    # 初始化WandB
    wandb.init(
        project="qwen-classify-mfu-test",
        name="metric-groups-test",
        config={"test": True}
    )
    
    # 测试不同分组的指标
    for step in range(1, 6):
        # 训练指标
        wandb.log({
            "training/loss": 0.5 + step * 0.01,
            "training/lr": 1e-5,
            "training/grad_norm": 1.0 + step * 0.1,
        }, step=step)
        
        # 性能指标
        wandb.log({
            "perf/mfu": 0.3 + step * 0.02,
            "perf/mfu_percent": (0.3 + step * 0.02) * 100,
            "perf/actual_flops": 1e12,
            "perf/flops_per_second": 1e11 + step * 1e10,
            "perf/tokens_per_second": 1000 + step * 100,
            "perf/samples_per_second": 10 + step,
        }, step=step)
        
        # 评估指标
        if step % 2 == 0:
            wandb.log({
                "eval/overall_loss": 0.4 + step * 0.005,
                "eval/overall_accuracy": 0.8 + step * 0.01,
            }, step=step)
        
        time.sleep(0.1)
    
    wandb.finish()
    print("✅ 指标分组测试完成")

def check_wandb_display_issues():
    """检查WandB显示问题的常见原因"""
    print("\n" + "=" * 80)
    print("🔍 检查WandB显示问题的常见原因")
    print("=" * 80)
    
    print("📋 常见问题及解决方案:")
    print()
    print("1. 指标不显示:")
    print("   - 检查指标名称是否正确 (perf/mfu, perf/actual_flops)")
    print("   - 检查是否在正确的step记录")
    print("   - 检查指标值是否为数值类型")
    print()
    print("2. 指标值为0:")
    print("   - 检查flops_profile_freq设置")
    print("   - 检查actual_flops是否正确设置")
    print("   - 检查模型引用是否正确")
    print()
    print("3. 指标分组问题:")
    print("   - 确保指标名称包含分组前缀 (perf/, training/, eval/)")
    print("   - 检查WandB界面中的分组设置")
    print()
    print("4. 频率问题:")
    print("   - 检查perf_log_freq设置")
    print("   - 检查flops_profile_freq设置")
    print("   - 确保频率设置合理")
    print()
    print("5. 调试建议:")
    print("   - 使用NCCL_DEBUG=INFO查看详细日志")
    print("   - 检查WandB运行日志")
    print("   - 验证指标是否成功发送到WandB")
    print("=" * 80)

if __name__ == "__main__":
    try:
        # 测试MFU和FLOPs指标显示
        test_wandb_mfu_display()
        
        # 测试指标分组
        test_wandb_metric_groups()
        
        # 检查常见问题
        check_wandb_display_issues()
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc() 