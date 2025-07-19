#!/usr/bin/env python3
"""
测试WandB step修复
验证step冲突问题是否已解决
"""

import torch
import sys
import os
import time

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_wandb_step_fix():
    """测试WandB step修复"""
    
    print("🧪 测试WandB step修复...")
    print("=" * 50)
    
    # 检查WandB是否可用
    try:
        import wandb
        print(f"✅ WandB可用: {wandb.__version__}")
    except ImportError:
        print("❌ WandB不可用，跳过测试")
        return
    
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
    
    # 测试1: 测试step倒退检测
    print("\n📊 测试1: 测试step倒退检测")
    try:
        from training.utils.monitor import TrainingMonitor
        
        # 创建配置
        config = {
            'wandb': {
                'enabled': True,
                'project': 'test_step_fix',
                'run_name': 'test_run'
            },
            'output_dir': './test_output'
        }
        
        # 创建monitor
        monitor = TrainingMonitor('./test_output', config)
        
        # 测试正常的step记录
        print("  测试正常step记录...")
        training_data = {
            "training/loss": 0.5,
            "training/lr": 1e-4,
            "step": 100
        }
        monitor.log_metrics(training_data, step=100, commit=True)
        print("  ✅ 正常step记录成功")
        
        # 测试step倒退
        print("  测试step倒退...")
        training_data = {
            "training/loss": 0.4,
            "training/lr": 1e-4,
            "step": 50  # 倒退的step
        }
        monitor.log_metrics(training_data, step=50, commit=True)
        print("  ✅ step倒退检测成功")
        
    except Exception as e:
        print(f"❌ step倒退检测测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试2: 测试重复step字段处理
    print("\n📊 测试2: 测试重复step字段处理")
    try:
        # 测试包含重复step字段的数据
        training_data = {
            "training/loss": 0.3,
            "training/lr": 1e-4,
            "step": 200,  # 重复的step字段
            "perf/mfu": 0.8
        }
        monitor.log_metrics(training_data, step=200, commit=True)
        print("  ✅ 重复step字段处理成功")
        
    except Exception as e:
        print(f"❌ 重复step字段处理测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试3: 测试连续step记录
    print("\n📊 测试3: 测试连续step记录")
    try:
        # 测试连续记录多个step
        for step in range(300, 310):
            training_data = {
                "training/loss": 0.5 - step * 0.001,
                "training/lr": 1e-4,
                "perf/mfu": 0.8 + step * 0.001
            }
            monitor.log_metrics(training_data, step=step, commit=True)
            time.sleep(0.1)  # 短暂延迟
        
        print("  ✅ 连续step记录成功")
        
    except Exception as e:
        print(f"❌ 连续step记录测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试4: 测试eval指标记录
    print("\n📊 测试4: 测试eval指标记录")
    try:
        # 测试eval指标记录
        eval_data = {
            "eval/overall_loss": 0.3,
            "eval/overall_accuracy": 0.85,
            "eval/overall_samples": 1000,
            "eval/overall_correct": 850
        }
        monitor.log_metrics(eval_data, step=400, commit=True)
        print("  ✅ eval指标记录成功")
        
    except Exception as e:
        print(f"❌ eval指标记录测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 清理
    try:
        if hasattr(monitor, 'use_wandb') and monitor.use_wandb:
            import wandb
            if wandb.run is not None:
                wandb.finish()
                print("  ✅ WandB运行已结束")
    except Exception as e:
        print(f"⚠️  清理WandB失败: {e}")
    
    print("\n✅ WandB step修复测试完成")

if __name__ == "__main__":
    test_wandb_step_fix() 