#!/usr/bin/env python3
"""
简单的WandB eval指标测试
验证eval指标是否能正确显示在WandB界面中
"""

import wandb
import time
import random

def test_wandb_eval_metrics():
    """测试WandB eval指标显示"""
    
    # 初始化WandB
    run = wandb.init(
        project="test-eval-metrics",
        name="eval-display-test",
        config={
            "test_type": "eval_metrics_display",
            "description": "测试eval指标在WandB中的显示"
        }
    )
    
    print("🚀 开始测试WandB eval指标...")
    
    # 1. 定义eval指标
    print("📋 定义eval指标...")
    wandb.define_metric("eval/overall_loss", summary="min")
    wandb.define_metric("eval/overall_accuracy", summary="max")
    wandb.define_metric("eval/foodx251_loss", summary="min")
    wandb.define_metric("eval/foodx251_accuracy", summary="max")
    wandb.define_metric("eval/foodx251_samples", summary="last")
    wandb.define_metric("eval/overall_samples", summary="last")
    wandb.define_metric("eval/overall_correct", summary="last")
    
    # 2. 强制初始化eval图表
    print("📊 强制初始化eval图表...")
    init_step = 999999
    initial_eval_data = {
        "eval/overall_loss": float('nan'),
        "eval/overall_accuracy": float('nan'),
        "eval/foodx251_loss": float('nan'),
        "eval/foodx251_accuracy": float('nan'),
        "eval/foodx251_samples": 0,
        "eval/overall_samples": 0,
        "eval/overall_correct": 0,
    }
    wandb.log(initial_eval_data, step=init_step, commit=False)
    
    # 3. 记录训练指标（作为对比）
    print("📈 记录训练指标...")
    for step in range(1, 21):
        # 训练指标
        train_data = {
            "training/loss": 10.0 - step * 0.4 + random.uniform(-0.1, 0.1),
            "training/lr": 1e-5,
            "training/epoch": step / 10,
            "global_step": step
        }
        wandb.log(train_data, step=step, commit=True)
        
        # 每5步记录eval指标
        if step % 5 == 0:
            eval_data = {
                "eval/overall_loss": 15.0 - step * 0.6 + random.uniform(-0.2, 0.2),
                "eval/overall_accuracy": min(0.8, step * 0.04 + random.uniform(-0.01, 0.01)),
                "eval/foodx251_loss": 16.0 - step * 0.7 + random.uniform(-0.3, 0.3),
                "eval/foodx251_accuracy": min(0.75, step * 0.035 + random.uniform(-0.01, 0.01)),
                "eval/foodx251_samples": 1000,
                "eval/overall_samples": 1000,
                "eval/overall_correct": int(1000 * min(0.8, step * 0.04)),
            }
            
            print(f"📊 Step {step}: 记录eval指标 {list(eval_data.keys())}")
            wandb.log(eval_data, step=step, commit=True)
        
        time.sleep(0.1)  # 短暂延迟
    
    # 4. 最终提交
    print("✅ 最终提交...")
    wandb.log({}, commit=True)
    
    print(f"🔗 查看结果: {wandb.run.url}")
    print("📊 检查WandB界面中是否显示以下eval指标组:")
    print("   - eval/overall_loss")
    print("   - eval/overall_accuracy") 
    print("   - eval/foodx251_loss")
    print("   - eval/foodx251_accuracy")
    print("   - eval/foodx251_samples")
    print("   - eval/overall_samples")
    print("   - eval/overall_correct")
    
    # 保持运行一段时间确保数据同步
    print("⏱️  等待数据同步...")
    time.sleep(3)
    
    wandb.finish()
    print("✅ 测试完成!")

if __name__ == "__main__":
    test_wandb_eval_metrics() 