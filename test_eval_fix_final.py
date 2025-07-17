#!/usr/bin/env python3
"""
测试eval指标修复是否有效
模拟真实的训练和评估流程
"""

import wandb
import time
import random

def test_eval_fix():
    """测试修复后的eval指标显示"""
    
    # 初始化WandB
    run = wandb.init(
        project="test-eval-fix",
        name="eval-fix-verification",
        config={
            "test_type": "eval_fix_validation",
            "description": "验证eval指标修复后的显示效果"
        }
    )
    
    print("🚀 开始测试修复后的eval指标...")
    
    # 1. 定义指标关系（模拟monitor的设置）
    print("📋 定义指标关系...")
    wandb.define_metric("step")
    wandb.define_metric("*", step_metric="step")
    
    print("✅ 已定义统一x轴：所有指标使用'step'")
    
    print("✅ 指标关系已定义")
    
    # 2. 模拟训练流程
    print("🏃 模拟训练流程...")
    for step in range(1, 51):  # 模拟50步训练
        
        # 每步记录训练指标
        train_data = {
            "training/loss": 10.0 - step * 0.15 + random.uniform(-0.1, 0.1),
            "training/lr": 1e-5 * (0.995 ** step),
            "training/epoch": step / 25,
            "step": step  # 🔥 使用统一的step字段
        }
        wandb.log(train_data, step=step, commit=True)
        
        # 每10步进行一次评估（模拟真实eval_steps）
        if step % 10 == 0:
            eval_data = {
                "eval/overall_loss": 15.0 - step * 0.2 + random.uniform(-0.3, 0.3),
                "eval/overall_accuracy": min(0.85, step * 0.015 + random.uniform(-0.02, 0.02)),
                "eval/foodx251_loss": 16.0 - step * 0.25 + random.uniform(-0.4, 0.4),
                "eval/foodx251_accuracy": min(0.8, step * 0.014 + random.uniform(-0.015, 0.015)),
                "eval/overall_samples": 1000,
                "eval/overall_correct": int(1000 * min(0.85, step * 0.015)),
                "step": step  # 🔥 关键：确保eval指标也有统一的step
            }
            
            print(f"📊 Step {step}: 记录eval指标")
            print(f"   eval/overall_loss: {eval_data['eval/overall_loss']:.4f}")
            print(f"   eval/overall_accuracy: {eval_data['eval/overall_accuracy']:.4f}")
            print(f"   step: {eval_data['step']}")
            
            wandb.log(eval_data, step=step, commit=True)
        
        # 短暂延迟模拟真实训练
        time.sleep(0.05)
    
    print("✅ 训练模拟完成")
    
    # 3. 最终提交
    print("🔄 最终数据同步...")
    wandb.log({}, commit=True)
    
    print(f"🔗 查看结果: {wandb.run.url}")
    print("📊 预期结果:")
    print("   1. training组指标应该连续显示（每步）")
    print("   2. eval组指标应该在step 10, 20, 30, 40, 50显示")
    print("   3. 两组指标应该在同一x轴上（step）")
    print("   4. eval指标应该随步数改善（loss下降，accuracy上升）")
    
    # 保持连接确保数据同步
    time.sleep(3)
    
    wandb.finish()
    print("✅ 测试完成!")

if __name__ == "__main__":
    test_eval_fix() 