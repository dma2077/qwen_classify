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
    
    total_steps = 20  # 减少到20步，便于调试
    eval_interval = 5  # 每5步评估一次
    
    for step in range(1, total_steps + 1):
        print(f"\n--- Step {step}/{total_steps} ---")
        
        # 每步记录训练指标
        train_data = {
            "training/loss": 10.0 - step * 0.3 + random.uniform(-0.1, 0.1),
            "training/lr": 1e-5 * (0.995 ** step),
            "training/epoch": step / 10,
            "step": step  # 🔥 使用统一的step字段
        }
        
        try:
            wandb.log(train_data, step=step, commit=True)
            print(f"✅ Training数据记录成功: step={step}")
        except Exception as e:
            print(f"❌ Training数据记录失败: {e}")
        
        # 每eval_interval步进行一次评估
        if step % eval_interval == 0:
            eval_data = {
                "eval/overall_loss": 15.0 - step * 0.4 + random.uniform(-0.3, 0.3),
                "eval/overall_accuracy": min(0.85, step * 0.03 + random.uniform(-0.02, 0.02)),
                "eval/foodx251_loss": 16.0 - step * 0.5 + random.uniform(-0.4, 0.4),
                "eval/foodx251_accuracy": min(0.8, step * 0.025 + random.uniform(-0.015, 0.015)),
                "eval/overall_samples": 1000,
                "eval/overall_correct": int(1000 * min(0.85, step * 0.03)),
                "step": step  # 🔥 关键：确保eval指标也有统一的step
            }
            
            print(f"📊 记录eval指标:")
            print(f"   eval/overall_loss: {eval_data['eval/overall_loss']:.4f}")
            print(f"   eval/overall_accuracy: {eval_data['eval/overall_accuracy']:.4f}")
            print(f"   step: {eval_data['step']}")
            
            try:
                wandb.log(eval_data, step=step, commit=True)
                print(f"✅ Eval数据记录成功: step={step}")
            except Exception as e:
                print(f"❌ Eval数据记录失败: {e}")
        
        # 短暂延迟模拟真实训练
        time.sleep(0.1)
    
    print(f"\n✅ 训练模拟完成 - 总共{total_steps}步")
    print(f"📊 预期数据:")
    print(f"   Training: 步骤 1-{total_steps}")
    print(f"   Eval: 步骤 {', '.join([str(i) for i in range(eval_interval, total_steps + 1, eval_interval)])}")
    
    # 3. 验证数据记录
    print("\n🔍 验证数据记录...")
    time.sleep(3)  # 等待数据同步
    
    try:
        history = run.history()
        print(f"   历史记录总条数: {len(history)}")
        
        # 检查step列
        if 'step' in history.columns:
            step_values = sorted(history['step'].dropna().unique().tolist())
            print(f"   记录的Step值: {step_values}")
        
        # 检查training列
        training_cols = [col for col in history.columns if col.startswith('training/')]
        if training_cols:
            print(f"   Training列: {training_cols}")
            for col in training_cols:
                non_null = history[col].dropna()
                steps_with_data = sorted(history[history[col].notna()]['step'].tolist())
                print(f"     {col}: {len(non_null)} 条记录，步骤: {steps_with_data}")
        
        # 检查eval列
        eval_cols = [col for col in history.columns if col.startswith('eval/')]
        if eval_cols:
            print(f"   Eval列: {eval_cols}")
            for col in eval_cols:
                non_null = history[col].dropna()
                steps_with_data = sorted(history[history[col].notna()]['step'].tolist())
                print(f"     {col}: {len(non_null)} 条记录，步骤: {steps_with_data}")
                
    except Exception as e:
        print(f"   ❌ 获取历史记录失败: {e}")
    
    # 4. 最终提交
    print("\n🔄 最终数据同步...")
    wandb.log({}, commit=True)
    
    print(f"\n🔗 查看结果: {wandb.run.url}")
    print("📊 预期结果:")
    print(f"   1. training组指标应该连续显示（步骤 1-{total_steps}）")
    print(f"   2. eval组指标应该在步骤 {', '.join([str(i) for i in range(eval_interval, total_steps + 1, eval_interval)])} 显示")
    print("   3. 两组指标应该在同一x轴上（step）")
    print("   4. eval指标应该随步数改善（loss下降，accuracy上升）")
    
    # 保持连接确保数据同步
    time.sleep(3)
    
    wandb.finish()
    print("✅ 测试完成!")

if __name__ == "__main__":
    test_eval_fix() 