#!/usr/bin/env python3
"""
诊断step记录问题
"""

import wandb
import time
import random

def debug_step_issue():
    """诊断step记录的具体问题"""
    
    print("🔍 开始诊断step记录问题...")
    
    # 初始化WandB
    run = wandb.init(
        project="debug-step-issue",
        name="step-diagnosis",
        config={
            "test_type": "step_diagnosis",
            "description": "诊断step记录问题"
        }
    )
    
    print(f"📊 WandB run: {run.url}")
    
    # 1. 定义指标关系
    print("\n📋 定义指标关系...")
    wandb.define_metric("step")
    wandb.define_metric("*", step_metric="step")
    print("✅ 指标关系已定义")
    
    # 2. 测试基础记录（每步都记录training，部分记录eval）
    print("\n🧪 开始记录数据...")
    
    for step in range(1, 21):  # 测试20步
        print(f"\n--- Step {step} ---")
        
        # 每步记录training数据
        train_data = {
            "training/loss": 10.0 - step * 0.3 + random.uniform(-0.1, 0.1),
            "training/lr": 1e-5 * (0.99 ** step),
            "step": step
        }
        
        print(f"  📈 记录training数据: step={step}")
        print(f"     数据: {train_data}")
        
        # 重要：单独记录training数据
        try:
            wandb.log(train_data, step=step, commit=True)
            print(f"     ✅ training数据记录成功")
        except Exception as e:
            print(f"     ❌ training数据记录失败: {e}")
        
        # 每5步记录eval数据
        if step % 5 == 0:
            # 等待一小段时间，确保training数据已同步
            time.sleep(0.5)
            
            eval_data = {
                "eval/loss": 8.0 - step * 0.2 + random.uniform(-0.2, 0.2),
                "eval/accuracy": min(0.9, step * 0.04 + random.uniform(-0.01, 0.01)),
                "step": step
            }
            
            print(f"  📊 记录eval数据: step={step}")
            print(f"     数据: {eval_data}")
            
            # 重要：单独记录eval数据
            try:
                wandb.log(eval_data, step=step, commit=True)
                print(f"     ✅ eval数据记录成功")
            except Exception as e:
                print(f"     ❌ eval数据记录失败: {e}")
        
        # 短暂延迟
        time.sleep(0.2)
    
    # 3. 验证数据记录
    print("\n🔍 验证数据记录...")
    time.sleep(3)  # 等待同步
    
    try:
        history = run.history()
        print(f"   历史记录总条数: {len(history)}")
        print(f"   所有列: {list(history.columns)}")
        
        # 检查step列
        if 'step' in history.columns:
            step_values = history['step'].dropna().tolist()
            print(f"   Step值: {sorted(step_values)}")
        
        # 检查training列
        training_cols = [col for col in history.columns if col.startswith('training/')]
        print(f"   Training列: {training_cols}")
        if training_cols:
            for col in training_cols:
                non_null = history[col].dropna()
                print(f"     {col}: {len(non_null)} 条记录")
        
        # 检查eval列
        eval_cols = [col for col in history.columns if col.startswith('eval/')]
        print(f"   Eval列: {eval_cols}")
        if eval_cols:
            for col in eval_cols:
                non_null = history[col].dropna()
                print(f"     {col}: {len(non_null)} 条记录")
                
        # 显示前10行数据
        print(f"\n   前10行数据:")
        print(history.head(10).to_string())
                
    except Exception as e:
        print(f"   ❌ 获取历史记录失败: {e}")
    
    # 4. 强制最终同步
    print("\n🔄 强制最终同步...")
    for i in range(3):
        wandb.log({}, commit=True)
        time.sleep(1)
    
    print(f"\n🔗 请检查WandB界面: {run.url}")
    print("📋 预期结果:")
    print("   - training/loss, training/lr: 应该在step 1-20都有数据")
    print("   - eval/loss, eval/accuracy: 应该在step 5,10,15,20有数据")
    print("   - 所有数据应该在同一个'step'轴上")
    
    input("\n⌨️  按Enter键结束...")
    wandb.finish()
    print("✅ 诊断完成!")

if __name__ == "__main__":
    debug_step_issue() 