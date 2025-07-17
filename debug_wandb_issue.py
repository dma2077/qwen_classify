#!/usr/bin/env python3
"""
诊断WandB eval指标不显示的问题
"""

import wandb
import time
import random

def debug_wandb_eval_issue():
    """诊断eval指标不显示的问题"""
    
    print("🔍 开始诊断WandB eval指标问题...")
    
    # 初始化WandB
    run = wandb.init(
        project="debug-eval-issue",
        name="eval-diagnosis",
        config={
            "test_type": "eval_diagnosis",
            "description": "诊断eval指标不显示问题"
        }
    )
    
    print(f"📊 WandB run初始化:")
    print(f"   Project: {run.project}")
    print(f"   Run ID: {run.id}")
    print(f"   Run Name: {run.name}")
    print(f"   Run URL: {run.url}")
    print(f"   Run State: {getattr(run, 'state', 'unknown')}")
    
    # 1. 测试基础的eval指标记录（不使用define_metric）
    print("\n🧪 测试1: 基础eval指标记录（无define_metric）")
    basic_eval_data = {
        "eval/basic_loss": 10.0,
        "eval/basic_accuracy": 0.5,
        "effective_step": 1
    }
    wandb.log(basic_eval_data, step=1, commit=True)
    print(f"   记录数据: {list(basic_eval_data.keys())}")
    time.sleep(2)
    
    # 2. 测试定义指标后的记录
    print("\n🧪 测试2: 使用define_metric后记录")
    
    # 定义指标
    wandb.define_metric("effective_step")
    wandb.define_metric("eval/*", step_metric="effective_step")
    wandb.define_metric("eval/defined_loss", summary="min", step_metric="effective_step")
    wandb.define_metric("eval/defined_accuracy", summary="max", step_metric="effective_step")
    print("   已定义指标关系")
    
    defined_eval_data = {
        "eval/defined_loss": 9.0,
        "eval/defined_accuracy": 0.6,
        "effective_step": 2
    }
    wandb.log(defined_eval_data, step=2, commit=True)
    print(f"   记录数据: {list(defined_eval_data.keys())}")
    time.sleep(2)
    
    # 3. 测试混合训练和eval指标
    print("\n🧪 测试3: 混合训练和eval指标")
    
    # 先记录一些训练指标
    for step in range(3, 8):
        train_data = {
            "training/loss": 10.0 - step * 0.5,
            "training/lr": 1e-5,
            "effective_step": step
        }
        wandb.log(train_data, step=step, commit=True)
        
        # 每2步记录eval指标
        if step % 2 == 0:
            eval_data = {
                "eval/mixed_loss": 12.0 - step * 0.8,
                "eval/mixed_accuracy": min(0.9, step * 0.1),
                "effective_step": step
            }
            wandb.log(eval_data, step=step, commit=True)
            print(f"   Step {step}: 记录eval指标")
        
        time.sleep(0.5)
    
    # 4. 检查WandB历史记录
    print("\n🔍 检查WandB历史记录:")
    try:
        history = run.history()
        print(f"   历史记录条数: {len(history)}")
        
        # 检查是否有eval指标
        eval_columns = [col for col in history.columns if col.startswith('eval/')]
        print(f"   Eval列: {eval_columns}")
        
        if eval_columns:
            print(f"   最后几条eval记录:")
            for col in eval_columns[:3]:  # 只显示前3个eval列
                non_null_values = history[col].dropna()
                if len(non_null_values) > 0:
                    print(f"     {col}: {non_null_values.tolist()}")
                else:
                    print(f"     {col}: 无数据")
        else:
            print("   ⚠️ 未找到eval列!")
            
    except Exception as e:
        print(f"   ❌ 获取历史记录失败: {e}")
    
    # 5. 强制同步测试
    print("\n🔄 强制同步测试:")
    final_eval_data = {
        "eval/final_test_loss": 5.0,
        "eval/final_test_accuracy": 0.95,
        "eval/final_test_samples": 1000,
        "effective_step": 10
    }
    
    # 多次记录同样的数据，强制同步
    for i in range(3):
        wandb.log(final_eval_data, step=10, commit=True)
        print(f"   第{i+1}次记录: commit=True")
        time.sleep(1)
    
    # 6. 检查是否有任何异常状态
    print("\n🔍 最终状态检查:")
    print(f"   Run状态: {getattr(run, 'state', 'unknown')}")
    print(f"   Run是否关闭: {getattr(run, '_closed', False)}")
    print(f"   Run模式: {getattr(run, 'mode', 'unknown')}")
    
    # 等待最终同步
    print("\n⏳ 等待最终同步...")
    time.sleep(5)
    
    print(f"\n🔗 请检查WandB界面: {run.url}")
    print("📋 预期看到的指标组:")
    print("   1. eval/ 组（应该包含所有eval指标）")
    print("   2. training/ 组（训练指标）")
    print("   3. 两组指标应在同一effective_step轴上")
    
    # 保持连接
    input("\n⌨️  按Enter键结束测试并关闭WandB run...")
    
    wandb.finish()
    print("✅ 诊断完成!")

if __name__ == "__main__":
    debug_wandb_issue() 