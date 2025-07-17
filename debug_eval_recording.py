#!/usr/bin/env python3
"""
诊断eval指标记录问题
"""

import wandb
import time
import random

def debug_eval_recording():
    """诊断为什么eval指标没有被记录"""
    
    print("🔍 开始诊断eval指标记录问题...")
    
    # 初始化WandB
    run = wandb.init(
        project="debug-eval-recording",
        name="eval-recording-diagnosis",
        config={"test_type": "eval_recording_debug"}
    )
    
    print(f"📋 WandB Run Info:")
    print(f"   Project: {run.project}")
    print(f"   Name: {run.name}")
    print(f"   ID: {run.id}")
    print(f"   URL: {run.url}")
    
    # 测试1: 最简单的eval指标记录
    print("\n🧪 测试1: 最简单的eval指标记录")
    try:
        simple_eval_data = {
            "eval/simple_test": 0.5,
            "step": 1
        }
        wandb.log(simple_eval_data, step=1)
        print("✅ 简单eval指标记录成功")
        time.sleep(1)
    except Exception as e:
        print(f"❌ 简单eval指标记录失败: {e}")
    
    # 测试2: 分别记录training和eval（不使用commit=True）
    print("\n🧪 测试2: 分别记录training和eval（无commit参数）")
    for step in [2, 3, 4, 5]:
        # Training数据
        train_data = {
            "training/loss": 10.0 - step * 0.5,
            "training/accuracy": step * 0.1,
            "step": step
        }
        wandb.log(train_data, step=step)
        print(f"📊 Step {step}: 记录training数据")
        
        # 如果是eval步骤，记录eval数据
        if step % 2 == 0:  # 每2步eval一次
            eval_data = {
                "eval/loss": 8.0 - step * 0.3,
                "eval/accuracy": step * 0.15,
                "step": step
            }
            wandb.log(eval_data, step=step)
            print(f"📊 Step {step}: 记录eval数据")
        
        time.sleep(0.5)
    
    # 测试3: 使用单独的step进行记录
    print("\n🧪 测试3: 使用单独的step记录")
    for step in [6, 7, 8]:
        # 只记录training
        wandb.log({"training/loss": 5.0 - step * 0.2, "step": step}, step=step)
        print(f"📊 Step {step}: 只记录training")
        time.sleep(0.5)
    
    for step in [6, 8]:  # 只在特定步骤记录eval
        # 只记录eval
        wandb.log({"eval/loss": 4.0 - step * 0.1, "step": step}, step=step)
        print(f"📊 Step {step}: 只记录eval")
        time.sleep(0.5)
    
    # 测试4: 强制定义指标并记录
    print("\n🧪 测试4: 强制定义指标并记录")
    
    # 重新定义指标
    wandb.define_metric("step")
    wandb.define_metric("training/*", step_metric="step")
    wandb.define_metric("eval/*", step_metric="step")
    
    for step in [9, 10]:
        # Training
        train_data = {
            "training/final_loss": 3.0 - step * 0.1,
            "training/final_accuracy": step * 0.05,
            "step": step
        }
        wandb.log(train_data, step=step, commit=False)
        print(f"📊 Step {step}: 记录training (commit=False)")
        
        # Eval
        eval_data = {
            "eval/final_loss": 2.0 - step * 0.05,
            "eval/final_accuracy": step * 0.08,
            "step": step
        }
        wandb.log(eval_data, step=step, commit=True)
        print(f"📊 Step {step}: 记录eval (commit=True)")
        
        time.sleep(1)
    
    # 测试5: 验证数据记录
    print("\n🔍 验证所有记录的数据...")
    time.sleep(3)  # 等待同步
    
    try:
        # 尝试访问summary来查看最终状态
        summary = run.summary
        print(f"📋 Run Summary:")
        for key, value in summary.items():
            if not key.startswith('_'):
                print(f"   {key}: {value}")
                
        # 尝试使用API获取历史
        api = wandb.Api()
        run_path = f"{run.entity}/{run.project}/{run.id}"
        api_run = api.run(run_path)
        
        history = list(api_run.scan_history())
        print(f"\n📊 历史记录分析:")
        print(f"   总记录数: {len(history)}")
        
        if history:
            all_keys = set()
            for record in history:
                all_keys.update(record.keys())
            
            training_keys = [k for k in all_keys if k.startswith('training/')]
            eval_keys = [k for k in all_keys if k.startswith('eval/')]
            
            print(f"   Training指标: {training_keys}")
            print(f"   Eval指标: {eval_keys}")
            
            # 详细分析每个指标
            for key in sorted(all_keys):
                if key.startswith(('training/', 'eval/')):
                    records_with_key = [r for r in history if key in r and r[key] is not None]
                    steps = [r.get('step') for r in records_with_key if 'step' in r]
                    print(f"   {key}: {len(records_with_key)} 条记录，步骤: {sorted(steps)}")
        
    except Exception as e:
        print(f"❌ 数据验证失败: {e}")
    
    print(f"\n🔗 查看完整结果: {run.url}")
    print("💡 请在WandB界面查看所有指标是否正确显示")
    
    # 保持连接
    time.sleep(2)
    wandb.finish()
    print("✅ 诊断完成!")

if __name__ == "__main__":
    debug_eval_recording() 