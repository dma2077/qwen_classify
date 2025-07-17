#!/usr/bin/env python3
"""
测试training指标修复
验证training指标在所有步骤都能正确显示
"""

import wandb
import time
import random

def test_training_fix():
    """测试training指标修复"""
    
    run = wandb.init(
        project="test-training-fix",
        name="training-fix-verification",
        config={"test_type": "training_fix_validation"}
    )
    
    print("🚀 测试training指标修复...")
    
    # 定义指标
    wandb.define_metric("step")
    wandb.define_metric("training/*", step_metric="step")
    wandb.define_metric("eval/*", step_metric="step")
    wandb.define_metric("perf/*", step_metric="step")
    
    # 模拟训练循环
    total_steps = 15
    eval_interval = 5
    
    for step in range(1, total_steps + 1):
        print(f"\n--- Step {step}/{total_steps} ---")
        
        is_eval_step = (step % eval_interval == 0)
        
        # 基础training数据
        train_data = {
            "training/loss": 10.0 - step * 0.3 + random.uniform(-0.1, 0.1),
            "training/lr": 1e-5 * (0.995 ** step),
            "training/epoch": step / 10,
            "training/grad_norm": random.uniform(0.1, 2.0),
            "step": step
        }
        
        # 每隔几步添加性能指标
        if step % 2 == 0:  # 模拟perf_log_freq
            train_data.update({
                "perf/step_time": random.uniform(0.5, 1.5),
                "perf/steps_per_second": random.uniform(0.7, 2.0),
                "perf/mfu": random.uniform(0.2, 0.6),
            })
        
        if is_eval_step:
            print(f"🔍 Step {step}: 评估步骤 - 合并记录training+eval")
            
            # 添加eval数据
            eval_data = {
                "eval/overall_loss": 15.0 - step * 0.4 + random.uniform(-0.3, 0.3),
                "eval/overall_accuracy": min(0.85, step * 0.03 + random.uniform(-0.02, 0.02)),
            }
            
            # 合并数据
            combined_data = {**train_data, **eval_data}
            
            print(f"📊 合并记录:")
            print(f"   Training指标: {[k for k in combined_data.keys() if k.startswith('training/') or k.startswith('perf/')]}")
            print(f"   Eval指标: {[k for k in combined_data.keys() if k.startswith('eval/')]}")
            
            wandb.log(combined_data, step=step, commit=True)
            print(f"✅ 合并数据记录成功")
            
        else:
            print(f"📈 Step {step}: 常规训练步骤 - 仅记录training")
            
            print(f"📊 Training指标: {[k for k in train_data.keys() if k != 'step']}")
            
            wandb.log(train_data, step=step, commit=True)
            print(f"✅ Training数据记录成功")
        
        time.sleep(0.2)
    
    # 验证结果
    print(f"\n✅ 模拟完成")
    print(f"📊 预期结果:")
    print(f"   - Training指标应该在所有步骤 1-{total_steps} 显示")
    print(f"   - Eval指标应该在步骤 {', '.join([str(i) for i in range(eval_interval, total_steps + 1, eval_interval)])} 显示")
    print(f"   - Perf指标应该在偶数步骤显示")
    
    print(f"\n🔗 查看结果: {run.url}")
    
    time.sleep(3)
    wandb.finish()
    print("✅ 测试完成!")

if __name__ == "__main__":
    test_training_fix() 