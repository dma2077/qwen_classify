#!/usr/bin/env python3
"""
测试最终修复后的WandB记录逻辑
模拟实际训练循环的行为
"""

import wandb
import time
import random

def test_final_wandb_fix():
    """测试最终修复后的WandB记录逻辑"""
    
    # 初始化WandB
    run = wandb.init(
        project="test-final-wandb-fix",
        name="final-fix-verification",
        config={
            "test_type": "final_wandb_fix_validation",
            "description": "验证最终修复后的WandB记录效果"
        }
    )
    
    print("🚀 开始测试最终修复后的WandB记录...")
    
    # 1. 定义指标关系（模拟修复后的monitor设置）
    print("📋 定义指标关系...")
    wandb.define_metric("step")
    wandb.define_metric("training/*", step_metric="step")
    wandb.define_metric("eval/*", step_metric="step")
    wandb.define_metric("perf/*", step_metric="step")
    
    print("✅ 已定义统一x轴：training/*, eval/*, perf/* 指标使用'step'")
    
    # 2. 模拟训练循环
    total_steps = 20
    eval_interval = 5  # 每5步评估一次
    
    print(f"\n📊 开始模拟训练循环（{total_steps}步，每{eval_interval}步评估）...")
    
    for step in range(1, total_steps + 1):
        print(f"\n--- Step {step}/{total_steps} ---")
        
        # 模拟检查是否是eval步骤
        is_eval_step = (step % eval_interval == 0)
        
        if is_eval_step:
            print(f"🔍 Step {step}: 执行评估+训练合并记录")
            
            # 模拟评估数据
            eval_loss = 15.0 - step * 0.4 + random.uniform(-0.3, 0.3)
            eval_accuracy = min(0.85, step * 0.03 + random.uniform(-0.02, 0.02))
            
            # 模拟当前训练数据
            current_training_data = {
                "training/loss": 10.0 - step * 0.3 + random.uniform(-0.1, 0.1),
                "training/lr": 1e-5 * (0.995 ** step),
                "training/epoch": step / 10,
                "training/grad_norm": random.uniform(0.1, 2.0),
            }
            
            # 准备eval数据
            eval_data = {
                "eval/overall_loss": eval_loss,
                "eval/overall_accuracy": eval_accuracy,
                "eval/foodx251_loss": eval_loss + random.uniform(-0.2, 0.2),
                "eval/foodx251_accuracy": eval_accuracy + random.uniform(-0.05, 0.05),
                "eval/overall_samples": 1000,
                "eval/overall_correct": int(1000 * eval_accuracy),
            }
            
            # 🔥 关键：合并training和eval数据，一次性记录
            combined_data = {**current_training_data, **eval_data}
            combined_data["step"] = step
            
            print(f"📊 合并记录数据:")
            print(f"   训练指标: {list(current_training_data.keys())}")
            print(f"   评估指标: {list(eval_data.keys())}")
            print(f"   step: {step}")
            
            try:
                wandb.log(combined_data, step=step, commit=True)
                print(f"✅ 训练+评估数据合并记录成功: step={step}")
            except Exception as e:
                print(f"❌ 合并数据记录失败: {e}")
        
        else:
            print(f"📈 Step {step}: 仅记录训练数据")
            
            # 只记录训练数据
            train_data = {
                "training/loss": 10.0 - step * 0.3 + random.uniform(-0.1, 0.1),
                "training/lr": 1e-5 * (0.995 ** step),
                "training/epoch": step / 10,
                "training/grad_norm": random.uniform(0.1, 2.0),
                "step": step
            }
            
            print(f"📊 记录训练数据:")
            print(f"   训练指标: {list(train_data.keys())}")
            print(f"   step: {step}")
            
            try:
                wandb.log(train_data, step=step, commit=True)
                print(f"✅ 训练数据记录成功: step={step}")
            except Exception as e:
                print(f"❌ 训练数据记录失败: {e}")
        
        # 短暂延迟模拟真实训练
        time.sleep(0.2)
    
    print(f"\n✅ 训练模拟完成 - 总共{total_steps}步")
    print(f"📊 预期数据:")
    print(f"   Training: 步骤 1-{total_steps} (所有步骤)")
    print(f"   Eval: 步骤 {', '.join([str(i) for i in range(eval_interval, total_steps + 1, eval_interval)])} (eval步骤)")
    
    # 3. 验证数据记录
    print("\n🔍 验证数据记录...")
    time.sleep(3)  # 等待数据同步
    
    try:
        # 使用WandB API获取run数据
        api = wandb.Api()
        run_path = f"{run.entity}/{run.project}/{run.id}"
        api_run = api.run(run_path)
        
        # 获取历史数据
        history = api_run.scan_history()
        history_list = list(history)
        
        print(f"   历史记录总条数: {len(history_list)}")
        
        if history_list:
            # 收集所有键
            all_keys = set()
            for record in history_list:
                all_keys.update(record.keys())
            
            # 检查step
            steps = [record.get('step') for record in history_list if 'step' in record]
            if steps:
                step_values = sorted([s for s in steps if s is not None])
                print(f"   记录的Step值: {step_values}")
            
            # 检查training列
            training_keys = [k for k in all_keys if k.startswith('training/')]
            if training_keys:
                print(f"   Training指标: {training_keys}")
                for key in training_keys:
                    records_with_key = [r for r in history_list if key in r and r[key] is not None]
                    steps_with_data = sorted([r.get('step') for r in records_with_key if 'step' in r])
                    print(f"     {key}: {len(records_with_key)} 条记录，步骤: {steps_with_data}")
            
            # 检查eval列
            eval_keys = [k for k in all_keys if k.startswith('eval/')]
            if eval_keys:
                print(f"   Eval指标: {eval_keys}")
                for key in eval_keys:
                    records_with_key = [r for r in history_list if key in r and r[key] is not None]
                    steps_with_data = sorted([r.get('step') for r in records_with_key if 'step' in r])
                    print(f"     {key}: {len(records_with_key)} 条记录，步骤: {steps_with_data}")
        else:
            print("   ⚠️ 没有找到历史记录")
                
    except Exception as e:
        print(f"   ❌ 获取历史记录失败: {e}")
        print(f"   💡 这可能是因为数据还在同步中，请稍后在WandB界面查看")
    
    # 4. 最终提交
    print("\n🔄 最终数据同步...")
    wandb.log({}, commit=True)
    
    print(f"\n🔗 查看结果: {run.url}")
    print("📊 预期结果:")
    print(f"   1. training组指标应该在所有步骤显示（步骤 1-{total_steps}）")
    print(f"   2. eval组指标应该在步骤 {', '.join([str(i) for i in range(eval_interval, total_steps + 1, eval_interval)])} 显示")
    print("   3. 两组指标应该在同一x轴上（step）对齐")
    print("   4. eval指标应该随步数改善（loss下降，accuracy上升）")
    print("   5. 在eval步骤中，training和eval指标应该同时出现")
    
    # 保持连接确保数据同步
    time.sleep(3)
    
    wandb.finish()
    print("✅ 测试完成!")

if __name__ == "__main__":
    test_final_wandb_fix() 