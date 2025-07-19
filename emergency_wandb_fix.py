#!/usr/bin/env python3
"""
紧急修复WandB显示问题
"""

import os
import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def emergency_wandb_fix():
    """紧急修复WandB显示问题"""
    print("🚨 紧急修复WandB显示问题...")
    
    try:
        import wandb
        print(f"✅ WandB版本: {wandb.__version__}")
    except ImportError:
        print("❌ WandB未安装")
        return
    
    # 创建新的测试运行
    print("🔧 创建新的测试运行...")
    
    run = wandb.init(
        project="emergency_fix_test",
        name=f"fix_test_{int(time.time())}",
        tags=["emergency", "fix", "test"],
        notes="紧急修复WandB显示问题的测试"
    )
    
    print(f"✅ 测试运行创建成功")
    print(f"   🔗 URL: {run.url}")
    
    # 定义指标
    print("🔧 定义指标...")
    wandb.define_metric("step")
    wandb.define_metric("training/*", step_metric="step")
    wandb.define_metric("eval/*", step_metric="step")
    wandb.define_metric("perf/*", step_metric="step")
    
    # 记录测试数据 - 模拟真实训练过程
    print("📊 记录测试数据...")
    
    for step in range(1, 51):  # 50个步骤
        # Training指标
        training_data = {
            "training/loss": 1.0 - step * 0.01,
            "training/lr": 5e-6,
            "training/epoch": step * 0.02,
            "training/grad_norm": 1.5
        }
        
        # 每10步记录eval指标
        if step % 10 == 0:
            eval_data = {
                "eval/overall_loss": 0.8 - step * 0.005,
                "eval/overall_accuracy": 0.5 + step * 0.01,
                "eval/overall_samples": 1000,
                "eval/overall_correct": 500 + step * 10
            }
            all_data = {**training_data, **eval_data}
        else:
            all_data = training_data
        
        # 每20步记录性能指标
        if step % 20 == 0:
            perf_data = {
                "perf/step_time": 4.0,
                "perf/mfu": 0.3,
                "perf/tokens_per_second": 1000
            }
            all_data.update(perf_data)
        
        # 记录数据
        wandb.log(all_data, step=step, commit=True)
        
        # 输出进度
        if step % 10 == 0:
            print(f"  📊 Step {step}: 已记录 {len(all_data)} 个指标")
            if "eval/overall_loss" in all_data:
                print(f"    📈 包含eval指标")
            if "perf/step_time" in all_data:
                print(f"    ⚡ 包含perf指标")
        
        # 短暂延迟
        time.sleep(0.05)
    
    # 强制同步
    print("🔄 强制同步数据...")
    try:
        if hasattr(run, 'sync'):
            run.sync()
        time.sleep(2)  # 等待同步完成
    except Exception as sync_error:
        print(f"⚠️ 同步失败: {sync_error}")
    
    # 检查历史数据
    print("🔍 检查历史数据...")
    try:
        history = run.history()
        if not history.empty:
            print(f"✅ 历史数据: {len(history)}行")
            print(f"📋 列名: {list(history.columns)}")
            
            # 检查各类指标
            training_cols = [col for col in history.columns if 'training/' in col]
            eval_cols = [col for col in history.columns if 'eval/' in col]
            perf_cols = [col for col in history.columns if 'perf/' in col]
            
            print(f"📈 Training指标: {len(training_cols)}个")
            print(f"📊 Eval指标: {len(eval_cols)}个")
            print(f"⚡ Perf指标: {len(perf_cols)}个")
            
            # 检查eval指标的数据点
            if eval_cols:
                for col in eval_cols:
                    non_null_count = history[col].notna().sum()
                    print(f"  - {col}: {non_null_count}个非空值")
        else:
            print("❌ 历史数据为空")
    except Exception as history_error:
        print(f"⚠️ 检查历史数据失败: {history_error}")
    
    print(f"\n🎉 测试完成！")
    print(f"🔗 请检查WandB URL: {run.url}")
    print("📊 应该能看到:")
    print("  • 50个training数据点")
    print("  • 5个eval数据点 (step 10, 20, 30, 40, 50)")
    print("  • 2个perf数据点 (step 20, 40)")
    
    # 结束运行
    wandb.finish()

def force_wandb_sync():
    """强制同步当前WandB运行"""
    print("🔄 尝试强制同步当前WandB运行...")
    
    try:
        import wandb
        if wandb.run is not None:
            print(f"✅ 发现活跃运行: {wandb.run.name}")
            print(f"🔗 URL: {wandb.run.url}")
            
            # 尝试多种同步方法
            try:
                if hasattr(wandb.run, 'sync'):
                    wandb.run.sync()
                    print("✅ sync()调用成功")
            except Exception as sync1_error:
                print(f"⚠️ sync()失败: {sync1_error}")
            
            try:
                if hasattr(wandb.run, '_sync_dir'):
                    wandb.run._sync_dir()
                    print("✅ _sync_dir()调用成功")
            except Exception as sync2_error:
                print(f"⚠️ _sync_dir()失败: {sync2_error}")
            
            # 等待同步
            time.sleep(3)
            print("🔄 同步等待完成")
            
        else:
            print("⚠️ 当前没有活跃的WandB运行")
    except Exception as e:
        print(f"❌ 强制同步失败: {e}")

if __name__ == "__main__":
    print("选择操作:")
    print("1. 紧急修复测试")
    print("2. 强制同步当前运行")
    
    choice = input("请输入选择 (1或2): ").strip()
    
    if choice == "1":
        emergency_wandb_fix()
    elif choice == "2":
        force_wandb_sync()
    else:
        print("无效选择")
        emergency_wandb_fix()  # 默认运行修复测试 