#!/usr/bin/env python3
"""
测试WandB step同步问题的诊断脚本
"""

import os
import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_wandb_step_sync():
    """测试WandB step同步机制"""
    print("🔍 测试WandB step同步机制...")
    
    try:
        import wandb
        print(f"✅ WandB版本: {wandb.__version__}")
    except ImportError:
        print("❌ WandB未安装")
        return False
    
    # 结束现有运行
    if wandb.run is not None:
        wandb.finish()
    
    # 创建新运行
    run = wandb.init(
        project="wandb_step_sync_test",
        name=f"step_sync_test_{int(time.time())}",
        reinit=True
    )
    
    print(f"✅ 新运行创建: {run.url}")
    
    # 定义指标
    wandb.define_metric("step")
    wandb.define_metric("training/loss", step_metric="step", summary="min")
    wandb.define_metric("training/lr", step_metric="step", summary="last")
    wandb.define_metric("perf/step_time", step_metric="step", summary="last")
    
    print("✅ 指标定义完成")
    
    # 测试不同的step记录方法
    test_steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    print("\n🧪 方法1: 使用step参数")
    for step in test_steps[:3]:
        data = {
            "training/loss": 2.0 - step * 0.01,
            "training/lr": 1e-5,
        }
        
        if step % 20 == 0:
            data["perf/step_time"] = 4.0 + step * 0.1
        
        print(f"   📊 记录step {step}...")
        wandb.log(data, step=step, commit=True)
        
        current_step = getattr(wandb.run, 'step', 0)
        print(f"     期望step: {step}, WandB step: {current_step}")
        
        time.sleep(0.1)
    
    print("\n🧪 方法2: 手动控制内部step")
    for step in test_steps[3:6]:
        data = {
            "training/loss": 2.0 - step * 0.01,
            "training/lr": 1e-5,
        }
        
        if step % 20 == 0:
            data["perf/step_time"] = 4.0 + step * 0.1
        
        print(f"   📊 记录step {step}...")
        
        # 手动设置内部step
        if hasattr(wandb.run, '_step'):
            wandb.run._step = step - 1
            print(f"     🔧 设置内部step为: {step - 1}")
        
        wandb.log(data, commit=True)
        
        current_step = getattr(wandb.run, 'step', 0)
        print(f"     期望step: {step}, WandB step: {current_step}")
        
        time.sleep(0.1)
    
    print("\n🧪 方法3: 混合方法")
    for step in test_steps[6:]:
        data = {
            "training/loss": 2.0 - step * 0.01,
            "training/lr": 1e-5,
            "step": step,  # 显式添加step字段
        }
        
        if step % 20 == 0:
            data["perf/step_time"] = 4.0 + step * 0.1
        
        print(f"   📊 记录step {step}...")
        
        # 手动设置内部step + 使用step参数
        if hasattr(wandb.run, '_step'):
            wandb.run._step = step - 1
        
        wandb.log(data, step=step, commit=True)
        
        current_step = getattr(wandb.run, 'step', 0)
        print(f"     期望step: {step}, WandB step: {current_step}")
        
        time.sleep(0.1)
    
    print("\n🔍 检查WandB summary...")
    try:
        if hasattr(wandb.run, 'summary') and wandb.run.summary:
            summary_keys = list(wandb.run.summary.keys())
            print(f"✅ Summary包含 {len(summary_keys)} 个指标")
            print(f"   指标列表: {summary_keys}")
            
            # 检查特定指标的值
            for key in ['training/loss', 'training/lr', 'perf/step_time']:
                if key in wandb.run.summary:
                    value = wandb.run.summary[key]
                    print(f"   {key}: {value}")
                else:
                    print(f"   ❌ {key}: 未找到")
        else:
            print("❌ WandB summary为空")
    except Exception as e:
        print(f"❌ 检查summary失败: {e}")
    
    print(f"\n🔗 WandB URL: {run.url}")
    print("📋 请在WandB界面检查:")
    print("1. 图表是否显示所有step的数据")
    print("2. step轴是否连续")
    print("3. 性能指标是否正确显示")
    
    return True, run.url

if __name__ == "__main__":
    print("🔍 WandB Step同步诊断工具")
    print("=" * 50)
    
    success, url = test_wandb_step_sync()
    
    if success:
        print("\n✅ 测试完成!")
        print(f"🔗 请检查WandB URL: {url}")
    else:
        print("\n❌ 测试失败") 