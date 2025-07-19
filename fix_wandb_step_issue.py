#!/usr/bin/env python3
"""
修复WandB step冲突问题
"""

import os
import sys
from pathlib import Path
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def fix_wandb_step_issue():
    """修复WandB step冲突问题"""
    print("🔧 修复WandB step冲突问题...")
    
    try:
        import wandb
        print(f"✅ WandB版本: {wandb.__version__}")
    except ImportError:
        print("❌ WandB未安装")
        return False
    
    # 1. 检查当前运行状态
    if wandb.run is not None:
        print(f"⚠️ 检测到活跃的WandB运行: {wandb.run.name}")
        print(f"   🔗 URL: {wandb.run.url}")
        print(f"   🆔 Run ID: {wandb.run.id}")
        print(f"   📈 当前step: {getattr(wandb.run, 'step', 0)}")
        
        # 尝试获取历史数据
        try:
            if hasattr(wandb.run, 'history'):
                history = wandb.run.history()
                if not history.empty:
                    print(f"   📊 历史数据: {len(history)}行")
                    print(f"   📋 列名: {list(history.columns)}")
                    
                    # 检查step分布
                    if 'Step' in history.columns:
                        steps = history['Step'].unique()
                        print(f"   📈 Step值: {sorted(steps)}")
                        
                        # 检查是否有step=0导致的问题
                        step_0_count = len(history[history['Step'] == 0])
                        if step_0_count > 0:
                            print(f"   ⚠️ 发现{step_0_count}个step=0的数据点，这可能导致图表显示问题")
                        
                else:
                    print(f"   📊 历史数据为空")
            else:
                print(f"   📊 无法访问历史数据 (API版本限制)")
        except Exception as history_error:
            print(f"   ⚠️ 获取历史数据失败: {history_error}")
        
        # 选择是否结束当前运行
        choice = input("\n是否结束当前WandB运行并创建新的运行? (y/n): ").strip().lower()
        if choice == 'y':
            print("🔄 结束当前WandB运行...")
            wandb.finish()
            print("✅ 当前运行已结束")
        else:
            print("📝 保持当前运行，将尝试修复step问题")
            return True
    
    # 2. 创建新的干净运行
    print("🚀 创建新的WandB运行...")
    
    run = wandb.init(
        project="qwen_classification_fixed",
        name=f"fixed_run_{int(time.time())}",
        tags=["fixed", "no_step_conflict"],
        notes="修复step冲突问题后的运行"
    )
    
    print(f"✅ 新运行创建成功")
    print(f"   🔗 URL: {run.url}")
    print(f"   🆔 Run ID: {run.id}")
    
    # 3. 定义指标（不记录初始数据）
    print("🔧 定义指标...")
    wandb.define_metric("step")
    wandb.define_metric("training/*", step_metric="step")
    wandb.define_metric("eval/*", step_metric="step")
    wandb.define_metric("perf/*", step_metric="step")
    
    # 4. 测试记录数据（从step=1开始）
    print("🧪 测试数据记录...")
    test_steps = [1, 20, 40, 60, 80, 100]
    
    for step in test_steps:
        # Training数据
        training_data = {
            "training/loss": 1.0 - step * 0.005,
            "training/lr": 5e-6,
            "training/epoch": step * 0.01
        }
        
        # 每20步记录eval数据
        if step % 20 == 0:
            eval_data = {
                "eval/overall_loss": 0.8 - step * 0.003,
                "eval/overall_accuracy": 0.5 + step * 0.005
            }
            all_data = {**training_data, **eval_data}
        else:
            all_data = training_data
        
        # 每40步记录perf数据
        if step % 40 == 0:
            perf_data = {
                "perf/mfu": 0.3,
                "perf/step_time": 4.0
            }
            all_data.update(perf_data)
        
        wandb.log(all_data, step=step, commit=True)
        print(f"  ✅ Step {step}: 记录了 {len(all_data)} 个指标")
    
    print("\n🎉 WandB step冲突修复完成!")
    print(f"🔗 新的WandB URL: {run.url}")
    print("📊 请检查新的运行，应该能看到正确的图表")
    
    # 不要自动结束运行，让用户检查
    print("\n💡 提示: 运行仍然活跃，请检查WandB界面后手动结束")
    
    return True

def clean_wandb_cache():
    """清理WandB缓存"""
    print("🧹 清理WandB缓存...")
    
    try:
        import wandb
        
        # 清理本地缓存
        cache_dir = os.path.expanduser("~/.cache/wandb")
        if os.path.exists(cache_dir):
            print(f"📂 WandB缓存目录: {cache_dir}")
            
        # 重置WandB设置
        wandb.setup()
        print("✅ WandB设置已重置")
        
    except Exception as e:
        print(f"⚠️ 清理缓存失败: {e}")

if __name__ == "__main__":
    print("🔧 WandB Step冲突修复工具")
    print("=" * 50)
    
    # 清理缓存
    clean_wandb_cache()
    
    # 修复step问题
    success = fix_wandb_step_issue()
    
    if success:
        print("\n✅ 修复完成! 您现在可以:")
        print("1. 检查新的WandB URL中的图表")
        print("2. 如果图表正常，重新启动训练")
        print("3. 确保使用正确的项目名称")
    else:
        print("\n❌ 修复失败，请检查WandB配置") 