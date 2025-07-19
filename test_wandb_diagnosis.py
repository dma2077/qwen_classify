#!/usr/bin/env python3
"""
WandB诊断脚本 - 检查数据同步问题
"""

import os
import sys
import time
import yaml
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_wandb_sync():
    """测试WandB数据同步"""
    print("🔍 开始WandB诊断...")
    
    try:
        import wandb
        print("✅ WandB已安装")
    except ImportError:
        print("❌ WandB未安装，请运行: pip install wandb")
        return
    
    # 创建测试配置
    test_config = {
        'wandb': {
            'enabled': True,
            'project': 'wandb_diagnosis_test',
            'run_name': 'sync_test'
        }
    }
    
    try:
        # 初始化WandB
        wandb.init(
            project=test_config['wandb']['project'],
            name=test_config['wandb']['run_name'],
            config=test_config
        )
        print("✅ WandB初始化成功")
        print(f"🔗 WandB URL: {wandb.run.url}")
        print(f"📊 项目: {wandb.run.project}")
        print(f"🏃 状态: {getattr(wandb.run, 'state', 'unknown')}")
        
        # 定义指标
        wandb.define_metric("step")
        wandb.define_metric("training/loss", step_metric="step", summary="min")
        wandb.define_metric("training/lr", step_metric="step", summary="last")
        print("✅ 指标定义成功")
        
        # 测试数据记录
        for step in range(1, 6):
            data = {
                "training/loss": 1.0 - step * 0.1,
                "training/lr": 1e-4,
                "step": step
            }
            
            print(f"\n📊 记录step {step}的数据...")
            wandb.log(data, step=step, commit=True)
            
            # 强制同步
            try:
                wandb.run.sync()
                print(f"  ✅ 数据已同步到云端")
            except Exception as sync_error:
                print(f"  ⚠️  同步失败: {sync_error}")
            
            # 检查WandB状态
            current_step = getattr(wandb.run, 'step', 0)
            print(f"  🔍 WandB当前step: {current_step}")
            
            time.sleep(1)  # 等待1秒
        
        # 最终同步
        print("\n🔄 进行最终同步...")
        wandb.run.sync()
        
        print("\n🎉 诊断完成！")
        print("📊 请检查WandB界面:")
        print(f"   🔗 {wandb.run.url}")
        print("   应该能看到:")
        print("   • training/loss 图表")
        print("   • training/lr 图表")
        print("   • 5个数据点 (step 1-5)")
        
        # 询问用户是否看到数据
        print("\n❓ 请在WandB界面上检查是否能看到数据，然后告诉我结果")
        
    except Exception as e:
        print(f"❌ WandB诊断失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            wandb.finish()
            print("✅ WandB已关闭")
        except:
            pass

def check_wandb_environment():
    """检查WandB环境"""
    print("🔍 检查WandB环境...")
    
    # 检查环境变量
    env_vars = ['WANDB_API_KEY', 'WANDB_PROJECT', 'WANDB_ENTITY']
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            print(f"  ✅ {var}: {value[:10]}..." if len(value) > 10 else f"  ✅ {var}: {value}")
        else:
            print(f"  ⚠️  {var}: 未设置")
    
    # 检查WandB配置
    try:
        import wandb
        print(f"  📦 WandB版本: {wandb.__version__}")
    except:
        print("  ❌ 无法获取WandB版本")

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 WandB诊断工具")
    print("=" * 60)
    
    check_wandb_environment()
    print()
    test_wandb_sync() 