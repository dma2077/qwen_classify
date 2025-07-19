#!/usr/bin/env python3
"""
诊断WandB eval指标显示问题
"""

import os
import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def diagnose_wandb_eval_issue():
    """诊断WandB eval指标显示问题"""
    print("🔍 开始诊断WandB eval指标显示问题...")
    
    # 检查WandB是否可用
    try:
        import wandb
        print("✅ WandB已安装")
        print(f"   📦 版本: {wandb.__version__}")
    except ImportError:
        print("❌ WandB未安装")
        print("   请运行: pip install wandb")
        return
    
    # 检查WandB登录状态
    try:
        api = wandb.Api()
        print("✅ WandB API可用")
    except Exception as e:
        print(f"❌ WandB API不可用: {e}")
        print("   请运行: wandb login")
        return
    
    # 检查当前WandB运行状态
    try:
        if wandb.run is not None:
            print("✅ 当前有活跃的WandB运行")
            print(f"   📊 项目: {wandb.run.project}")
            print(f"   🏃 运行名称: {wandb.run.name}")
            print(f"   🔗 URL: {wandb.run.url}")
            print(f"   🏃 状态: {getattr(wandb.run, 'state', 'unknown')}")
            print(f"   📈 当前step: {getattr(wandb.run, 'step', 0)}")
        else:
            print("⚠️ 当前没有活跃的WandB运行")
    except Exception as e:
        print(f"⚠️ 检查WandB运行状态失败: {e}")
    
    # 检查WandB配置
    try:
        if wandb.run is not None:
            print("\n📋 WandB配置信息:")
            print(f"   📊 项目: {wandb.run.project}")
            print(f"   🏃 运行名称: {wandb.run.name}")
            print(f"   🏢 实体: {getattr(wandb.run, 'entity', 'unknown')}")
            print(f"   🆔 运行ID: {getattr(wandb.run, 'id', 'unknown')}")
            print(f"   🔗 完整URL: {wandb.run.url}")
            
            # 检查历史数据
            try:
                history = wandb.run.history()
                if not history.empty:
                    print(f"\n📊 历史数据信息:")
                    print(f"   📈 数据点数量: {len(history)}")
                    print(f"   📋 列名: {list(history.columns)}")
                    
                    # 检查eval指标
                    eval_columns = [col for col in history.columns if 'eval' in col.lower()]
                    if eval_columns:
                        print(f"   📊 Eval指标列: {eval_columns}")
                        for col in eval_columns:
                            non_null_count = history[col].notna().sum()
                            print(f"     - {col}: {non_null_count}个非空值")
                    else:
                        print("   ⚠️ 没有找到eval指标列")
                else:
                    print("   ⚠️ 历史数据为空")
            except Exception as history_error:
                print(f"   ⚠️ 获取历史数据失败: {history_error}")
    except Exception as e:
        print(f"⚠️ 检查WandB配置失败: {e}")
    
    # 提供解决方案
    print("\n💡 可能的解决方案:")
    print("1. 刷新WandB网页界面")
    print("2. 检查WandB项目设置中的图表配置")
    print("3. 确保eval指标有足够的数据点")
    print("4. 检查WandB运行是否正常同步")
    print("5. 尝试重新启动训练")

def test_wandb_logging():
    """测试WandB日志记录功能"""
    print("\n🧪 测试WandB日志记录功能...")
    
    try:
        import wandb
        
        # 创建测试运行
        wandb.init(
            project="test_eval_diagnosis",
            name=f"diagnosis_test_{int(time.time())}",
            tags=["diagnosis", "test"]
        )
        
        print("✅ 测试WandB运行创建成功")
        print(f"   🔗 URL: {wandb.run.url}")
        
        # 记录测试数据
        test_data = {
            "test/training_loss": 0.5,
            "test/eval_loss": 0.3,
            "test/eval_accuracy": 0.8
        }
        
        wandb.log(test_data, step=1, commit=True)
        print("✅ 测试数据记录成功")
        
        # 检查数据是否记录
        history = wandb.run.history()
        print(f"📊 历史数据: {len(history)}行")
        print(f"📋 列名: {list(history.columns)}")
        
        # 结束运行
        wandb.finish()
        print("✅ 测试运行结束")
        
    except Exception as e:
        print(f"❌ 测试WandB日志记录失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_wandb_eval_issue()
    test_wandb_logging() 