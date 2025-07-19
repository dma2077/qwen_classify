#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试WandB修复
"""

import sys
import os
import tempfile
import shutil

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_wandb_quick_fix():
    """快速测试WandB修复"""
    print("🧪 快速测试WandB修复...")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"📁 临时目录: {temp_dir}")
    
    try:
        from training.utils.monitor import TrainingMonitor
        
        # 创建配置
        config = {
            'wandb': {
                'enabled': True,
                'project': 'test_wandb_quick_fix',
                'run_name': 'test_quick_run'
            },
            'output_dir': temp_dir
        }
        
        # 创建monitor
        monitor = TrainingMonitor(temp_dir, config)
        
        print("✅ TrainingMonitor创建成功")
        
        # 测试_is_main_process方法
        is_main = monitor._is_main_process()
        print(f"✅ _is_main_process() 返回: {is_main}")
        
        # 测试简单的指标记录
        if monitor.use_wandb:
            print("✅ WandB已启用")
            
            # 测试记录一个简单的指标
            test_data = {
                "training/test_loss": 0.5,
                "perf/test_mfu": 0.8
            }
            
            try:
                monitor.log_metrics(test_data, step=1, commit=True)
                print("✅ 指标记录成功")
            except Exception as e:
                print(f"❌ 指标记录失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("⚠️  WandB未启用")
        
        print("\n✅ 快速测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理
        try:
            if 'monitor' in locals() and hasattr(monitor, 'use_wandb') and monitor.use_wandb:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
                    print("  ✅ WandB运行已结束")
        except Exception as e:
            print(f"  ⚠️  清理WandB失败: {e}")
        
        # 清理临时目录
        try:
            shutil.rmtree(temp_dir)
            print(f"  ✅ 临时目录已清理: {temp_dir}")
        except Exception as e:
            print(f"  ⚠️  清理临时目录失败: {e}")

if __name__ == "__main__":
    test_wandb_quick_fix() 