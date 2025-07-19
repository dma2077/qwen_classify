#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试WandB step一致性
"""

import sys
import os
import time
import tempfile
import shutil

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_wandb_step_consistency():
    """测试WandB step一致性"""
    print("🧪 测试WandB step一致性...")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"📁 临时目录: {temp_dir}")
    
    try:
        from training.utils.monitor import TrainingMonitor
        
        # 创建配置
        config = {
            'wandb': {
                'enabled': True,
                'project': 'test_wandb_step_consistency',
                'run_name': 'test_step_consistency'
            },
            'output_dir': temp_dir
        }
        
        # 创建monitor
        monitor = TrainingMonitor(temp_dir, config)
        
        print("✅ TrainingMonitor创建成功")
        
        # 测试连续记录，检查step是否一致
        print("\n📊 测试连续记录...")
        for step in range(1, 11):
            training_data = {
                "training/loss": 0.5 - step * 0.01,
                "training/lr": 1e-4,
                "training/epoch": 0.1 * step,
                "training/grad_norm": 1.0 + step * 0.1
            }
            
            try:
                monitor.log_metrics(training_data, step=step, commit=True)
                print(f"  ✅ Step {step}: 记录成功")
                
                # 检查WandB的当前step
                import wandb
                if wandb.run is not None:
                    current_wandb_step = getattr(wandb.run, 'step', 0)
                    print(f"     📊 WandB当前step: {current_wandb_step}")
                    
                    # 检查step是否一致
                    if current_wandb_step == step:
                        print(f"     ✅ Step一致")
                    else:
                        print(f"     ⚠️  Step不一致: 期望{step}, 实际{current_wandb_step}")
                
                time.sleep(0.1)  # 短暂延迟
                
            except Exception as e:
                print(f"  ❌ Step {step}: 记录失败 - {e}")
        
        print("\n📊 测试eval指标记录...")
        eval_data = {
            "eval/overall_loss": 0.3,
            "eval/overall_accuracy": 0.85,
            "eval/overall_samples": 1000,
            "eval/overall_correct": 850
        }
        
        try:
            monitor.log_metrics(eval_data, step=10, commit=True)
            print("  ✅ eval指标记录成功")
            
            # 检查WandB的当前step
            import wandb
            if wandb.run is not None:
                current_wandb_step = getattr(wandb.run, 'step', 0)
                print(f"     📊 WandB当前step: {current_wandb_step}")
                
                if current_wandb_step == 10:
                    print(f"     ✅ Step一致")
                else:
                    print(f"     ⚠️  Step不一致: 期望10, 实际{current_wandb_step}")
                    
        except Exception as e:
            print(f"  ❌ eval指标记录失败 - {e}")
        
        print("\n✅ WandB step一致性测试完成")
        
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
    test_wandb_step_consistency() 