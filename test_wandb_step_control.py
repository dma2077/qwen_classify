#!/usr/bin/env python3
"""
测试WandB step控制修复
"""

import os
import sys
import time
import yaml
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_wandb_step_control():
    """测试WandB step控制是否正确"""
    print("🔧 测试WandB step控制修复...")
    
    try:
        import wandb
        print(f"✅ WandB版本: {wandb.__version__}")
    except ImportError:
        print("❌ WandB未安装")
        return False
    
    # 清理现有WandB状态
    if wandb.run is not None:
        print(f"🔄 结束现有运行")
        wandb.finish()
    
    # 使用修复后的TrainingMonitor
    try:
        # 加载配置
        config_file = "configs/multi_datasets_config.yaml"
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # 处理配置
        from training.utils.config_utils import prepare_config
        config = prepare_config(config)
        
        # 设置临时输出目录
        output_dir = "./test_wandb_step_control_output"
        os.makedirs(output_dir, exist_ok=True)
        config['output_dir'] = output_dir
        
        # 创建monitor
        from training.utils.monitor import TrainingMonitor
        monitor = TrainingMonitor(output_dir, config)
        
        print(f"✅ TrainingMonitor创建成功")
        
        if not monitor.use_wandb:
            print("❌ WandB未启用")
            return False
        
    except Exception as e:
        print(f"❌ Monitor创建失败: {e}")
        return False
    
    # 测试特定的step控制
    print("\n📊 测试step控制...")
    
    # 测试步骤：记录多个不连续的step
    test_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40]
    
    for i, step in enumerate(test_steps):
        print(f"\n📈 测试第{i+1}次记录，目标step: {step}")
        
        # 检查记录前的WandB状态
        if wandb.run is not None:
            pre_step = getattr(wandb.run, 'step', 0)
            print(f"   🔍 记录前WandB step: {pre_step}")
        
        # 构建测试数据
        test_data = {
            "training/loss": 2.0 - step * 0.03,
            "training/lr": 1e-5,
            "training/epoch": step * 0.05,
            "training/grad_norm": 1.0 + step * 0.02
        }
        
        # 每10步添加eval指标
        if step % 10 == 0:
            test_data.update({
                "eval/overall_loss": 1.5 - step * 0.02,
                "eval/overall_accuracy": 0.5 + step * 0.01,
                "eval/overall_samples": 1000,
                "eval/overall_correct": int(1000 * (0.5 + step * 0.01))
            })
        
        # 每15步添加性能指标
        if step % 15 == 0:
            test_data.update({
                "perf/step_time": 3.0 + step * 0.1,
                "perf/mfu": 0.3 + step * 0.005,
                "perf/tokens_per_second": 1000 + step * 20
            })
        
        # 记录数据
        try:
            print(f"   📊 开始记录数据...")
            monitor.log_metrics(test_data, step=step, commit=True)
            
            # 检查记录后的WandB状态
            if wandb.run is not None:
                post_step = getattr(wandb.run, 'step', 0)
                print(f"   🔍 记录后WandB step: {post_step}")
                
                # 验证step是否正确
                if post_step == step:
                    print(f"   ✅ Step控制成功: {step}")
                    success_count = i + 1
                else:
                    print(f"   ❌ Step控制失败: 期望{step}, 实际{post_step}")
                    break
            
            # 短暂延迟
            time.sleep(0.2)
            
        except Exception as e:
            print(f"   ❌ 记录失败: {e}")
            break
    
    # 最终验证
    print(f"\n🔍 最终验证...")
    
    if wandb.run is not None:
        final_step = getattr(wandb.run, 'step', 0)
        expected_final_step = max(test_steps)
        
        print(f"✅ 最终WandB step: {final_step}")
        print(f"🔗 WandB URL: {wandb.run.url}")
        
        # 分析结果
        if final_step == expected_final_step:
            print(f"🎉 Step控制完全成功！")
            print(f"   ✅ 所有{len(test_steps)}个step都正确记录")
            success = True
        else:
            print(f"⚠️ Step控制部分成功")
            print(f"   期望最终step: {expected_final_step}")
            print(f"   实际最终step: {final_step}")
            success = False
        
        # 检查summary数据
        try:
            if hasattr(wandb.run, 'summary') and wandb.run.summary:
                summary_keys = list(wandb.run.summary.keys())
                print(f"📋 WandB summary: {len(summary_keys)}个指标")
                
                # 按类型分组
                training_keys = [k for k in summary_keys if k.startswith('training/')]
                eval_keys = [k for k in summary_keys if k.startswith('eval/')]
                perf_keys = [k for k in summary_keys if k.startswith('perf/')]
                
                print(f"   📈 Training指标: {len(training_keys)}")
                print(f"   📊 Eval指标: {len(eval_keys)}")
                print(f"   ⚡ Perf指标: {len(perf_keys)}")
            else:
                print("⚠️ WandB summary为空")
        except Exception as e:
            print(f"❌ 检查summary失败: {e}")
        
        print(f"\n📊 请检查WandB界面: {wandb.run.url}")
        print("🔍 验证要点:")
        print("1. Training图表的x轴应该显示: 1, 5, 10, 15, 20, 25, 30, 35, 40")
        print("2. Eval图表的x轴应该显示: 10, 20, 30, 40") 
        print("3. Perf图表的x轴应该显示: 15, 30")
        print("4. 所有图表都不应该显示step=0或其他错误的step值")
        
        return success, wandb.run.url
    else:
        print("❌ WandB运行不存在")
        return False, None

if __name__ == "__main__":
    print("🔧 WandB Step控制测试工具")
    print("=" * 50)
    
    success, url = test_wandb_step_control()
    
    if success:
        print(f"\n🎉 Step控制修复成功!")
        print(f"🔗 查看结果: {url}")
        print("\n💡 现在您可以重新开始训练，应该能看到正确的step进展。")
        print("💡 training指标应该在正确的effective_step显示，不再是step=0。")
    else:
        print(f"\n❌ Step控制修复失败")
        if url:
            print(f"🔗 查看失败结果: {url}")
        print("💡 可能需要进一步调试WandB step机制。") 