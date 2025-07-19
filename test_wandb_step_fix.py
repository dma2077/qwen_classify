#!/usr/bin/env python3
"""
测试WandB step修复是否有效
"""

import os
import sys
import time
import yaml
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_wandb_step_fix():
    """测试WandB step修复是否有效"""
    print("🔧 测试WandB step修复...")
    
    try:
        import wandb
        print(f"✅ WandB版本: {wandb.__version__}")
    except ImportError:
        print("❌ WandB未安装")
        return False
    
    # 1. 完全清理现有WandB状态
    print("\n🧹 清理WandB状态...")
    if wandb.run is not None:
        print(f"🔄 结束现有运行: {wandb.run.name}")
        wandb.finish()
    
    # 重置WandB状态
    try:
        wandb.setup(_reset=True)
        print("✅ WandB状态已重置")
    except Exception as e:
        print(f"⚠️ WandB重置失败: {e}")
    
    # 2. 使用修复后的TrainingMonitor
    print("\n🔧 测试修复后的TrainingMonitor...")
    try:
        # 加载配置
        config_file = "configs/multi_datasets_config.yaml"
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # 处理配置
        from training.utils.config_utils import prepare_config
        config = prepare_config(config)
        
        # 设置临时输出目录
        output_dir = "./test_wandb_step_fix_output"
        os.makedirs(output_dir, exist_ok=True)
        config['output_dir'] = output_dir
        
        # 创建monitor
        from training.utils.monitor import TrainingMonitor
        monitor = TrainingMonitor(output_dir, config)
        
        print(f"✅ TrainingMonitor创建成功")
        print(f"   use_wandb: {monitor.use_wandb}")
        
        if not monitor.use_wandb:
            print("❌ WandB未启用，无法测试")
            return False
        
    except Exception as e:
        print(f"❌ TrainingMonitor创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 测试step记录（关键测试）
    print("\n📊 测试step记录...")
    
    # 检查初始WandB状态
    if wandb.run is not None:
        initial_step = getattr(wandb.run, 'step', 0)
        print(f"🔍 初始WandB step: {initial_step}")
    
    # 测试多个step的记录
    test_steps = [1, 5, 10, 15, 20, 25, 30]
    
    for step in test_steps:
        print(f"\n📈 测试step {step}...")
        
        # 构建测试数据
        test_data = {
            "training/loss": 2.0 - step * 0.05,
            "training/lr": 1e-5,
            "training/epoch": step * 0.1,
            "training/grad_norm": 1.2
        }
        
        # 每5步添加eval指标
        if step % 5 == 0:
            test_data.update({
                "eval/overall_loss": 1.5 - step * 0.03,
                "eval/overall_accuracy": 0.4 + step * 0.02,
                "eval/overall_samples": 1000,
                "eval/overall_correct": int(1000 * (0.4 + step * 0.02))
            })
        
        # 每10步添加性能指标
        if step % 10 == 0:
            test_data.update({
                "perf/step_time": 3.5 + step * 0.1,
                "perf/mfu": 0.25 + step * 0.01,
                "perf/tokens_per_second": 800 + step * 10
            })
        
        # 记录前的WandB状态
        pre_step = getattr(wandb.run, 'step', 0) if wandb.run else 0
        print(f"   🔍 记录前WandB step: {pre_step}")
        
        # 使用修复后的log_metrics
        try:
            monitor.log_metrics(test_data, step=step, commit=True)
            print(f"   ✅ Step {step} 记录成功")
        except Exception as e:
            print(f"   ❌ Step {step} 记录失败: {e}")
            continue
        
        # 记录后的WandB状态
        post_step = getattr(wandb.run, 'step', 0) if wandb.run else 0
        print(f"   🔍 记录后WandB step: {post_step}")
        
        # 验证step是否正确
        if post_step == step:
            print(f"   ✅ Step验证成功: {step}")
        elif post_step > pre_step:
            print(f"   ⚠️ Step发生变化: {pre_step} → {post_step} (期望: {step})")
        else:
            print(f"   ❌ Step未更新: {post_step} (期望: {step})")
        
        # 短暂延迟
        time.sleep(0.2)
    
    # 4. 检查最终WandB状态
    print("\n🔍 检查最终WandB状态...")
    
    if wandb.run is not None:
        final_step = getattr(wandb.run, 'step', 0)
        print(f"✅ 最终WandB step: {final_step}")
        print(f"🔗 WandB URL: {wandb.run.url}")
        
        # 检查summary数据
        try:
            if hasattr(wandb.run, 'summary') and wandb.run.summary:
                summary_keys = list(wandb.run.summary.keys())
                print(f"📋 WandB summary: {len(summary_keys)}个指标")
                
                # 按类型分组显示
                training_keys = [k for k in summary_keys if k.startswith('training/')]
                eval_keys = [k for k in summary_keys if k.startswith('eval/')]
                perf_keys = [k for k in summary_keys if k.startswith('perf/')]
                
                print(f"   📈 Training指标: {len(training_keys)} - {training_keys}")
                print(f"   📊 Eval指标: {len(eval_keys)} - {eval_keys}")
                print(f"   ⚡ Perf指标: {len(perf_keys)} - {perf_keys}")
                
                # 显示一些关键指标的值
                for key in ['training/loss', 'eval/overall_accuracy', 'perf/step_time']:
                    if key in wandb.run.summary:
                        value = wandb.run.summary[key]
                        print(f"   {key}: {value}")
            else:
                print("⚠️ WandB summary为空")
                
        except Exception as e:
            print(f"❌ 检查summary失败: {e}")
    
    # 5. 输出测试结果
    print("\n📋 测试结果分析:")
    
    if wandb.run is not None:
        final_step = getattr(wandb.run, 'step', 0)
        expected_final_step = max(test_steps)
        
        print(f"🔍 Step测试结果:")
        print(f"   期望最终step: {expected_final_step}")
        print(f"   实际最终step: {final_step}")
        
        if final_step == expected_final_step:
            print("   ✅ Step记录完全正确！")
            success = True
        elif final_step > 0 and final_step != expected_final_step:
            print("   ⚠️ Step记录部分正确，但数值不完全匹配")
            success = True
        elif final_step == 0:
            print("   ❌ Step仍然固定为0，修复失败")
            success = False
        else:
            print("   ⚠️ Step状态不明确")
            success = False
        
        print(f"\n🔗 请检查WandB界面: {wandb.run.url}")
        print("📊 期望看到:")
        print("   1. Training图表应该显示step 1,5,10,15,20,25,30的数据")
        print("   2. Eval图表应该显示step 5,10,15,20,25,30的数据")
        print("   3. Perf图表应该显示step 10,20,30的数据")
        print("   4. 所有图表的x轴应该显示正确的step值，而不是全部为0")
        
        return success, wandb.run.url
    else:
        print("❌ WandB运行不存在，测试失败")
        return False, None

if __name__ == "__main__":
    print("🔧 WandB Step修复测试工具")
    print("=" * 50)
    
    success, url = test_wandb_step_fix()
    
    if success:
        print(f"\n🎉 Step修复测试成功!")
        print(f"🔗 查看结果: {url}")
        print("\n💡 如果WandB图表现在显示正确的step，说明修复成功！")
        print("💡 您可以重新开始训练，应该能看到正确的step进展。")
    else:
        print(f"\n❌ Step修复测试失败")
        print("💡 可能需要进一步检查WandB配置或网络连接。") 