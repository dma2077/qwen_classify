#!/usr/bin/env python3
"""
测试WandB图表显示修复
验证training和perf指标是否在正确的step显示
"""

import os
import sys
import time
import yaml
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_wandb_chart_display():
    """测试WandB图表显示是否正确"""
    print("🔧 测试WandB图表显示修复...")
    
    try:
        import wandb
        print(f"✅ WandB版本: {wandb.__version__}")
    except ImportError:
        print("❌ WandB未安装")
        return False
    
    # 完全清理WandB状态
    if wandb.run is not None:
        print(f"🔄 结束现有运行")
        wandb.finish()
    
    try:
        # 重置WandB
        wandb.setup(_reset=True)
        print("✅ WandB状态已重置")
    except Exception as e:
        print(f"⚠️ WandB重置失败: {e}")
    
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
        output_dir = "./test_wandb_chart_display_output"
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
        import traceback
        traceback.print_exc()
        return False
    
    # 检查WandB运行状态
    try:
        if wandb.run is not None:
            print(f"✅ WandB运行活跃: {wandb.run.name}")
            print(f"🔗 WandB URL: {wandb.run.url}")
            print(f"📊 WandB项目: {wandb.run.project}")
        else:
            print("❌ WandB运行未创建")
            return False
    except Exception as e:
        print(f"❌ 检查WandB状态失败: {e}")
        return False
    
    # 测试训练指标的step显示
    print("\n📊 测试训练指标step显示...")
    
    training_steps = [1, 3, 5, 7, 10, 12, 15, 18, 20, 25, 30]
    
    for step in training_steps:
        print(f"\n📈 记录训练指标 - step {step}")
        
        # 构建训练数据
        training_data = {
            "training/loss": 3.0 - step * 0.08,
            "training/lr": 1e-5 * (1 + step * 0.1),
            "training/epoch": step * 0.03,
            "training/grad_norm": 1.2 + step * 0.05
        }
        
        # 每5步添加性能指标
        if step % 5 == 0:
            training_data.update({
                "perf/step_time": 2.5 + step * 0.2,
                "perf/steps_per_second": 1.0 / (2.5 + step * 0.2),
                "perf/mfu": 0.25 + step * 0.01,
                "perf/mfu_percent": (0.25 + step * 0.01) * 100,
                "perf/tokens_per_second": 1500 + step * 30,
                "perf/samples_per_second": 8 + step * 0.2
            })
            print(f"   ⚡ 包含性能指标")
        
        # 每10步添加eval指标
        if step % 10 == 0:
            training_data.update({
                "eval/overall_loss": 2.0 - step * 0.05,
                "eval/overall_accuracy": 0.4 + step * 0.02,
                "eval/overall_samples": 2000,
                "eval/overall_correct": int(2000 * (0.4 + step * 0.02))
            })
            print(f"   📊 包含评估指标")
        
        # 记录数据
        try:
            print(f"   🔧 记录数据到WandB (step={step})...")
            monitor.log_metrics(training_data, step=step, commit=True)
            
            # 验证WandB状态
            if wandb.run is not None:
                wandb_step = getattr(wandb.run, 'step', 0)
                print(f"   🔍 WandB内部step: {wandb_step}")
                
                if wandb_step == step:
                    print(f"   ✅ Step同步正确: {step}")
                else:
                    print(f"   ⚠️ Step不同步: 期望{step}, WandB{wandb_step}")
            
            # 等待数据同步
            time.sleep(0.3)
            
        except Exception as e:
            print(f"   ❌ 记录失败: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # 等待数据完全同步
    print(f"\n⏱️ 等待数据同步...")
    time.sleep(3)
    
    # 最终验证
    print(f"\n🔍 最终验证...")
    
    if wandb.run is not None:
        final_step = getattr(wandb.run, 'step', 0)
        expected_final = max(training_steps)
        
        print(f"✅ 最终WandB step: {final_step}")
        print(f"🔗 WandB URL: {wandb.run.url}")
        
        # 检查数据记录
        try:
            if hasattr(wandb.run, 'summary') and wandb.run.summary:
                summary = wandb.run.summary
                summary_keys = list(summary.keys())
                
                print(f"📋 WandB summary: {len(summary_keys)}个指标")
                
                # 按类型分组统计
                training_keys = [k for k in summary_keys if k.startswith('training/')]
                eval_keys = [k for k in summary_keys if k.startswith('eval/')]
                perf_keys = [k for k in summary_keys if k.startswith('perf/')]
                
                print(f"   📈 Training指标: {len(training_keys)} - {training_keys}")
                print(f"   📊 Eval指标: {len(eval_keys)} - {eval_keys}")
                print(f"   ⚡ Perf指标: {len(perf_keys)} - {perf_keys}")
                
                # 显示关键指标的当前值
                key_metrics = ['training/loss', 'training/lr', 'eval/overall_accuracy', 'perf/mfu']
                for metric in key_metrics:
                    if metric in summary:
                        value = summary[metric]
                        print(f"   {metric}: {value}")
                
            else:
                print("⚠️ WandB summary为空")
                
        except Exception as e:
            print(f"❌ 检查summary失败: {e}")
        
        # 分析结果
        success = (final_step == expected_final)
        
        print(f"\n📊 测试结果分析:")
        if success:
            print(f"🎉 图表显示修复成功!")
            print(f"   ✅ 所有step都正确记录到WandB")
        else:
            print(f"⚠️ 可能存在step同步问题")
            print(f"   期望最终step: {expected_final}")
            print(f"   实际最终step: {final_step}")
        
        print(f"\n🔗 请立即检查WandB URL: {wandb.run.url}")
        print("🔍 重点验证:")
        print("1. 📈 Training组图表:")
        print(f"   - training/loss 图表的x轴应显示: {training_steps}")
        print(f"   - training/lr 图表的x轴应显示: {training_steps}")
        print(f"   - 所有training指标都不应该停留在step=0")
        
        print("2. ⚡ Perf组图表:")
        perf_steps = [s for s in training_steps if s % 5 == 0]
        print(f"   - perf/mfu 图表的x轴应显示: {perf_steps}")
        print(f"   - perf/step_time 图表的x轴应显示: {perf_steps}")
        print(f"   - 所有perf指标都不应该停留在step=0")
        
        print("3. 📊 Eval组图表:")
        eval_steps = [s for s in training_steps if s % 10 == 0]
        print(f"   - eval/overall_accuracy 图表的x轴应显示: {eval_steps}")
        print(f"   - eval/overall_loss 图表的x轴应显示: {eval_steps}")
        
        print("\n💡 如果图表仍显示step=0:")
        print("1. 刷新WandB页面")
        print("2. 检查图表的x轴设置")
        print("3. 可能需要等待几分钟让数据完全同步")
        
        return success, wandb.run.url
    else:
        print("❌ WandB运行不存在")
        return False, None

if __name__ == "__main__":
    print("🔧 WandB图表显示测试工具")
    print("=" * 60)
    
    success, url = test_wandb_chart_display()
    
    if success:
        print(f"\n🎉 图表显示修复成功!")
        print(f"🔗 查看结果: {url}")
        print("\n💡 现在您的训练过程中:")
        print("💡 • Training指标应该在正确的effective_step显示")
        print("💡 • Perf指标应该在正确的effective_step显示")
        print("💡 • 不再停留在step=0")
    else:
        print(f"\n❌ 图表显示修复失败")
        if url:
            print(f"🔗 查看结果: {url}")
        print("💡 可能需要进一步调试WandB图表机制") 