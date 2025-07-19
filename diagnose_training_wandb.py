#!/usr/bin/env python3
"""
诊断训练代码中的WandB配置和eval图表显示问题
"""

import os
import sys
import time
import yaml
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def diagnose_training_wandb():
    """诊断训练代码中的WandB配置"""
    print("🔍 诊断训练代码中的WandB配置...")
    
    try:
        import wandb
        print(f"✅ WandB版本: {wandb.__version__}")
    except ImportError:
        print("❌ WandB未安装")
        return False
    
    # 1. 测试配置加载
    print("\n📋 测试配置加载...")
    try:
        # 加载您的配置文件
        config_file = "configs/multi_datasets_config.yaml"
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ 配置文件加载成功: {config_file}")
        
        # 检查WandB配置
        wandb_config = config.get('wandb', {})
        print(f"📊 WandB配置:")
        print(f"   enabled: {wandb_config.get('enabled', False)}")
        print(f"   project: {wandb_config.get('project', 'N/A')}")
        print(f"   run_name: {wandb_config.get('run_name', 'N/A')}")
        
        if not wandb_config.get('enabled', False):
            print("❌ WandB在配置中被禁用！")
            return False
            
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False
    
    # 2. 测试TrainingMonitor初始化
    print("\n🔧 测试TrainingMonitor初始化...")
    try:
        from training.utils.config_utils import prepare_config
        from training.utils.monitor import TrainingMonitor
        
        # 处理配置
        config = prepare_config(config)
        
        # 创建临时输出目录
        output_dir = "./test_training_wandb_output"
        os.makedirs(output_dir, exist_ok=True)
        config['output_dir'] = output_dir
        
        # 创建monitor
        monitor = TrainingMonitor(output_dir, config)
        
        print(f"✅ TrainingMonitor创建成功")
        print(f"   use_wandb: {monitor.use_wandb}")
        print(f"   _is_main_process(): {monitor._is_main_process()}")
        
        if not monitor.use_wandb:
            print("❌ TrainingMonitor中WandB未启用！")
            return False
            
    except Exception as e:
        print(f"❌ TrainingMonitor初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 检查WandB运行状态
    print("\n🔍 检查WandB运行状态...")
    try:
        if wandb.run is not None:
            print(f"✅ WandB运行已活跃")
            print(f"   项目: {wandb.run.project}")
            print(f"   运行名称: {wandb.run.name}")
            print(f"   URL: {wandb.run.url}")
            print(f"   状态: {getattr(wandb.run, 'state', 'unknown')}")
            print(f"   当前step: {getattr(wandb.run, 'step', 0)}")
        else:
            print("⚠️ 没有活跃的WandB运行")
            
    except Exception as e:
        print(f"❌ 检查WandB运行状态失败: {e}")
    
    # 4. 测试指标记录（模拟训练过程）
    print("\n📊 测试指标记录...")
    try:
        # 测试training指标
        for step in [20, 40, 60]:
            training_data = {
                "training/loss": 2.0 - step * 0.01,
                "training/lr": 1e-5,
                "training/epoch": step * 0.01,
                "training/grad_norm": 1.5
            }
            
            # 每20步添加性能指标
            if step % 20 == 0:
                training_data.update({
                    "perf/step_time": 4.2,
                    "perf/steps_per_second": 1.0 / 4.2,
                    "perf/mfu": 0.35
                })
            
            print(f"   📈 记录training指标 (step={step})...")
            monitor.log_metrics(training_data, step=step, commit=True)
            
            # 验证WandB状态
            if wandb.run is not None:
                current_step = getattr(wandb.run, 'step', 0)
                print(f"     WandB当前step: {current_step}")
                
        # 测试eval指标（模拟评估）
        for eval_step in [40, 60]:
            eval_data = {
                "eval/overall_loss": 1.5 - eval_step * 0.005,
                "eval/overall_accuracy": 0.3 + eval_step * 0.003,
                "eval/overall_samples": 1000,
                "eval/overall_correct": int(1000 * (0.3 + eval_step * 0.003)),
                "eval/food101_loss": 1.5 - eval_step * 0.005,
                "eval/food101_accuracy": 0.3 + eval_step * 0.003,
                "eval/food101_samples": 500
            }
            
            print(f"   📊 记录eval指标 (step={eval_step})...")
            monitor.log_metrics(eval_data, step=eval_step, commit=True)
            
            # 验证eval指标
            eval_metrics_list = [k for k in eval_data.keys() if k.startswith('eval/')]
            print(f"     记录的eval指标: {eval_metrics_list}")
            
        print("✅ 指标记录测试完成")
        
    except Exception as e:
        print(f"❌ 指标记录测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. 检查WandB summary和历史数据
    print("\n🔍 检查WandB数据...")
    try:
        if wandb.run is not None:
            # 等待数据同步
            time.sleep(2)
            
            # 检查summary
            if hasattr(wandb.run, 'summary') and wandb.run.summary:
                summary_keys = list(wandb.run.summary.keys())
                print(f"✅ WandB summary包含 {len(summary_keys)} 个指标")
                
                # 分类显示指标
                training_summary = [k for k in summary_keys if k.startswith('training/')]
                eval_summary = [k for k in summary_keys if k.startswith('eval/')]
                perf_summary = [k for k in summary_keys if k.startswith('perf/')]
                
                print(f"   📈 Training指标: {training_summary}")
                print(f"   📊 Eval指标: {eval_summary}")
                print(f"   ⚡ Perf指标: {perf_summary}")
                
                # 检查关键指标的值
                for key in ['training/loss', 'eval/overall_accuracy', 'perf/step_time']:
                    if key in wandb.run.summary:
                        value = wandb.run.summary[key]
                        print(f"   {key}: {value}")
                    else:
                        print(f"   ❌ {key}: 未找到")
                        
            else:
                print("❌ WandB summary为空或不可用")
                
    except Exception as e:
        print(f"❌ 检查WandB数据失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. 输出诊断结果
    print("\n📋 诊断结果:")
    try:
        if wandb.run is not None:
            print(f"✅ WandB运行正常")
            print(f"🔗 请检查WandB URL: {wandb.run.url}")
            print(f"📊 项目: {wandb.run.project}")
            
            print("\n📊 期望在WandB界面看到:")
            print("1. Training组图表:")
            print("   - training/loss: 应该显示step 20, 40, 60的数据")
            print("   - training/lr: 应该显示恒定值1e-5")
            print("   - training/epoch: 应该递增")
            print("   - training/grad_norm: 应该显示恒定值1.5")
            
            print("2. Eval组图表:")
            print("   - eval/overall_loss: 应该显示step 40, 60的数据")
            print("   - eval/overall_accuracy: 应该显示递增趋势")
            print("   - eval/overall_samples: 应该显示恒定值1000")
            print("   - eval/food101_*: 应该显示数据集特定指标")
            
            print("3. Perf组图表:")
            print("   - perf/step_time: 应该显示step 20, 40, 60的数据")
            print("   - perf/mfu: 应该显示恒定值0.35")
            
            print("\n🔧 如果图表仍然不正常:")
            print("1. 检查是否有多个WandB项目或运行")
            print("2. 尝试刷新WandB页面")
            print("3. 检查WandB账户权限")
            print("4. 考虑使用新的项目名称重新开始训练")
            
            return True, wandb.run.url
        else:
            print("❌ WandB运行未正常创建")
            return False, None
            
    except Exception as e:
        print(f"❌ 输出诊断结果失败: {e}")
        return False, None

def test_eval_chart_visibility():
    """专门测试eval图表可见性"""
    print("\n🧪 专门测试eval图表可见性...")
    
    try:
        import wandb
        
        # 如果已有运行，继续使用，否则创建新的
        if wandb.run is None:
            run = wandb.init(
                project="eval_chart_test",
                name=f"eval_test_{int(time.time())}",
                tags=["eval_test", "chart_visibility"]
            )
        else:
            run = wandb.run
        
        print(f"🔧 使用WandB运行: {run.name}")
        
        # 强制定义eval指标
        wandb.define_metric("step")
        wandb.define_metric("eval/*", step_metric="step")
        
        # 记录多个eval数据点，确保图表可见
        eval_steps = [100, 200, 300, 400, 500]
        
        for step in eval_steps:
            eval_data = {
                "eval/overall_loss": 2.0 - step * 0.002,
                "eval/overall_accuracy": 0.5 + step * 0.0008,
                "eval/overall_samples": 1000,
                "eval/overall_correct": int(1000 * (0.5 + step * 0.0008)),
                "eval/test_dataset_loss": 2.0 - step * 0.002,
                "eval/test_dataset_accuracy": 0.5 + step * 0.0008
            }
            
            # 使用step参数确保数据在正确的位置
            wandb.log(eval_data, step=step, commit=True)
            print(f"   📊 Step {step}: eval数据已记录")
            
            time.sleep(0.1)  # 短暂延迟
        
        print(f"✅ eval图表测试完成")
        print(f"🔗 检查eval图表: {run.url}")
        print("📊 应该能在WandB界面看到eval组的所有指标图表")
        
        return True, run.url
        
    except Exception as e:
        print(f"❌ eval图表测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    print("🔍 训练WandB诊断工具")
    print("=" * 60)
    
    # 主要诊断
    success, url = diagnose_training_wandb()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ 训练WandB诊断成功!")
        
        # 额外的eval图表测试
        eval_success, eval_url = test_eval_chart_visibility()
        
        if eval_success:
            print(f"\n🎉 所有测试完成!")
            print(f"🔗 主要测试URL: {url}")
            print(f"🔗 Eval图表测试URL: {eval_url}")
        else:
            print(f"\n⚠️ Eval图表测试失败，但主要诊断成功")
            print(f"🔗 主要测试URL: {url}")
    else:
        print("\n❌ 训练WandB诊断失败")
        print("请检查配置和WandB设置") 