#!/usr/bin/env python3
"""
紧急修复WandB step显示问题
"""

import os
import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def emergency_fix_wandb_step():
    """紧急修复WandB step问题"""
    print("🚨 紧急修复WandB step显示问题...")
    
    try:
        import wandb
        print(f"✅ WandB版本: {wandb.__version__}")
    except ImportError:
        print("❌ WandB未安装")
        return False
    
    # 1. 强制结束所有现有运行
    print("🔄 检查并结束现有WandB运行...")
    if wandb.run is not None:
        print(f"⚠️ 发现现有运行: {wandb.run.name}")
        print(f"   URL: {wandb.run.url}")
        wandb.finish()
        print("✅ 现有运行已结束")
    
    # 2. 清理WandB状态
    print("🧹 清理WandB状态...")
    try:
        # 重置wandb状态
        wandb.setup(_reset=True)
        print("✅ WandB状态已重置")
    except Exception as e:
        print(f"⚠️ 状态重置失败: {e}")
    
    # 3. 创建全新的运行
    print("🚀 创建全新的WandB运行...")
    
    # 使用新的项目名称避免冲突
    project_name = f"qwen_classification_emergency_fix_{int(time.time())}"
    run_name = f"emergency_fix_run_{int(time.time())}"
    
    run = wandb.init(
        project=project_name,
        name=run_name,
        tags=["emergency_fix", "step_fix", "clean_start"],
        notes="紧急修复step显示问题的全新运行",
        reinit=True  # 强制重新初始化
    )
    
    print(f"✅ 新运行创建成功")
    print(f"   📊 项目: {project_name}")
    print(f"   🏃 运行名称: {run_name}")
    print(f"   🔗 URL: {run.url}")
    print(f"   🆔 Run ID: {run.id}")
    
    # 4. 正确定义指标
    print("🔧 定义指标...")
    
    # 重要：使用step作为全局step变量
    wandb.define_metric("global_step")
    
    # 使用global_step作为x轴
    wandb.define_metric("training/loss", step_metric="global_step")
    wandb.define_metric("training/lr", step_metric="global_step")
    wandb.define_metric("training/epoch", step_metric="global_step")
    wandb.define_metric("training/grad_norm", step_metric="global_step")
    
    wandb.define_metric("eval/overall_loss", step_metric="global_step")
    wandb.define_metric("eval/overall_accuracy", step_metric="global_step")
    wandb.define_metric("eval/overall_samples", step_metric="global_step")
    wandb.define_metric("eval/overall_correct", step_metric="global_step")
    
    wandb.define_metric("perf/step_time", step_metric="global_step")
    wandb.define_metric("perf/mfu", step_metric="global_step")
    wandb.define_metric("perf/tokens_per_second", step_metric="global_step")
    
    print("✅ 指标定义完成")
    
    # 5. 测试正确的数据记录（从step=1开始，绝不使用step=0）
    print("🧪 测试数据记录...")
    
    test_steps = [1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    
    for step in test_steps:
        print(f"   📊 记录step {step}...")
        
        # 构建数据
        data = {
            "global_step": step,  # 明确设置global_step
            "training/loss": 2.0 - step * 0.008,
            "training/lr": 5e-6,
            "training/epoch": step * 0.01,
            "training/grad_norm": 1.5
        }
        
        # 每20步添加eval指标
        if step % 20 == 0:
            data.update({
                "eval/overall_loss": 1.5 - step * 0.005,
                "eval/overall_accuracy": 0.3 + step * 0.003,
                "eval/overall_samples": 1000,
                "eval/overall_correct": int(1000 * (0.3 + step * 0.003))
            })
        
        # 每40步添加perf指标
        if step % 40 == 0:
            data.update({
                "perf/step_time": 4.2,
                "perf/mfu": 0.35,
                "perf/tokens_per_second": 1200
            })
        
        # 关键：不使用step参数，让WandB自动处理
        wandb.log(data, commit=True)
        
        print(f"     ✅ Step {step}: 记录了 {len(data)} 个指标")
        
        # 短暂延迟确保数据顺序
        time.sleep(0.1)
    
    # 6. 验证数据记录
    print("\n🔍 验证数据记录...")
    try:
        # 等待数据同步
        time.sleep(2)
        
        print(f"✅ 数据记录完成")
        print(f"📊 总共记录了 {len(test_steps)} 个step的数据")
        print(f"🔗 请检查WandB URL: {run.url}")
        
    except Exception as e:
        print(f"⚠️ 验证失败: {e}")
    
    print("\n🎉 紧急修复完成!")
    print("📋 修复要点:")
    print("1. ✅ 创建了全新的WandB项目和运行")
    print("2. ✅ 正确定义了指标和step轴")
    print("3. ✅ 从step=1开始记录数据（避免step=0）")
    print("4. ✅ 使用global_step作为统一的x轴")
    print("5. ✅ 测试了完整的数据记录流程")
    
    print(f"\n🔗 新的WandB项目: {project_name}")
    print(f"🔗 新的WandB URL: {run.url}")
    
    print("\n💡 下一步:")
    print("1. 检查新的WandB URL，确认图表显示正常")
    print("2. 如果正常，请使用这个项目名称重新配置您的训练")
    print("3. 确保训练代码中不记录step=0的数据")
    
    return True, project_name, run.url

def update_training_config(new_project_name):
    """更新训练配置以使用新的项目名称"""
    print(f"\n🔧 更新训练配置...")
    
    config_suggestion = f"""
# 更新您的YAML配置文件中的wandb部分:
wandb:
  enabled: true
  project: "{new_project_name}"
  run_name: "training_run_{{timestamp}}"
  tags: ["fixed", "no_step_conflict"]
  notes: "使用修复后的step配置进行训练"
"""
    
    print(config_suggestion)
    
    # 创建一个临时配置文件
    temp_config_file = "wandb_fixed_config.yaml"
    with open(temp_config_file, 'w') as f:
        f.write(config_suggestion)
    
    print(f"✅ 临时配置已保存到: {temp_config_file}")

if __name__ == "__main__":
    print("🚨 WandB Step显示问题紧急修复工具")
    print("=" * 60)
    
    success, project_name, url = emergency_fix_wandb_step()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ 紧急修复成功完成!")
        
        update_training_config(project_name)
        
        print("\n🔍 验证步骤:")
        print(f"1. 打开WandB URL: {url}")
        print("2. 检查是否能看到正确的图表:")
        print("   - Training图表应该显示step 1-200的数据")
        print("   - Eval图表应该显示step 20,40,60,80,100,120,140,160,180,200的数据")
        print("   - Perf图表应该显示step 40,80,120,160,200的数据")
        print("3. 如果图表正常，使用新的项目名称重新启动训练")
        
    else:
        print("\n❌ 紧急修复失败")
        print("请检查WandB配置和网络连接") 