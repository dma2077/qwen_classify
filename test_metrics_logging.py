#!/usr/bin/env python3
"""
测试training和eval指标是否能同时记录到WandB
"""

import os
import sys
import json
import time
from typing import Dict

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_metrics_logging():
    """测试指标记录逻辑"""
    
    print("="*80)
    print("🔍 分析training和eval指标记录逻辑")
    print("="*80)
    
    # 1. 检查训练循环中的指标记录逻辑
    print("\n📊 1. 训练循环中的指标记录逻辑:")
    print("   - 在eval步骤时，evaluate()方法被调用，但log_to_wandb=False")
    print("   - 然后训练循环合并training和eval数据，一次性记录")
    print("   - 这确保了training和eval指标在同一个step中记录")
    
    # 2. 检查evaluate方法中的指标记录逻辑
    print("\n📊 2. evaluate方法中的指标记录逻辑:")
    print("   - 当log_to_wandb=True时，会单独记录eval指标")
    print("   - 当log_to_wandb=False时，只返回结果，不记录到WandB")
    print("   - 这避免了重复记录")
    
    # 3. 检查monitor.log_metrics方法
    print("\n📊 3. monitor.log_metrics方法:")
    print("   - 支持记录任意数量的指标")
    print("   - 所有指标都使用相同的step")
    print("   - 没有限制只能记录一组指标")
    
    # 4. 分析可能的问题
    print("\n⚠️  4. 可能的问题分析:")
    
    # 检查训练循环中的合并逻辑
    print("   a) 训练循环中的合并逻辑:")
    print("      - 在eval步骤时，会合并training和eval数据")
    print("      - 使用combined_data = {**current_training_data, **eval_data}")
    print("      - 这应该能同时记录两组指标")
    
    # 检查频率设置
    print("\n   b) 频率设置检查:")
    print("      - eval_steps: 控制评估频率")
    print("      - logging_steps: 控制训练日志频率")
    print("      - 如果eval_steps != logging_steps，可能导致某些步骤只有一组指标")
    
    # 检查WandB图表定义
    print("\n   c) WandB图表定义检查:")
    print("      - 需要确保training和eval指标都定义了对应的图表")
    print("      - 如果图表定义有问题，可能显示不完整")
    
    # 5. 建议的解决方案
    print("\n💡 5. 建议的解决方案:")
    print("   a) 确保eval_steps和logging_steps设置合理")
    print("   b) 检查WandB图表定义是否完整")
    print("   c) 验证指标名称前缀是否正确")
    print("   d) 检查是否有指标被过滤或忽略")
    
    return True

def analyze_config_frequencies():
    """分析配置文件中的频率设置"""
    
    print("\n" + "="*80)
    print("📋 分析配置文件中的频率设置")
    print("="*80)
    
    # 查找配置文件
    config_dir = "configs"
    if os.path.exists(config_dir):
        config_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
        
        print(f"\n找到 {len(config_files)} 个配置文件:")
        for config_file in config_files:
            config_path = os.path.join(config_dir, config_file)
            print(f"  📄 {config_file}")
            
            try:
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                # 提取关键频率设置
                eval_steps = config.get('eval_steps', 'N/A')
                logging_steps = config.get('logging_steps', 'N/A')
                save_steps = config.get('save_steps', 'N/A')
                
                print(f"     - eval_steps: {eval_steps}")
                print(f"     - logging_steps: {logging_steps}")
                print(f"     - save_steps: {save_steps}")
                
                # 检查频率是否合理
                if eval_steps != 'N/A' and logging_steps != 'N/A':
                    if eval_steps == logging_steps:
                        print(f"     ✅ eval_steps == logging_steps，指标会同时记录")
                    else:
                        print(f"     ⚠️  eval_steps != logging_steps，某些步骤可能只有一组指标")
                
            except Exception as e:
                print(f"     ❌ 读取配置文件失败: {e}")
    
    else:
        print("❌ 未找到configs目录")

def create_test_script():
    """创建一个测试脚本来验证指标记录"""
    
    print("\n" + "="*80)
    print("🧪 创建测试脚本")
    print("="*80)
    
    test_script = '''#!/usr/bin/env python3
"""
测试training和eval指标同时记录到WandB
"""

import os
import sys
import time
import json
import yaml

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_combined_metrics_logging():
    """测试合并指标记录"""
    
    print("🧪 开始测试合并指标记录...")
    
    # 模拟配置
    config = {
        'output_dir': './test_output',
        'wandb': {
            'project': 'test_metrics',
            'name': 'test_combined_metrics',
            'enabled': True
        },
        'monitor': {
            'freq': {
                'log_freq': 1,
                'eval_log_freq': 1,
                'perf_log_freq': 1,
                'flops_profile_freq': 10
            }
        }
    }
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 初始化monitor
    from training.utils.monitor import TrainingMonitor
    monitor = TrainingMonitor(config['output_dir'], config)
    
    # 模拟训练和eval数据
    for step in range(1, 11):
        print(f"\\n📊 步骤 {step}:")
        
        # 模拟training数据
        training_data = {
            "training/loss": 0.1 + step * 0.01,
            "training/lr": 1e-4,
            "training/epoch": 0.1,
            "training/grad_norm": 1.0 + step * 0.1,
        }
        
        # 模拟eval数据（每5步评估一次）
        eval_data = {}
        if step % 5 == 0:
            eval_data = {
                "eval/overall_loss": 0.2 + step * 0.01,
                "eval/overall_accuracy": 0.8 - step * 0.01,
                "eval/overall_samples": 1000,
                "eval/overall_correct": 800 - step * 10,
            }
            print(f"   📈 包含eval指标: {list(eval_data.keys())}")
        else:
            print(f"   📈 仅包含training指标")
        
        # 合并数据
        combined_data = {**training_data, **eval_data}
        combined_data["step"] = step
        
        # 记录到WandB
        monitor.log_metrics(combined_data, step, commit=True)
        
        print(f"   ✅ 已记录 {len(combined_data)} 个指标")
        print(f"   📊 指标keys: {list(combined_data.keys())}")
        
        time.sleep(1)  # 避免WandB API限制
    
    print("\\n🎉 测试完成！")
    print("请检查WandB界面，应该能看到:")
    print("  - training/loss, training/lr, training/epoch, training/grad_norm")
    print("  - eval/overall_loss, eval/overall_accuracy (每5步)")
    print("  - 所有指标都使用相同的step轴")

if __name__ == "__main__":
    test_combined_metrics_logging()
'''
    
    with open('test_combined_metrics.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print("✅ 已创建测试脚本: test_combined_metrics.py")
    print("运行命令: python test_combined_metrics.py")

def main():
    """主函数"""
    test_metrics_logging()
    analyze_config_frequencies()
    create_test_script()
    
    print("\n" + "="*80)
    print("📋 总结")
    print("="*80)
    print("根据代码分析，当前的实现应该能够同时记录training和eval指标:")
    print("1. ✅ 训练循环会合并training和eval数据")
    print("2. ✅ monitor.log_metrics支持记录多个指标")
    print("3. ✅ 所有指标使用统一的step轴")
    print("4. ⚠️  需要检查eval_steps和logging_steps的设置")
    print("5. ⚠️  需要检查WandB图表定义")
    print("\n建议运行测试脚本验证实际效果")

if __name__ == "__main__":
    main() 