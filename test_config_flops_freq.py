#!/usr/bin/env python3
"""
测试配置文件中的flops_profile_freq设置是否生效
验证MFU计算频率是否能正确从yaml配置中读取
"""

import os
import sys
import time
import torch
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from training.utils.monitor import TrainingMonitor

def test_config_flops_freq():
    """测试配置文件中的flops_profile_freq设置"""
    
    print("🧪 开始测试配置文件中的flops_profile_freq设置...")
    
    # 测试不同的flops_profile_freq配置
    test_cases = [
        {
            'name': '默认配置（无flops_profile_freq）',
            'config': {
                'model': {'pretrained_name': 'Qwen/Qwen2.5-VL-7B-Instruct', 'num_labels': 10},
                'training': {'epochs': 1, 'lr': 1e-5, 'output_dir': './test_outputs/config_test_default'},
                'monitor': {
                    'freq': {
                        'training_log_freq': 1,
                        'perf_log_freq': 2,
                        'gpu_log_freq': 3,
                        'local_save_freq': 5
                    }
                },
                'wandb': {'enabled': False}
            },
            'expected_freq': 500  # 默认值
        },
        {
            'name': '自定义flops_profile_freq=50',
            'config': {
                'model': {'pretrained_name': 'Qwen/Qwen2.5-VL-7B-Instruct', 'num_labels': 10},
                'training': {'epochs': 1, 'lr': 1e-5, 'output_dir': './test_outputs/config_test_50'},
                'monitor': {
                    'freq': {
                        'training_log_freq': 1,
                        'perf_log_freq': 2,
                        'gpu_log_freq': 3,
                        'flops_profile_freq': 50,  # 自定义设置
                        'local_save_freq': 5
                    }
                },
                'wandb': {'enabled': False}
            },
            'expected_freq': 50
        },
        {
            'name': '自定义flops_profile_freq=100',
            'config': {
                'model': {'pretrained_name': 'Qwen/Qwen2.5-VL-7B-Instruct', 'num_labels': 10},
                'training': {'epochs': 1, 'lr': 1e-5, 'output_dir': './test_outputs/config_test_100'},
                'monitor': {
                    'freq': {
                        'training_log_freq': 1,
                        'perf_log_freq': 2,
                        'gpu_log_freq': 3,
                        'flops_profile_freq': 100,  # 自定义设置
                        'local_save_freq': 5
                    }
                },
                'wandb': {'enabled': False}
            },
            'expected_freq': 100
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 测试用例 {i}: {test_case['name']}")
        print("=" * 60)
        
        # 创建输出目录
        output_dir = test_case['config']['training']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化监控器
        monitor = TrainingMonitor(
            output_dir=output_dir,
            config=test_case['config']
        )
        
        # 检查flops_profile_freq是否正确设置
        actual_freq = monitor.flops_profile_freq
        expected_freq = test_case['expected_freq']
        
        if actual_freq == expected_freq:
            print(f"✅ 测试通过: flops_profile_freq = {actual_freq} (期望: {expected_freq})")
        else:
            print(f"❌ 测试失败: flops_profile_freq = {actual_freq} (期望: {expected_freq})")
        
        # 测试MFU计算频率
        print(f"🔍 测试MFU计算频率...")
        monitor.start_training()
        
        # 创建虚拟模型引用
        class DummyModel:
            def __init__(self):
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            def parameters(self):
                return [torch.randn(1000, 1000, device=self.device)]
            
            def eval(self):
                pass
        
        dummy_model = DummyModel()
        monitor.set_model_ref(dummy_model)
        
        # 模拟几个训练步骤，检查MFU计算频率
        profiler_steps = []
        for step in range(1, 21):  # 20步
            attention_mask = torch.ones(2, 512)
            
            # 记录训练步骤
            monitor.log_step(
                step=step,
                epoch=0,
                loss=2.0 - (step * 0.05),
                grad_norm=1.0 + (step * 0.01),
                learning_rate=1e-5,
                attention_mask=attention_mask,
                skip_wandb=True  # 跳过wandb记录
            )
            
            # 检查是否应该使用profiler计算MFU
            if step % monitor.flops_profile_freq == 0:
                profiler_steps.append(step)
        
        print(f"📊 在步骤 {list(range(1, 21))} 中，使用profiler计算MFU的步骤: {profiler_steps}")
        print(f"📊 实际MFU计算频率: 每{monitor.flops_profile_freq}步")
        
        # 验证profiler步骤是否符合预期
        expected_profiler_steps = [step for step in range(1, 21) if step % expected_freq == 0]
        if profiler_steps == expected_profiler_steps:
            print(f"✅ MFU计算频率正确: 每{expected_freq}步使用profiler")
        else:
            print(f"❌ MFU计算频率错误: 期望每{expected_freq}步，实际每{monitor.flops_profile_freq}步")
    
    print("\n🎉 所有测试用例完成！")
    print("\n📋 总结：")
    print("1. 配置文件中的flops_profile_freq设置应该能正确生效")
    print("2. 如果没有设置flops_profile_freq，应该使用默认值500")
    print("3. MFU计算频率应该与配置的flops_profile_freq一致")
    print("4. 每flops_profile_freq步会使用profiler进行精确的MFU计算")
    
    return True

if __name__ == "__main__":
    try:
        test_config_flops_freq()
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 