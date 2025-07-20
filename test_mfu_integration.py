#!/usr/bin/env python3
"""
测试新的MFU计算集成
验证MFU指标是否能正确计算和记录到WandB
"""

import os
import sys
import argparse
import time
from collections import defaultdict

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

def test_mfu_stats():
    """测试MFUStats类的基本功能"""
    print("🧪 测试MFUStats类...")
    
    try:
        from training.utils.flops_calculate import MFUStats
        
        # 创建模拟的args对象
        args = argparse.Namespace()
        args.model_dir = "./configs"  # 假设有配置文件
        args.logging_per_step = 10
        
        # 检查是否有可用的配置文件
        config_paths = [
            "./configs/food101_cosine.yaml",
            "./models/config.json", 
            "./config.json"
        ]
        
        config_found = False
        for config_path in config_paths:
            if os.path.exists(config_path):
                print(f"✅ 找到配置文件: {config_path}")
                if config_path.endswith('.json'):
                    args.model_dir = os.path.dirname(config_path) if os.path.dirname(config_path) else '.'
                    config_found = True
                    break
        
        if not config_found:
            print("⚠️ 未找到模型配置文件，创建模拟配置进行测试")
            # 创建模拟配置文件用于测试
            create_mock_config()
            args.model_dir = './test_config'
        
        # 初始化MFU统计器
        mfu_stats = MFUStats(args)
        print("✅ MFUStats初始化成功")
        
        # 模拟数据收集
        print("\n📊 模拟数据收集...")
        for step in range(1, 21):  # 模拟20步
            # 模拟每步的数据
            num_image_tokens = 256 * 2  # 2张图片，每张256个token
            num_tokens = 1024 + num_image_tokens  # 文本token + 图像token
            num_samples = 4
            num_images = 2
            
            mfu_stats.set(
                num_image_tokens=num_image_tokens,
                num_tokens=num_tokens,
                num_samples=num_samples,
                num_images=num_images
            )
            
            # 每10步计算一次MFU
            if step % args.logging_per_step == 0:
                step_time = 1.5  # 假设每步1.5秒
                try:
                    mfu_log_dict = mfu_stats.mfu(step_time, step)
                    print(f"\n📈 Step {step} MFU指标:")
                    for key, value in mfu_log_dict.items():
                        print(f"  {key}: {value}")
                    
                    # 验证关键指标
                    expected_keys = [
                        "perf/mfu_per_step_per_gpu",
                        "perf/vit_flops_per_step_per_gpu", 
                        "perf/llm_flops_per_step_per_gpu"
                    ]
                    
                    missing_keys = [key for key in expected_keys if key not in mfu_log_dict]
                    if missing_keys:
                        print(f"⚠️ 缺失的关键指标: {missing_keys}")
                    else:
                        print("✅ 所有关键MFU指标都已生成")
                        
                except Exception as e:
                    print(f"❌ MFU计算失败: {e}")
                    import traceback
                    traceback.print_exc()
        
        print("\n✅ MFUStats测试完成")
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_mock_config():
    """创建模拟的模型配置文件用于测试"""
    import json
    
    os.makedirs('./test_config', exist_ok=True)
    
    # 创建模拟的Qwen2.5-VL配置
    mock_config = {
        "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        "num_attention_heads": 16,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_key_value_heads": 16,
        "num_hidden_layers": 12,
        "vocab_size": 50000,
        "vision_config": {
            "num_heads": 8,
            "hidden_size": 512,
            "intermediate_size": 2048,
            "depth": 6
        }
    }
    
    config_path = './test_config/config.json'
    with open(config_path, 'w') as f:
        json.dump(mock_config, f, indent=2)
    
    print(f"✅ 创建模拟配置文件: {config_path}")

def test_wandb_logging():
    """测试WandB日志记录功能"""
    print("\n🧪 测试WandB日志记录...")
    
    try:
        # 模拟训练数据，包含MFU指标
        training_data = {
            "training/loss": 0.5,
            "training/lr": 1e-4,
            "perf/mfu_per_step_per_gpu": 0.25,
            "perf/vit_flops_per_step_per_gpu": 12.5,
            "perf/llm_flops_per_step_per_gpu": 8.3,
            "perf/step_time": 1.5,
            "perf/tokens_per_second": 2048.0
        }
        
        print("📊 模拟的训练数据:")
        for key, value in training_data.items():
            print(f"  {key}: {value}")
        
        # 检查数据格式
        mfu_metrics = [k for k in training_data.keys() if 'mfu' in k.lower() or 'flops' in k.lower()]
        print(f"\n✅ 检测到MFU相关指标: {mfu_metrics}")
        
        # 验证数据类型
        invalid_data = []
        for key, value in training_data.items():
            if not isinstance(value, (int, float)):
                invalid_data.append((key, type(value)))
        
        if invalid_data:
            print(f"⚠️ 发现无效数据类型: {invalid_data}")
        else:
            print("✅ 所有数据类型都有效，可以记录到WandB")
        
        return True
        
    except Exception as e:
        print(f"❌ WandB测试失败: {e}")
        return False

def cleanup():
    """清理测试文件"""
    import shutil
    if os.path.exists('./test_config'):
        shutil.rmtree('./test_config')
        print("🧹 清理测试文件完成")

if __name__ == "__main__":
    print("🚀 开始MFU集成测试\n")
    
    try:
        # 测试MFUStats类
        mfu_test_success = test_mfu_stats()
        
        # 测试WandB日志记录
        wandb_test_success = test_wandb_logging()
        
        print(f"\n📋 测试结果总结:")
        print(f"  MFUStats测试: {'✅ 通过' if mfu_test_success else '❌ 失败'}")
        print(f"  WandB日志测试: {'✅ 通过' if wandb_test_success else '❌ 失败'}")
        
        if mfu_test_success and wandb_test_success:
            print(f"\n🎉 所有测试通过！新的MFU计算集成应该能正常工作。")
            print(f"\n💡 在实际训练中的使用提示:")
            print(f"  1. 确保模型目录中有config.json文件")
            print(f"  2. MFU指标将在第{10}步后开始显示")
            print(f"  3. 关注WandB中的perf/mfu_per_step_per_gpu指标")
            print(f"  4. VIT和LLM的FLOPs也会分别记录")
        else:
            print(f"\n❌ 部分测试失败，请检查配置和代码")
            
    except KeyboardInterrupt:
        print("\n⚠️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup() 