#!/usr/bin/env python3
"""
测试脚本：验证 eval_ratio 功能是否正确工作
"""

import os
import sys
import yaml
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.dataloader import create_dataloaders

def test_eval_ratio():
    """测试eval_ratio功能"""
    
    print("🧪 开始测试 eval_ratio 功能")
    print("=" * 60)
    
    # 创建测试配置
    test_config = {
        'model': {
            'pretrained_name': "/llm_reco/dehua/model/Qwen2.5-VL-7B-Instruct",
            'num_labels': 101
        },
        'datasets': {
            'dataset_configs': {
                'food101': {
                    'num_classes': 101,
                    'description': "Food-101 dataset",
                    'eval_ratio': 0.01  # 只使用1%的数据进行评估
                }
            },
            'enable_logits_masking': False,
            'shuffle_datasets': False
        },
        'data': {
            'train_jsonl': "/llm_reco/dehua/code/qwen_classify/data/dataset_food_label/food101_train.jsonl",
            'val_jsonl': "/llm_reco/dehua/code/qwen_classify/data/dataset_food_label/food101_test.jsonl"
        },
        'training': {
            'num_workers': 0,  # 设置为0避免多进程问题
            'evaluation': {
                'partial_eval_during_training': True,
                'full_eval_at_end': True,
                'eval_best_model_only': True
            }
        },
        'deepspeed': {
            'train_micro_batch_size_per_gpu': 1,
            'train_batch_size': 4
        }
    }
    
    print("📋 测试配置:")
    print(f"  • eval_ratio: {test_config['datasets']['dataset_configs']['food101']['eval_ratio']}")
    print(f"  • partial_eval_during_training: {test_config['training']['evaluation']['partial_eval_during_training']}")
    print("")
    
    try:
        # 检查数据文件是否存在
        train_file = test_config['data']['train_jsonl']
        val_file = test_config['data']['val_jsonl']
        
        if not os.path.exists(train_file):
            print(f"❌ 训练文件不存在: {train_file}")
            return False
            
        if not os.path.exists(val_file):
            print(f"❌ 验证文件不存在: {val_file}")
            return False
        
        print("✅ 数据文件检查通过")
        
        # 创建数据加载器
        print("🔄 创建数据加载器...")
        train_loader, val_loader = create_dataloaders(test_config)
        
        print(f"✅ 数据加载器创建成功")
        print(f"📊 训练集大小: {len(train_loader.dataset):,} samples")
        print(f"📊 验证集大小: {len(val_loader.dataset):,} samples")
        
        # 检查验证集是否被正确采样
        # 读取原始验证文件的行数
        with open(val_file, 'r') as f:
            original_val_count = sum(1 for _ in f)
        
        expected_val_count = int(original_val_count * test_config['datasets']['dataset_configs']['food101']['eval_ratio'])
        actual_val_count = len(val_loader.dataset)
        
        print("")
        print("🔍 验证 eval_ratio 功能:")
        print(f"  • 原始验证集样本数: {original_val_count:,}")
        print(f"  • eval_ratio: {test_config['datasets']['dataset_configs']['food101']['eval_ratio']}")
        print(f"  • 期望采样后样本数: {expected_val_count:,}")
        print(f"  • 实际采样后样本数: {actual_val_count:,}")
        
        # 允许一定的误差范围（因为随机采样）
        error_margin = max(1, int(expected_val_count * 0.1))  # 10%的误差范围
        
        if abs(actual_val_count - expected_val_count) <= error_margin:
            print("✅ eval_ratio 功能工作正常！")
            reduction_ratio = (1 - actual_val_count / original_val_count) * 100
            print(f"📈 评估数据减少了 {reduction_ratio:.1f}%，这将显著提升训练速度")
            return True
        else:
            print("❌ eval_ratio 功能异常！")
            print(f"   期望范围: {expected_val_count - error_margin} - {expected_val_count + error_margin}")
            print(f"   实际值: {actual_val_count}")
            return False
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_no_eval_ratio():
    """测试不使用eval_ratio的情况"""
    
    print("\n" + "=" * 60)
    print("🧪 测试对比：不使用 eval_ratio")
    print("=" * 60)
    
    # 创建对比配置（不使用eval_ratio）
    test_config = {
        'model': {
            'pretrained_name': "/llm_reco/dehua/model/Qwen2.5-VL-7B-Instruct",
            'num_labels': 101
        },
        'datasets': {
            'dataset_configs': {
                'food101': {
                    'num_classes': 101,
                    'description': "Food-101 dataset",
                    'eval_ratio': 1.0  # 使用100%的数据
                }
            },
            'enable_logits_masking': False,
            'shuffle_datasets': False
        },
        'data': {
            'train_jsonl': "/llm_reco/dehua/code/qwen_classify/data/dataset_food_label/food101_train.jsonl",
            'val_jsonl': "/llm_reco/dehua/code/qwen_classify/data/dataset_food_label/food101_test.jsonl"
        },
        'training': {
            'num_workers': 0,
            'evaluation': {
                'partial_eval_during_training': True,
                'full_eval_at_end': True,
                'eval_best_model_only': True
            }
        },
        'deepspeed': {
            'train_micro_batch_size_per_gpu': 1,
            'train_batch_size': 4
        }
    }
    
    try:
        train_loader, val_loader = create_dataloaders(test_config)
        
        # 读取原始验证文件的行数
        val_file = test_config['data']['val_jsonl']
        with open(val_file, 'r') as f:
            original_val_count = sum(1 for _ in f)
        
        actual_val_count = len(val_loader.dataset)
        
        print(f"📊 不使用 eval_ratio 时:")
        print(f"  • 原始验证集样本数: {original_val_count:,}")
        print(f"  • 实际使用样本数: {actual_val_count:,}")
        
        if actual_val_count == original_val_count:
            print("✅ 不使用 eval_ratio 时正常使用全部数据")
            return True
        else:
            print(f"❌ 期望使用全部数据，但实际只使用了 {actual_val_count}/{original_val_count}")
            return False
            
    except Exception as e:
        print(f"❌ 对比测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 开始 eval_ratio 功能测试")
    print("这个测试将验证修复后的 eval_ratio 功能是否正确工作")
    print("")
    
    # 运行测试
    test1_pass = test_eval_ratio()
    test2_pass = test_no_eval_ratio()
    
    print("\n" + "=" * 60)
    print("📋 测试结果汇总:")
    print(f"  • eval_ratio 功能测试: {'✅ 通过' if test1_pass else '❌ 失败'}")
    print(f"  • 对比测试（无 eval_ratio）: {'✅ 通过' if test2_pass else '❌ 失败'}")
    
    if test1_pass and test2_pass:
        print("\n🎉 所有测试通过！eval_ratio 功能已修复并正常工作")
        print("💡 现在您可以在单数据集模式下使用 eval_ratio 来加速训练过程")
    else:
        print("\n❌ 部分测试失败，请检查代码修改")
        sys.exit(1) 