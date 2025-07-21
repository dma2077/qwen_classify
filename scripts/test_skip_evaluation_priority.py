#!/usr/bin/env python3
"""
测试脚本：验证 skip_evaluation 参数的最高优先级功能
"""

import os
import sys
import yaml
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_skip_evaluation_priority():
    """测试 skip_evaluation 参数的最高优先级功能"""
    
    print("🧪 开始测试 skip_evaluation 最高优先级功能")
    print("=" * 80)
    
    # 加载测试配置文件
    config_path = project_root / "configs" / "test_skip_evaluation_priority.yaml"
    
    if not config_path.exists():
        print(f"❌ 测试配置文件不存在: {config_path}")
        return False
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("📋 原始配置参数:")
    print(f"  • skip_evaluation: {config['training']['skip_evaluation']}")
    print(f"  • save_all_checkpoints: {config['training']['save_all_checkpoints']}")
    print(f"  • best_model_tracking.enabled: {config['training']['best_model_tracking']['enabled']}")
    print(f"  • best_model_tracking.save_best_only: {config['training']['best_model_tracking']['save_best_only']}")
    print(f"  • evaluation.partial_eval_during_training: {config['training']['evaluation']['partial_eval_during_training']}")
    print(f"  • evaluation.full_eval_at_end: {config['training']['evaluation']['full_eval_at_end']}")
    print(f"  • evaluation.eval_best_model_only: {config['training']['evaluation']['eval_best_model_only']}")
    print("")
    
    try:
        # 导入 DeepSpeedTrainer 并初始化
        from training.deepspeed_trainer import DeepSpeedTrainer
        
        print("🔄 初始化 DeepSpeedTrainer...")
        trainer = DeepSpeedTrainer(config)
        
        print("✅ DeepSpeedTrainer 初始化成功")
        print("")
        
        # 验证参数是否被正确覆盖
        print("🔍 验证参数覆盖结果:")
        
        tests = [
            ("skip_evaluation", trainer.skip_evaluation, True, "应该保持为True"),
            ("best_model_enabled", trainer.best_model_enabled, False, "应该被强制设为False"),
            ("save_best_only", trainer.save_best_only, False, "应该被强制设为False"),
            ("save_all_checkpoints", trainer.save_all_checkpoints, True, "应该被强制设为True"),
            ("partial_eval_during_training", trainer.partial_eval_during_training, False, "应该被强制设为False"),
            ("full_eval_at_end", trainer.full_eval_at_end, False, "应该被强制设为False"),
            ("eval_best_model_only", trainer.eval_best_model_only, False, "应该被强制设为False"),
        ]
        
        all_passed = True
        for param_name, actual_value, expected_value, description in tests:
            if actual_value == expected_value:
                print(f"  ✅ {param_name}: {actual_value} ({description})")
            else:
                print(f"  ❌ {param_name}: {actual_value}, 期望: {expected_value} ({description})")
                all_passed = False
        
        print("")
        
        # 验证配置也被正确修改
        print("🔍 验证配置文件内的参数也被覆盖:")
        config_tests = [
            ("training.best_model_tracking.enabled", config['training']['best_model_tracking']['enabled'], False),
            ("training.best_model_tracking.save_best_only", config['training']['best_model_tracking']['save_best_only'], False),
            ("training.save_all_checkpoints", config['training']['save_all_checkpoints'], True),
            ("training.evaluation.partial_eval_during_training", config['training']['evaluation']['partial_eval_during_training'], False),
            ("training.evaluation.full_eval_at_end", config['training']['evaluation']['full_eval_at_end'], False),
            ("training.evaluation.eval_best_model_only", config['training']['evaluation']['eval_best_model_only'], False),
        ]
        
        for param_path, actual_value, expected_value in config_tests:
            if actual_value == expected_value:
                print(f"  ✅ {param_path}: {actual_value}")
            else:
                print(f"  ❌ {param_path}: {actual_value}, 期望: {expected_value}")
                all_passed = False
        
        print("")
        
        # 测试评估函数是否正确跳过
        print("🔍 测试评估函数是否正确跳过:")
        try:
            eval_loss, eval_accuracy = trainer.evaluate(step=1, log_to_wandb=False)
            if eval_loss == 0.0 and eval_accuracy == 0.0:
                print("  ✅ evaluate() 方法正确返回默认值 (0.0, 0.0)")
            else:
                print(f"  ❌ evaluate() 方法返回了非默认值: ({eval_loss}, {eval_accuracy})")
                all_passed = False
        except Exception as e:
            print(f"  ❌ evaluate() 方法调用失败: {e}")
            all_passed = False
        
        # 测试带返回结果的评估函数
        try:
            eval_loss, eval_accuracy, eval_results = trainer.evaluate(step=1, log_to_wandb=False, return_results=True)
            expected_results = {'overall_loss': 0.0, 'overall_accuracy': 0.0}
            if eval_loss == 0.0 and eval_accuracy == 0.0 and eval_results == expected_results:
                print("  ✅ evaluate() 方法 (return_results=True) 正确返回默认值")
            else:
                print(f"  ❌ evaluate() 方法 (return_results=True) 返回了非默认值")
                print(f"    eval_loss: {eval_loss}, eval_accuracy: {eval_accuracy}")
                print(f"    eval_results: {eval_results}")
                all_passed = False
        except Exception as e:
            print(f"  ❌ evaluate() 方法 (return_results=True) 调用失败: {e}")
            all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_without_skip_evaluation():
    """测试不设置 skip_evaluation 时的正常行为"""
    
    print("\n" + "=" * 80)
    print("🧪 对比测试：不设置 skip_evaluation")
    print("=" * 80)
    
    # 创建对比配置（不设置 skip_evaluation）
    config = {
        'model': {
            'pretrained_name': "/llm_reco/dehua/model/Qwen2.5-VL-7B-Instruct",
            'num_labels': 101
        },
        'training': {
            'lr': 1e-5,
            'output_dir': "/tmp/test_normal",
            'logging_steps': 10,
            'save_steps': 100,
            'eval_steps': 50,
            'best_model_tracking': {
                'enabled': True,
                'save_best_only': True
            },
            'evaluation': {
                'partial_eval_during_training': True,
                'full_eval_at_end': True,
                'eval_best_model_only': True
            }
        },
        'datasets': {
            'dataset_configs': {
                'food101': {'num_classes': 101}
            }
        }
    }
    
    try:
        from training.deepspeed_trainer import DeepSpeedTrainer
        
        trainer = DeepSpeedTrainer(config)
        
        print("📋 不设置 skip_evaluation 时的参数值:")
        print(f"  • skip_evaluation: {trainer.skip_evaluation}")
        print(f"  • best_model_enabled: {trainer.best_model_enabled}")
        print(f"  • save_best_only: {trainer.save_best_only}")
        print(f"  • partial_eval_during_training: {trainer.partial_eval_during_training}")
        print(f"  • full_eval_at_end: {trainer.full_eval_at_end}")
        print(f"  • eval_best_model_only: {trainer.eval_best_model_only}")
        
        # 验证参数应该保持原始值
        expected_values = {
            'skip_evaluation': False,
            'best_model_enabled': True,
            'save_best_only': True,
            'partial_eval_during_training': True,
            'full_eval_at_end': True,
            'eval_best_model_only': True
        }
        
        all_correct = True
        for param_name, expected_value in expected_values.items():
            actual_value = getattr(trainer, param_name)
            if actual_value == expected_value:
                print(f"  ✅ {param_name}: {actual_value} (正确)")
            else:
                print(f"  ❌ {param_name}: {actual_value}, 期望: {expected_value}")
                all_correct = False
        
        return all_correct
        
    except Exception as e:
        print(f"❌ 对比测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 开始 skip_evaluation 最高优先级功能测试")
    print("这个测试将验证 skip_evaluation=true 是否能强制覆盖所有相关参数")
    print("")
    
    # 运行测试
    test1_pass = test_skip_evaluation_priority()
    test2_pass = test_without_skip_evaluation()
    
    print("\n" + "=" * 80)
    print("📋 测试结果汇总:")
    print(f"  • skip_evaluation 最高优先级测试: {'✅ 通过' if test1_pass else '❌ 失败'}")
    print(f"  • 对比测试（正常模式）: {'✅ 通过' if test2_pass else '❌ 失败'}")
    
    if test1_pass and test2_pass:
        print("\n🎉 所有测试通过！skip_evaluation 最高优先级功能正常工作")
        print("💡 现在 skip_evaluation=true 会强制覆盖所有相关的评估参数")
    else:
        print("\n❌ 部分测试失败，请检查代码修改")
        sys.exit(1) 