#!/usr/bin/env python3
"""
测试WandB配置修复
"""

import os
import sys
import yaml
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_config_processing():
    """测试配置处理"""
    print("🧪 测试配置处理...")
    
    # 模拟用户的配置
    user_config = {
        'model': {
            'pretrained_name': "/test/model/path",
            'num_labels': 172
        },
        'training': {
            'epochs': 5,  # 用户使用的字段名
            'lr': 5e-6,   # 用户使用的字段名
            'output_dir': "/test/output"
        },
        'data': {
            'train_jsonl': "/test/train.jsonl",
            'val_jsonl': "/test/val.jsonl"
        },
        'deepspeed': {
            'config_file': "configs/ds_s2.json",  # 用户使用的结构
            'zero_stage': 2,
            'bf16': True
        },
        'wandb': {
            'enabled': True,  # 关键：用户启用了WandB
            'project': "qwen_classification",
            'run_name': "test_run"
        }
    }
    
    print("📋 原始配置:")
    print(f"  • training.epochs: {user_config['training']['epochs']}")
    print(f"  • training.lr: {user_config['training']['lr']}")
    print(f"  • deepspeed结构: {type(user_config['deepspeed'])}")
    print(f"  • wandb.enabled: {user_config['wandb']['enabled']}")
    
    # 处理配置
    from training.utils.config_utils import prepare_config
    
    try:
        processed_config = prepare_config(user_config)
        
        print("\n✅ 配置处理成功!")
        print("📋 处理后的配置:")
        print(f"  • training.num_epochs: {processed_config['training'].get('num_epochs', 'NOT_FOUND')}")
        print(f"  • training.learning_rate: {processed_config['training'].get('learning_rate', 'NOT_FOUND')}")
        print(f"  • deepspeed: {processed_config['deepspeed']}")
        print(f"  • wandb.enabled: {processed_config['wandb']['enabled']}")
        
        return processed_config
        
    except Exception as e:
        print(f"❌ 配置处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_wandb_initialization():
    """测试WandB初始化"""
    print("\n🧪 测试WandB初始化...")
    
    # 获取处理后的配置
    config = test_config_processing()
    if config is None:
        print("❌ 无法测试WandB初始化，配置处理失败")
        return
    
    # 创建测试输出目录
    output_dir = "./test_wandb_output"
    os.makedirs(output_dir, exist_ok=True)
    config['output_dir'] = output_dir
    
    try:
        from training.utils.monitor import TrainingMonitor
        
        print("🔧 创建TrainingMonitor...")
        monitor = TrainingMonitor(output_dir, config)
        
        print(f"📊 Monitor状态:")
        print(f"  • use_wandb: {monitor.use_wandb}")
        print(f"  • WANDB_AVAILABLE: {getattr(monitor, 'WANDB_AVAILABLE', 'N/A')}")
        print(f"  • _is_main_process(): {monitor._is_main_process()}")
        
        if monitor.use_wandb:
            print("✅ WandB初始化成功!")
            
            # 测试指标记录
            test_metrics = {
                "training/loss": 0.5,
                "training/lr": 5e-6,
                "eval/overall_accuracy": 0.8,
                "eval/overall_loss": 0.3
            }
            
            monitor.log_metrics(test_metrics, step=1, commit=True)
            print("✅ 测试指标记录成功!")
            
        else:
            print("⚠️ WandB未初始化")
            
    except Exception as e:
        print(f"❌ WandB初始化失败: {e}")
        import traceback
        traceback.print_exc()

def test_user_yaml_config():
    """测试用户实际的YAML配置"""
    print("\n🧪 测试用户YAML配置...")
    
    # 创建临时YAML文件（模拟用户的配置）
    yaml_content = """
model:
  pretrained_name: "/llm_reco/dehua/model/Qwen2.5-VL-7B-Instruct"
  num_labels: 172

training:
  epochs: 5
  lr: 5e-6
  weight_decay: 0.01
  output_dir: "/test/output"
  eval_steps: 100

deepspeed:
  config_file: "configs/ds_s2.json"
  zero_stage: 2
  bf16: true

wandb:
  enabled: true
  project: "qwen_classification"
  run_name: "test_run"

data:
  train_jsonl: "/test/train.jsonl"
  val_jsonl: "/test/val.jsonl"
"""
    
    # 写入临时文件
    temp_yaml = "./temp_test_config.yaml"
    with open(temp_yaml, 'w') as f:
        f.write(yaml_content)
    
    try:
        # 加载YAML配置
        with open(temp_yaml, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        print("📋 YAML配置加载成功")
        print(f"  • training.epochs: {yaml_config['training']['epochs']}")
        print(f"  • training.lr: {yaml_config['training']['lr']}")
        print(f"  • wandb.enabled: {yaml_config['wandb']['enabled']}")
        
        # 处理配置
        from training.utils.config_utils import prepare_config
        processed_config = prepare_config(yaml_config)
        
        print("✅ YAML配置处理成功!")
        
    except Exception as e:
        print(f"❌ YAML配置测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理临时文件
        if os.path.exists(temp_yaml):
            os.remove(temp_yaml)

if __name__ == "__main__":
    print("🚀 开始WandB配置修复测试...")
    
    test_config_processing()
    test_wandb_initialization()
    test_user_yaml_config()
    
    print("\n🎉 测试完成!") 