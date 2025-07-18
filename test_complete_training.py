#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试complete_train.py的修复
"""

import sys
import os
import yaml

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_config_loading():
    """测试配置文件加载"""
    print("🧪 测试配置文件加载...")
    
    try:
        # 测试你的配置文件
        config_path = "configs/food101_cosine_hold_5e_6_ls.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("✅ 配置文件加载成功")
        
        # 检查关键字段
        print(f"📋 模型名称: {config['model']['pretrained_name']}")
        print(f"📋 类别数: {config['model']['num_labels']}")
        print(f"📋 学习率: {config['training']['lr']}")
        print(f"📋 训练轮数: {config['training']['epochs']}")
        print(f"📋 损失函数: {config['loss']['type']}")
        
        return config
        
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return None

def test_model_creation(config):
    """测试模型创建"""
    print("\n🧪 测试模型创建...")
    
    try:
        from models.qwen2_5_vl_classify import Qwen2_5_VLForImageClassification
        
        # 获取配置
        loss_config = config.get('loss', {'type': 'cross_entropy'})
        dataset_configs = config.get('datasets', {}).get('dataset_configs', {})
        enable_logits_masking = config.get('datasets', {}).get('enable_logits_masking', True)
        
        # 创建模型
        model = Qwen2_5_VLForImageClassification(
            pretrained_model_name=config['model']['pretrained_name'],
            num_labels=config['model']['num_labels'],
            loss_config=loss_config,
            dataset_configs=dataset_configs,
            enable_logits_masking=enable_logits_masking
        )
        
        print("✅ 模型创建成功")
        return model
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return None

def test_optimizer_creation(model, config):
    """测试优化器创建"""
    print("\n🧪 测试优化器创建...")
    
    try:
        from optimizer.optimizer import create_optimizer
        
        optimizer = create_optimizer(model, config)
        print("✅ 优化器创建成功")
        return optimizer
        
    except Exception as e:
        print(f"❌ 优化器创建失败: {e}")
        return None

def test_lr_scheduler_creation(optimizer, config):
    """测试学习率调度器创建"""
    print("\n🧪 测试学习率调度器创建...")
    
    try:
        from training.lr_scheduler import create_lr_scheduler
        
        # 使用临时的steps_per_epoch
        temp_steps_per_epoch = 1000
        lr_scheduler = create_lr_scheduler(optimizer, config, temp_steps_per_epoch)
        
        print("✅ 学习率调度器创建成功")
        return lr_scheduler
        
    except Exception as e:
        print(f"❌ 学习率调度器创建失败: {e}")
        return None

def test_dataloader_creation(config):
    """测试数据加载器创建"""
    print("\n🧪 测试数据加载器创建...")
    
    try:
        from data.dataloader import create_dataloaders
        
        train_loader, val_loader = create_dataloaders(config)
        
        print("✅ 数据加载器创建成功")
        print(f"📋 训练集大小: {len(train_loader.dataset)}")
        print(f"📋 验证集大小: {len(val_loader.dataset)}")
        
        return train_loader, val_loader
        
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        return None, None

def main():
    """主测试函数"""
    print("🚀 开始测试complete_train.py的修复")
    print("=" * 50)
    
    # 测试配置文件加载
    config = test_config_loading()
    if config is None:
        return
    
    # 测试模型创建
    model = test_model_creation(config)
    if model is None:
        return
    
    # 测试优化器创建
    optimizer = test_optimizer_creation(model, config)
    if optimizer is None:
        return
    
    # 测试学习率调度器创建
    lr_scheduler = test_lr_scheduler_creation(optimizer, config)
    if lr_scheduler is None:
        return
    
    # 测试数据加载器创建
    train_loader, val_loader = test_dataloader_creation(config)
    if train_loader is None:
        return
    
    print("\n" + "=" * 50)
    print("✅ 所有测试通过！complete_train.py应该可以正常运行了")

if __name__ == "__main__":
    main() 