#!/usr/bin/env python3
"""
测试损失函数创建是否正常工作
"""
import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.losses import create_loss_function

def test_loss_creation():
    """测试各种损失函数的创建"""
    
    print("🧪 测试损失函数创建...")
    
    # 测试配置
    test_configs = [
        {'type': 'cross_entropy'},
        {'type': 'label_smoothing', 'smoothing': 0.1, 'temperature': 1.0},
        {'type': 'focal', 'alpha': 1.0, 'gamma': 2.0},
        {'type': 'arcface', 'in_features': 768, 'out_features': 101, 's': 30.0, 'm': 0.5},
        {'type': 'supcon', 'temperature': 0.07},
        {'type': 'symmetric_ce', 'alpha': 1.0, 'beta': 1.0, 'num_classes': 101},
        {'type': 'mixup', 'alpha': 1.0},
    ]
    
    for config in test_configs:
        try:
            loss_type = config.get('type', 'cross_entropy')
            loss_kwargs = {k: v for k, v in config.items() if k != 'type'}
            
            print(f"\n📋 创建损失函数: {loss_type}")
            print(f"📋 损失函数参数: {loss_kwargs}")
            
            loss_function = create_loss_function(loss_type, **loss_kwargs)
            print(f"✅ 成功创建 {loss_type}: {type(loss_function)}")
            
            # 简单测试损失函数是否能正常调用
            if loss_type not in ['arcface', 'supcon']:
                # 标准损失函数测试
                logits = torch.randn(4, 101)  # batch_size=4, num_classes=101
                labels = torch.randint(0, 101, (4,))
                loss = loss_function(logits, labels)
                print(f"✅ 测试调用成功, loss: {loss.item():.4f}")
            else:
                print(f"✅ 创建成功 (跳过调用测试)")
                
        except Exception as e:
            print(f"❌ 创建 {config.get('type', 'unknown')} 失败: {e}")
    
    print("\n🎉 损失函数创建测试完成!")

if __name__ == "__main__":
    test_loss_creation() 