#!/usr/bin/env python3
"""
调试损失函数创建过程
"""
import torch
import torch.nn as nn
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_loss_function():
    """调试损失函数创建过程"""
    print("🔍 开始调试损失函数创建...")
    
    # 测试1: 直接创建损失函数
    print("\n1️⃣ 测试直接创建损失函数:")
    try:
        from training.losses import create_loss_function
        
        # 测试默认配置
        loss_config = {'type': 'label_smoothing', 'smoothing': 0.1, 'temperature': 1.0}
        loss_type = loss_config.get('type', 'cross_entropy')
        loss_kwargs = {k: v for k, v in loss_config.items() if k != 'type'}
        
        print(f"   loss_type: {loss_type}")
        print(f"   loss_kwargs: {loss_kwargs}")
        
        loss_function = create_loss_function(loss_type, **loss_kwargs)
        print(f"   创建的损失函数: {type(loss_function)}")
        print(f"   损失函数对象: {loss_function}")
        
        # 测试调用
        logits = torch.randn(2, 101)
        labels = torch.randint(0, 101, (2,))
        loss = loss_function(logits, labels)
        print(f"   测试调用结果: {loss.item():.4f}")
        
    except Exception as e:
        print(f"❌ 直接创建失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试2: 模拟模型创建过程
    print("\n2️⃣ 测试模型创建过程:")
    try:
        from models.qwen2_5_vl_classify import Qwen2_5_VLForImageClassification
        
        # 这会触发模型的创建过程
        print("   尝试创建模型...")
        # 注意：这可能会失败，因为需要实际的模型文件
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        print("   这是预期的，因为需要实际的模型文件")
    
    # 测试3: 检查可能的导入问题
    print("\n3️⃣ 检查导入:")
    try:
        import torch.nn as nn
        print(f"   torch.nn.CrossEntropyLoss: {nn.CrossEntropyLoss}")
        
        # 检查是否有其他可能的导入
        print("   可用的损失函数:")
        print(f"   - nn.CrossEntropyLoss: {nn.CrossEntropyLoss}")
        print(f"   - nn.BCELoss: {nn.BCELoss}")
        print(f"   - nn.MSELoss: {nn.MSELoss}")
        
    except Exception as e:
        print(f"❌ 导入检查失败: {e}")
    
    print("\n🎉 调试完成!")

if __name__ == "__main__":
    debug_loss_function() 