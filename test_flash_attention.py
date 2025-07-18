#!/usr/bin/env python3
"""
测试FlashAttention是否正确启用
"""

import os
import sys
import torch
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_flash_attention():
    """测试FlashAttention是否正确启用"""
    
    print("🔍 测试FlashAttention支持...")
    
    # 1. 检查CUDA支持
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name()}")
    
    # 2. 检查transformers版本
    try:
        import transformers
        print(f"Transformers版本: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers未安装")
        return
    
    # 3. 检查FlashAttention支持
    try:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLPreTrainedModel
        print("✅ Qwen2.5-VL模型支持FlashAttention")
    except ImportError as e:
        print(f"❌ 无法导入Qwen2.5-VL模型: {e}")
        return
    
    # 4. 测试模型初始化
    try:
        from models.qwen2_5_vl_classify import Qwen2_5_VLForImageClassification
        
        # 创建一个小模型进行测试
        model = Qwen2_5_VLForImageClassification(
            pretrained_model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            num_labels=10
        )
        
        # 检查attention实现
        if hasattr(model, 'config') and hasattr(model.config, '_attn_implementation'):
            attn_impl = model.config._attn_implementation
            print(f"✅ 模型attention实现: {attn_impl}")
            
            if attn_impl == "flash_attention_2":
                print("🎉 FlashAttention 2 已成功启用!")
            elif attn_impl == "flash_attention_1":
                print("🎉 FlashAttention 1 已成功启用!")
            else:
                print(f"ℹ️ 使用 {attn_impl} attention")
        else:
            print("⚠️ 无法检测attention实现")
            
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_flash_attention() 