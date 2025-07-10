#!/usr/bin/env python3
"""
检查 Qwen2.5-VL processor 的输出
"""
import os
import sys
import yaml
from PIL import Image
import torch
from transformers import AutoProcessor

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def test_processor():
    # 加载配置
    config = load_config()
    pretrained_model_name = config['model']['pretrained_name']
    
    # 创建processor
    processor = AutoProcessor.from_pretrained(pretrained_model_name)
    
    # 创建测试图像
    test_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
    
    # 创建测试消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is in this image?"},
            ],
        }
    ]
    
    # 处理消息
    text = processor.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    print("Generated text:")
    print(text)
    print("\n" + "="*50 + "\n")
    
    # 使用processor处理图像和文本
    inputs = processor(
        text=[text],
        images=[test_image],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    
    # 检查输出
    print("Processor output keys:")
    for key in inputs.keys():
        print(f"  {key}: {inputs[key].shape if hasattr(inputs[key], 'shape') else type(inputs[key])}")
    
    # 检查是否包含image_grid_thw
    if "image_grid_thw" in inputs:
        print(f"\nimage_grid_thw found:")
        print(f"  shape: {inputs['image_grid_thw'].shape}")
        print(f"  values: {inputs['image_grid_thw']}")
    else:
        print("\nimage_grid_thw NOT found in processor output!")
        print("Available keys:", list(inputs.keys()))

if __name__ == "__main__":
    test_processor() 