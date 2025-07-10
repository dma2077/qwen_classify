#!/usr/bin/env python3
"""
示例：如何加载训练过程中保存的checkpoint
"""

import os
import sys
import json
from transformers import AutoModel, AutoProcessor
from PIL import Image

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_checkpoint_info(checkpoint_dir):
    """加载checkpoint的训练信息"""
    info_path = os.path.join(checkpoint_dir, "training_info.json")
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            return json.load(f)
    return None


def list_available_checkpoints(output_dir):
    """列出所有可用的checkpoint"""
    checkpoints = []
    if not os.path.exists(output_dir):
        return checkpoints
    
    for item in os.listdir(output_dir):
        if item.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, item)):
            checkpoint_dir = os.path.join(output_dir, item)
            # 检查是否包含HF格式文件
            if os.path.exists(os.path.join(checkpoint_dir, "config.json")):
                step = int(item.split("-")[1])
                checkpoints.append((step, checkpoint_dir))
    
    return sorted(checkpoints)


def load_model_from_checkpoint(checkpoint_dir):
    """从checkpoint加载模型"""
    print(f"Loading model from: {checkpoint_dir}")
    
    # 检查是否存在HF格式文件
    if not os.path.exists(os.path.join(checkpoint_dir, "config.json")):
        raise FileNotFoundError(f"No HuggingFace format model found in {checkpoint_dir}")
    
    # 加载模型和processor
    model = AutoModel.from_pretrained(checkpoint_dir, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(checkpoint_dir)
    
    # 加载训练信息
    info = load_checkpoint_info(checkpoint_dir)
    if info:
        print(f"Training info:")
        print(f"  Step: {info['step']}")
        print(f"  Total samples: {info['total_samples']}")
        print(f"  Average loss: {info['avg_loss']:.4f}")
    
    return model, processor


def demo_inference(model, processor, image_path=None):
    """演示推理过程"""
    if image_path and os.path.exists(image_path):
        # 使用提供的图像
        image = Image.open(image_path).convert("RGB")
        print(f"Using image: {image_path}")
    else:
        # 创建一个示例图像（随机图像）
        image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        print("Using dummy image for demonstration")
    
    # 准备输入
    text = "This is an image of food, what dish is it?"
    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt"
    )
    
    # 推理
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax(dim=-1).item()
    
    print(f"Text input: {text}")
    print(f"Predicted class: {predicted_class}")
    print(f"Logits shape: {logits.shape}")
    
    return predicted_class


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and test checkpoints")
    parser.add_argument('--output_dir', type=str, default="outputs",
                       help='Training output directory')
    parser.add_argument('--checkpoint_step', type=int, default=None,
                       help='Specific checkpoint step to load (default: latest)')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to test image')
    parser.add_argument('--list_only', action='store_true',
                       help='Only list available checkpoints')
    
    args = parser.parse_args()
    
    # 列出可用的checkpoints
    checkpoints = list_available_checkpoints(args.output_dir)
    
    if not checkpoints:
        print(f"No checkpoints found in {args.output_dir}")
        return
    
    print(f"Available checkpoints in {args.output_dir}:")
    for step, checkpoint_dir in checkpoints:
        info = load_checkpoint_info(checkpoint_dir)
        if info:
            print(f"  Step {step}: avg_loss={info['avg_loss']:.4f}, samples={info['total_samples']}")
        else:
            print(f"  Step {step}: (no training info)")
    
    if args.list_only:
        return
    
    # 选择要加载的checkpoint
    if args.checkpoint_step:
        # 加载指定步数的checkpoint
        target_checkpoint = None
        for step, checkpoint_dir in checkpoints:
            if step == args.checkpoint_step:
                target_checkpoint = checkpoint_dir
                break
        
        if not target_checkpoint:
            print(f"Checkpoint at step {args.checkpoint_step} not found")
            return
    else:
        # 加载最新的checkpoint
        target_checkpoint = checkpoints[-1][1]
    
    # 加载模型
    try:
        model, processor = load_model_from_checkpoint(target_checkpoint)
        print("Model loaded successfully!")
        
        # 演示推理
        print("\nRunning inference demo...")
        demo_inference(model, processor, args.image_path)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import torch
    main() 