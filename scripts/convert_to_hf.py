#!/usr/bin/env python3
"""
Convert DeepSpeed checkpoint to HuggingFace format.
This script loads a DeepSpeed checkpoint and converts it to standard HuggingFace format.
"""

import os
import sys
import argparse
import torch
import json
import shutil
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.qwen2_5_vl_classify import Qwen2_5_VLForImageClassification
from training.model import load_config
from transformers import AutoProcessor


def find_checkpoint_files(checkpoint_dir):
    """Find all possible checkpoint files in the directory"""
    checkpoint_files = []
    
    # Common checkpoint file patterns
    patterns = [
        "pytorch_model.bin",
        "model.pt",
        "model.pth",
        "checkpoint.pt",
        "checkpoint.pth"
    ]
    
    for pattern in patterns:
        path = os.path.join(checkpoint_dir, pattern)
        if os.path.exists(path):
            checkpoint_files.append(path)
    
    # Also check for DeepSpeed specific files
    ds_patterns = [
        "zero_pp_rank_0_mp_rank_00_model_states.pt",
        "mp_rank_00_model_states.pt"
    ]
    
    for pattern in ds_patterns:
        path = os.path.join(checkpoint_dir, pattern)
        if os.path.exists(path):
            checkpoint_files.append(path)
    
    # Find any .bin or .pth files
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.bin') or file.endswith('.pth'):
            full_path = os.path.join(checkpoint_dir, file)
            if full_path not in checkpoint_files:
                checkpoint_files.append(full_path)
    
    return checkpoint_files


def load_model_state_dict(checkpoint_path):
    """Load model state dict from checkpoint file"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # If it's a dict, try to find the model state dict
            if 'model' in checkpoint:
                return checkpoint['model']
            elif 'state_dict' in checkpoint:
                return checkpoint['state_dict']
            elif 'module' in checkpoint:
                return checkpoint['module']
            else:
                # Assume the checkpoint itself is the state dict
                return checkpoint
        else:
            # If it's not a dict, assume it's the model itself
            return checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None


def clean_state_dict_keys(state_dict):
    """Clean state dict keys to match HuggingFace format"""
    cleaned_state_dict = {}
    
    for key, value in state_dict.items():
        # Remove common prefixes that might be added during training
        cleaned_key = key
        
        # Remove module prefix (common in DataParallel/DistributedDataParallel)
        if cleaned_key.startswith('module.'):
            cleaned_key = cleaned_key[7:]
        
        # Remove _orig_mod prefix (common in torch.compile)
        if cleaned_key.startswith('_orig_mod.'):
            cleaned_key = cleaned_key[10:]
        
        cleaned_state_dict[cleaned_key] = value
    
    return cleaned_state_dict


def convert_checkpoint_to_hf(checkpoint_dir, output_dir, config_file="configs/config.yaml"):
    """Convert checkpoint to HuggingFace format"""
    
    # Load training configuration
    config = load_config(config_file)
    print(f"Loaded config: {config}")
    
    # Create model instance
    print("Creating model instance...")
    model = Qwen2_5_VLForImageClassification(
        pretrained_model_name=config["model"]["pretrained_name"],
        num_labels=config["model"]["num_labels"]
    )
    
    # Find checkpoint files
    checkpoint_files = find_checkpoint_files(checkpoint_dir)
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    print(f"Found checkpoint files: {checkpoint_files}")
    
    # Try to load from each checkpoint file
    loaded = False
    for checkpoint_path in checkpoint_files:
        print(f"\nTrying to load from: {checkpoint_path}")
        
        state_dict = load_model_state_dict(checkpoint_path)
        if state_dict is None:
            continue
        
        # Clean state dict keys
        state_dict = clean_state_dict_keys(state_dict)
        
        try:
            # Try to load the state dict
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
            
            print(f"Successfully loaded model from {checkpoint_path}")
            loaded = True
            break
            
        except Exception as e:
            print(f"Failed to load state dict: {e}")
            continue
    
    if not loaded:
        raise RuntimeError("Failed to load model from any checkpoint file")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model in HuggingFace format
    print(f"\nSaving model to HuggingFace format: {output_dir}")
    model.save_pretrained(output_dir)
    
    # Also save the processor
    print("Saving processor...")
    processor = AutoProcessor.from_pretrained(config["model"]["pretrained_name"])
    processor.save_pretrained(output_dir)
    
    # Copy training config for reference
    if os.path.exists(config_file):
        shutil.copy2(config_file, os.path.join(output_dir, "training_config.yaml"))
    
    print(f"\nModel successfully converted and saved to: {output_dir}")
    print(f"Model files:")
    for file in sorted(os.listdir(output_dir)):
        print(f"  - {file}")


def test_converted_model(model_dir):
    """Test the converted model to ensure it works"""
    print(f"\nTesting converted model from: {model_dir}")
    
    try:
        # Load the model
        from transformers import AutoModel, AutoProcessor
        
        model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_dir)
        
        print("✓ Model loaded successfully")
        print(f"✓ Model type: {type(model)}")
        print(f"✓ Model config: {model.config}")
        print(f"✓ Processor type: {type(processor)}")
        
        # Test model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing model: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert DeepSpeed checkpoint to HuggingFace format")
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Path to DeepSpeed checkpoint directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for HuggingFace model')
    parser.add_argument('--config_file', type=str, default="configs/config.yaml",
                       help='Path to training configuration file')
    parser.add_argument('--test', action='store_true',
                       help='Test the converted model after conversion')
    
    args = parser.parse_args()
    
    # Check if checkpoint directory exists
    if not os.path.exists(args.checkpoint_dir):
        print(f"Error: Checkpoint directory {args.checkpoint_dir} does not exist!")
        sys.exit(1)
    
    # Check if config file exists
    if not os.path.exists(args.config_file):
        print(f"Error: Config file {args.config_file} does not exist!")
        sys.exit(1)
    
    # Convert checkpoint
    try:
        convert_checkpoint_to_hf(args.checkpoint_dir, args.output_dir, args.config_file)
        
        print("\n" + "="*50)
        print("Conversion completed successfully!")
        print("="*50)
        
        if args.test:
            print("\nTesting converted model...")
            if test_converted_model(args.output_dir):
                print("\n✓ Model test passed!")
            else:
                print("\n✗ Model test failed!")
                sys.exit(1)
        
        print(f"\nYou can now use the model with:")
        print(f"```python")
        print(f"from transformers import AutoModel, AutoProcessor")
        print(f"model = AutoModel.from_pretrained('{args.output_dir}', trust_remote_code=True)")
        print(f"processor = AutoProcessor.from_pretrained('{args.output_dir}')")
        print(f"```")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 