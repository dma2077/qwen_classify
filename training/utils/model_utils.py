import os
from transformers import AutoProcessor

def save_hf_model(model, config, output_dir):
    """Save model in HuggingFace format"""
    try:
        # Get the actual model from DeepSpeed wrapper
        if hasattr(model, 'module'):
            actual_model = model.module
        else:
            actual_model = model
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model directly to output_dir
        actual_model.save_pretrained(output_dir)
        
        # Save processor
        processor = AutoProcessor.from_pretrained(config["model"]["pretrained_name"])
        processor.save_pretrained(output_dir)
        
        print(f"HF model saved to: {output_dir}")
        return output_dir
    except Exception as e:
        print(f"Failed to save HF model: {e}")
        return None 