from torch.utils.data import DataLoader
from transformers import AutoProcessor
from .dataset import MyFoodDataset
from .collator import create_collate_fn
import json

def create_dataloaders(config):
    """创建训练和验证数据加载器"""
    # 从配置中获取参数
    train_jsonl = config['data']['train_jsonl']
    val_jsonl = config['data']['val_jsonl']
    pretrained_model_name = config['model']['pretrained_name']
    num_workers = config['training'].get('num_workers', 0)
    
    # 从DeepSpeed配置中读取批次大小
    if 'deepspeed' in config:
        if isinstance(config['deepspeed'], str):
            # 如果是文件路径，读取文件
            with open(config['deepspeed'], 'r') as f:
                deepspeed_config = json.load(f)
        else:
            # 如果是字典，直接使用
            deepspeed_config = config['deepspeed']
        
        # 使用micro_batch_size_per_gpu作为DataLoader的batch_size
        batch_size = deepspeed_config.get('train_micro_batch_size_per_gpu', 1)
    else:
        batch_size = config['training'].get('batch_size', 8)
    
    # 准备 processor
    processor = AutoProcessor.from_pretrained(pretrained_model_name)
    
    # 构造训练数据集
    train_dataset = MyFoodDataset(train_jsonl)
    train_collate_fn = create_collate_fn(processor)
    
    # 构造验证数据集
    val_dataset = MyFoodDataset(val_jsonl)
    val_collate_fn = create_collate_fn(processor)
    
    # 创建训练数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        pin_memory=True,
    )
    
    # 创建验证数据加载器
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collate_fn,
        pin_memory=True,
    )
    
    return train_loader, val_loader