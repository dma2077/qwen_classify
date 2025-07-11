from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor
from .dataset import MyFoodDataset
from .collator import create_collate_fn
import json
import torch.distributed as dist

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
    
    # 检查是否使用分布式训练
    use_distributed = dist.is_available() and dist.is_initialized()
    
    # 只在主进程中打印分布式信息
    is_main_process = not use_distributed or dist.get_rank() == 0
    
    if is_main_process:
        print(f"分布式检查:")
        print(f"  • dist.is_available(): {dist.is_available()}")
        print(f"  • dist.is_initialized(): {dist.is_initialized()}")
        print(f"  • 使用分布式训练: {use_distributed}")
    
    # 创建分布式采样器（如果使用分布式训练）
    if use_distributed:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        if is_main_process:
            print(f"  • 世界大小: {world_size}")
            print(f"  • 当前进程: {rank}")
        
        train_sampler = DistributedSampler(
            train_dataset,
            shuffle=True,
            drop_last=True  # 确保所有GPU处理相同数量的批次
        )
        val_sampler = DistributedSampler(
            val_dataset,
            shuffle=False,
            drop_last=False
        )
        shuffle_train = False  # 分布式采样器已经处理了shuffle
        shuffle_val = False
        
        if is_main_process:
            print(f"  • 每个GPU将处理训练样本数: {len(train_sampler)}")
            print(f"  • 每个GPU将处理验证样本数: {len(val_sampler)}")
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True
        shuffle_val = False
        if is_main_process:
            print(f"  • 未使用分布式采样器")
    
    # 创建训练数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        pin_memory=True,
        drop_last=True  # 确保批次大小一致
    )
    
    # 创建验证数据加载器
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle_val,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=val_collate_fn,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader