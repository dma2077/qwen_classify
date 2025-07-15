from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor
from .dataset import MyFoodDataset, MultiDatasetLoader
from .collator import create_collate_fn
import json
import torch.distributed as dist

def create_dataloaders(config):
    """创建训练和验证数据加载器，支持多数据集配置"""
    # 从配置中获取参数
    pretrained_model_name = config['model']['pretrained_name']
    num_workers = config['training'].get('num_workers', 0)
    
    # 获取多数据集配置
    dataset_configs = config.get('datasets', {}).get('dataset_configs', {})
    shuffle_datasets = config.get('datasets', {}).get('shuffle_datasets', True)
    
    # 准备评估比例字典
    eval_ratios = {}
    for dataset_name, dataset_config in dataset_configs.items():
        eval_ratios[dataset_name] = dataset_config.get('eval_ratio', 0.2)
    
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
    
    # 判断使用单文件还是多文件模式
    data_config = config.get('data', {})
    
    if 'train_jsonl_list' in data_config and 'val_jsonl_list' in data_config:
        # 多文件模式
        train_jsonl_list = data_config['train_jsonl_list']
        val_jsonl_list = data_config['val_jsonl_list']
        
        # 获取评估配置
        eval_config = config.get('training', {}).get('evaluation', {})
        use_partial_eval = eval_config.get('partial_eval_during_training', True)
        
        # 构造训练数据集
        train_dataset = MultiDatasetLoader(
            jsonl_file_list=train_jsonl_list,
            dataset_configs=dataset_configs,
            shuffle_datasets=shuffle_datasets,
            eval_ratios=eval_ratios,
            is_eval=False,
            use_partial_eval=False  # 训练时总是使用全部数据
        )
        
        # 构造验证数据集（使用部分评估）
        val_dataset = MultiDatasetLoader(
            jsonl_file_list=val_jsonl_list,
            dataset_configs=dataset_configs,
            shuffle_datasets=False,  # 验证时不shuffle
            eval_ratios=eval_ratios,
            is_eval=True,
            use_partial_eval=use_partial_eval
        )
        
        # 保存原始文件列表，用于完整评估
        val_dataset._original_file_list = val_jsonl_list
        
    else:
        # 单文件模式（向后兼容）
        train_jsonl = data_config.get('train_jsonl')
        val_jsonl = data_config.get('val_jsonl')
        
        if not train_jsonl or not val_jsonl:
            raise ValueError("请在配置中提供 train_jsonl_list/val_jsonl_list 或 train_jsonl/val_jsonl")
        
        # 构造训练数据集，传递数据集配置
        train_dataset = MyFoodDataset(train_jsonl, dataset_configs=dataset_configs)
        
        # 构造验证数据集，传递数据集配置
        val_dataset = MyFoodDataset(val_jsonl, dataset_configs=dataset_configs)
    
    train_collate_fn = create_collate_fn(processor)
    val_collate_fn = create_collate_fn(processor)
    
    # 检查是否使用分布式训练
    use_distributed = dist.is_available() and dist.is_initialized()
    
    # 只在主进程中打印分布式信息
    is_main_process = not use_distributed or dist.get_rank() == 0
    
    if is_main_process:
        print(f"\n分布式检查:")
        print(f"  • dist.is_available(): {dist.is_available()}")
        print(f"  • dist.is_initialized(): {dist.is_initialized()}")
        print(f"  • 使用分布式训练: {use_distributed}")
        
        # 打印数据集配置信息
        if dataset_configs:
            print(f"\n📊 数据集配置:")
            for dataset_name, config_info in dataset_configs.items():
                num_classes = config_info.get('num_classes', 'N/A')
                eval_ratio = config_info.get('eval_ratio', 'N/A')
                description = config_info.get('description', 'No description')
                print(f"  • {dataset_name}: {num_classes} classes, eval_ratio={eval_ratio} - {description}")
        
        # 打印评估配置
        if 'train_jsonl_list' in data_config:
            eval_config = config.get('training', {}).get('evaluation', {})
            print(f"\n🔍 评估配置:")
            print(f"  • 训练过程中部分评估: {eval_config.get('partial_eval_during_training', True)}")
            print(f"  • 训练结束后完整评估: {eval_config.get('full_eval_at_end', True)}")
            print(f"  • 仅评估最佳模型: {eval_config.get('eval_best_model_only', True)}")
    
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

def create_full_eval_dataloader(config, model_processor=None):
    """创建完整评估数据加载器（用于训练结束后的完整评估）"""
    # 获取配置
    data_config = config.get('data', {})
    dataset_configs = config.get('datasets', {}).get('dataset_configs', {})
    
    if 'val_jsonl_list' not in data_config:
        return None
    
    val_jsonl_list = data_config['val_jsonl_list']
    
    # 准备评估比例字典（完整评估时设为1.0）
    eval_ratios = {name: 1.0 for name in dataset_configs.keys()}
    
    # 创建完整验证数据集
    full_val_dataset = MultiDatasetLoader(
        jsonl_file_list=val_jsonl_list,
        dataset_configs=dataset_configs,
        shuffle_datasets=False,  # 完整评估时不shuffle
        eval_ratios=eval_ratios,
        is_eval=True,
        use_partial_eval=False  # 使用完整数据
    )
    
    # 获取processor
    if model_processor is None:
        pretrained_model_name = config['model']['pretrained_name']
        processor = AutoProcessor.from_pretrained(pretrained_model_name)
    else:
        processor = model_processor
    
    val_collate_fn = create_collate_fn(processor)
    
    # 获取批次大小
    if 'deepspeed' in config:
        if isinstance(config['deepspeed'], str):
            with open(config['deepspeed'], 'r') as f:
                deepspeed_config = json.load(f)
        else:
            deepspeed_config = config['deepspeed']
        batch_size = deepspeed_config.get('train_micro_batch_size_per_gpu', 1)
    else:
        batch_size = config['training'].get('batch_size', 8)
    
    # 检查分布式设置
    use_distributed = dist.is_available() and dist.is_initialized()
    
    if use_distributed:
        val_sampler = DistributedSampler(
            full_val_dataset,
            shuffle=False,
            drop_last=False
        )
        shuffle_val = False
    else:
        val_sampler = None
        shuffle_val = False
    
    # 创建完整评估数据加载器
    full_val_loader = DataLoader(
        full_val_dataset,
        batch_size=batch_size,
        shuffle=shuffle_val,
        sampler=val_sampler,
        num_workers=config['training'].get('num_workers', 0),
        collate_fn=val_collate_fn,
        pin_memory=True,
        drop_last=False
    )
    
    return full_val_loader