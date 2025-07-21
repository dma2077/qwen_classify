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
        
        # 🔥 修复batch size逻辑：
        # - 训练DataLoader使用：train_micro_batch_size_per_gpu（DeepSpeed会处理gradient accumulation）
        # - 评估时希望使用：micro_batch_size_per_gpu × num_gpus（gradient_accumulation_steps=1的等效）
        
        micro_batch_size_per_gpu = deepspeed_config.get('train_micro_batch_size_per_gpu', 1)
        total_batch_size = deepspeed_config.get('train_batch_size', 1)
        gradient_accumulation_steps = deepspeed_config.get('gradient_accumulation_steps', 1)
        # 训练DataLoader使用micro batch size（DeepSpeed会自动处理gradient accumulation）
        train_batch_size = micro_batch_size_per_gpu
        
        # 计算GPU数量：优先从分布式获取，否则从DeepSpeed配置反推
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            num_gpus = dist.get_world_size()
            print(f"🔧 从分布式环境获取GPU数量: {num_gpus}")
        else:
            # 如果分布式未初始化，从DeepSpeed配置反推
            # train_batch_size = micro_batch_size_per_gpu × num_gpus × gradient_accumulation_steps
            # 所以 num_gpus = train_batch_size / (micro_batch_size_per_gpu × gradient_accumulation_steps)
            calculated_num_gpus = total_batch_size // (micro_batch_size_per_gpu * gradient_accumulation_steps)
            num_gpus = max(1, calculated_num_gpus)  # 至少为1
            print(f"🔧 从DeepSpeed配置计算GPU数量: {num_gpus}")
            print(f"   计算公式: {total_batch_size} / ({micro_batch_size_per_gpu} × {gradient_accumulation_steps}) = {num_gpus}")
        
        # 评估batch size = micro_batch_size_per_gpu × num_gpus（相当于gradient_accumulation_steps=1时的总batch size）
        eval_batch_size = micro_batch_size_per_gpu * num_gpus
        
        # 验证计算是否正确
        expected_total_batch = micro_batch_size_per_gpu * num_gpus * gradient_accumulation_steps
        if expected_total_batch != total_batch_size:
            print(f"⚠️ batch size配置检查:")
            print(f"   micro_batch_size_per_gpu: {micro_batch_size_per_gpu}")
            print(f"   num_gpus: {num_gpus}")
            print(f"   gradient_accumulation_steps: {gradient_accumulation_steps}")
            print(f"   计算的总batch size: {expected_total_batch}")
            print(f"   配置的train_batch_size: {total_batch_size}")
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
    
    # 只在主进程中打印关键信息
    is_main_process = not use_distributed or dist.get_rank() == 0
    
    if is_main_process:
        # 打印批次大小配置信息
        if 'deepspeed' in config:
            print(f"📊 批次大小配置:")
            print(f"  • 训练批次大小: {train_batch_size} (per GPU)")
            print(f"  • 评估批次大小: {eval_batch_size} (total)")
        else:
            print(f"📊 批次大小配置: {batch_size}")
        
        # 打印数据集配置信息
        if dataset_configs:
            print(f"📊 数据集配置:")
            for dataset_name, config_info in dataset_configs.items():
                num_classes = config_info.get('num_classes', 'N/A')
                eval_ratio = config_info.get('eval_ratio', 'N/A')
                print(f"  • {dataset_name}: {num_classes} classes, eval_ratio={eval_ratio}")
    
    # 创建分布式采样器（如果使用分布式训练）
    if use_distributed:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
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
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True
        shuffle_val = False
    
    # 创建训练数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size if 'deepspeed' in config else batch_size,
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
        batch_size=eval_batch_size if 'deepspeed' in config else batch_size,
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
        # 🔥 修复：评估时使用总的有效批次大小，而不是每个GPU的批次大小
        batch_size = deepspeed_config.get('train_batch_size', 1)
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