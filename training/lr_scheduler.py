from transformers import get_cosine_schedule_with_warmup

def create_lr_scheduler(optimizer, config, steps_per_epoch):
    """创建学习率调度器"""
    # 从配置中获取参数
    num_warmup_steps = config['training']['warmup_steps']
    num_epochs = config['training']['num_epochs']
    
    # 类型检查和转换
    if isinstance(num_warmup_steps, str):
        num_warmup_steps = int(num_warmup_steps)
    if isinstance(num_epochs, str):
        num_epochs = int(num_epochs)
    
    # 计算有效训练步数（考虑DeepSpeed的分布式训练和梯度累积）
    # 获取DeepSpeed配置
    deepspeed_config = config.get('deepspeed', {})
    if isinstance(deepspeed_config, str):
        import json
        with open(deepspeed_config, 'r') as f:
            deepspeed_config = json.load(f)
    
    # 获取DeepSpeed参数
    micro_batch_size_per_gpu = deepspeed_config.get('train_micro_batch_size_per_gpu', 1)
    gradient_accumulation_steps = deepspeed_config.get('gradient_accumulation_steps', 1)
    train_batch_size = deepspeed_config.get('train_batch_size', 32)
    
    # 计算有效训练步数（基于DataLoader步数和梯度累积）
    effective_steps_per_epoch = steps_per_epoch // gradient_accumulation_steps
    num_training_steps = effective_steps_per_epoch * num_epochs
    
    print(f"每GPU微批次大小: {micro_batch_size_per_gpu}")
    print(f"梯度累积步数: {gradient_accumulation_steps}")
    print(f"总有效批次大小: {train_batch_size}")
    print(f"每GPU DataLoader步数: {steps_per_epoch:,}")
    print(f"有效训练步数每epoch: {effective_steps_per_epoch:,}")
    print(f"总有效训练步数: {num_training_steps:,}")
    
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=0.5,
    )
