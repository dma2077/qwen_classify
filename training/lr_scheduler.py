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
    
    # 计算总训练步数
    num_training_steps = steps_per_epoch * num_epochs
    
    print(f"学习率调度器参数: warmup_steps={num_warmup_steps}, num_epochs={num_epochs}, steps_per_epoch={steps_per_epoch}")
    print(f"总训练步数: {num_training_steps}")
    
    # 检查优化器参数组
    print(f"优化器参数组数量: {len(optimizer.param_groups)}")
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"参数组 {i}: lr={param_group['lr']} (type: {type(param_group['lr'])})")
    
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=0.5,
    )
