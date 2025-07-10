from transformers import get_cosine_schedule_with_warmup

def create_lr_scheduler(optimizer, config, steps_per_epoch):
    """创建学习率调度器"""
    # 从配置中获取参数
    num_warmup_steps = config['training']['warmup_steps']
    num_epochs = config['training']['num_epochs']
    
    # 计算总训练步数
    num_training_steps = steps_per_epoch * num_epochs
    
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=0.5,
    )
