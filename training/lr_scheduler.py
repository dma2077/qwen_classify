import math
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule_with_warmup
)
from torch.optim.lr_scheduler import LambdaLR

def _calculate_warmup_steps(warmup_config, total_training_steps):
    """计算warmup步数，支持绝对值和比例
    
    Args:
        warmup_config: warmup配置，可以是：
            - int/str: 绝对步数 (如 200)
            - float: 比例 (如 0.1 表示总步数的10%)
        total_training_steps: 总训练步数
        
    Returns:
        int: 实际的warmup步数
    """
    # 类型转换和处理
    if isinstance(warmup_config, str):
        try:
            # 尝试转换为float以支持字符串形式的比例 (如 "0.1")
            warmup_value = float(warmup_config)
        except ValueError:
            # 如果转换失败，当作整数处理
            warmup_value = int(warmup_config)
    else:
        warmup_value = warmup_config
    
    # 判断是比例还是绝对值
    if isinstance(warmup_value, float) and 0 < warmup_value < 1:
        # 比例形式：0 < value < 1
        num_warmup_steps = int(warmup_value * total_training_steps)
        warmup_type = f"比例 ({warmup_value:.1%})"
    elif isinstance(warmup_value, float) and warmup_value >= 1:
        # 大于等于1的float，当作绝对值处理
        num_warmup_steps = int(warmup_value)
        warmup_type = "绝对步数"
    else:
        # 整数形式：绝对步数
        num_warmup_steps = int(warmup_value)
        warmup_type = "绝对步数"
    
    # 确保warmup步数不超过总训练步数
    num_warmup_steps = min(num_warmup_steps, total_training_steps)
    
    return num_warmup_steps, warmup_type

def create_lr_scheduler(optimizer, config, total_effective_steps):
    """创建学习率调度器
    
    支持的调度器类型：
    - cosine: 余弦衰减调度器
    - cosine_with_hold: 余弦+平稳期调度器
    - linear: 线性衰减调度器  
    - polynomial: 多项式衰减调度器
    - exponential: 指数衰减调度器
    - constant: 常数调度器（warmup后保持不变）
    - cosine_restarts: 带重启的余弦调度器
    
    Args:
        optimizer: 优化器
        config: 训练配置
        total_effective_steps: 总的有效训练步数（已经考虑了epochs和批次大小）
    """
    # 从配置中获取参数
    training_config = config['training']
    lr_config = training_config.get('lr_scheduler', {})
    scheduler_type = lr_config.get('type', 'cosine')
    warmup_steps_config = training_config['warmup_steps']
    
    # 直接使用传入的总有效步数
    num_training_steps = total_effective_steps
    
    # 获取DeepSpeed配置用于显示信息
    deepspeed_config = config.get('deepspeed', {})
    if isinstance(deepspeed_config, str):
        import json
        with open(deepspeed_config, 'r') as f:
            deepspeed_config = json.load(f)
    
    # 获取DeepSpeed参数用于显示
    micro_batch_size_per_gpu = deepspeed_config.get('train_micro_batch_size_per_gpu', 1)
    gradient_accumulation_steps = deepspeed_config.get('gradient_accumulation_steps', 1)
    train_batch_size = deepspeed_config.get('train_batch_size', 32)
    
    # 处理warmup_steps：支持绝对值和比例
    num_warmup_steps, warmup_type = _calculate_warmup_steps(warmup_steps_config, num_training_steps)
    
    # 只在主进程中打印训练配置信息
    try:
        import torch.distributed as dist
        # 更可靠的主进程检查：只有在分布式训练中且rank为0的进程才是主进程
        is_main_process = not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
    except ImportError:
        is_main_process = True
    
    # 计算steps_per_epoch用于cosine_restarts调度器
    training_config = config['training']
    num_epochs = training_config.get('epochs') or training_config.get('num_epochs')
    if isinstance(num_epochs, str):
        num_epochs = int(num_epochs)
    steps_per_epoch = num_training_steps // num_epochs if num_epochs > 0 else num_training_steps
    
    # 根据调度器类型创建相应的调度器
    if scheduler_type == 'cosine':
        return create_cosine_scheduler(optimizer, lr_config, num_warmup_steps, num_training_steps, is_main_process)
    elif scheduler_type == 'cosine_with_hold':
        return create_cosine_with_hold_scheduler(optimizer, lr_config, num_warmup_steps, num_training_steps, is_main_process)
    elif scheduler_type == 'linear':
        return create_linear_scheduler(optimizer, lr_config, num_warmup_steps, num_training_steps, is_main_process)
    elif scheduler_type == 'polynomial':
        return create_polynomial_scheduler(optimizer, lr_config, num_warmup_steps, num_training_steps, is_main_process)
    elif scheduler_type == 'exponential':
        return create_exponential_scheduler(optimizer, lr_config, num_warmup_steps, num_training_steps, is_main_process)
    elif scheduler_type == 'constant':
        return create_constant_scheduler(optimizer, lr_config, num_warmup_steps, num_training_steps, is_main_process)
    elif scheduler_type == 'cosine_restarts':
        return create_cosine_restarts_scheduler(optimizer, lr_config, num_warmup_steps, num_training_steps, steps_per_epoch, is_main_process)
    else:
        raise ValueError(f"不支持的调度器类型: {scheduler_type}")


def create_cosine_scheduler(optimizer, lr_config, num_warmup_steps, num_training_steps, is_main_process):
    """创建余弦衰减调度器"""
    num_cycles = lr_config.get('num_cycles', 0.5)
    final_lr_ratio = lr_config.get('final_lr_ratio', 0.0)  # 最终学习率相对于初始学习率的比例
    
    # 如果需要自定义最终学习率比例，使用自定义实现
    if final_lr_ratio != 0.0:
        return create_custom_cosine_scheduler(optimizer, num_warmup_steps, num_training_steps, num_cycles, final_lr_ratio)
    else:
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
        )


def create_cosine_with_hold_scheduler(optimizer, lr_config, num_warmup_steps, num_training_steps, is_main_process):
    """创建余弦+平稳期调度器"""
    hold_steps = lr_config.get('hold_steps', None)
    hold_ratio = lr_config.get('hold_ratio', 0.3)  # 平稳期占总步数的比例
    final_lr_ratio = lr_config.get('final_lr_ratio', 0.0)
    num_cycles = lr_config.get('num_cycles', 0.5)
    
    # 如果没有指定hold_steps，则根据hold_ratio计算
    if hold_steps is None:
        total_non_warmup_steps = num_training_steps - num_warmup_steps
        hold_steps = int(total_non_warmup_steps * hold_ratio)
    
    # 计算各阶段步数
    decay_steps = num_training_steps - num_warmup_steps - hold_steps
    
    return create_custom_cosine_with_hold_scheduler(
        optimizer, num_warmup_steps, hold_steps, num_training_steps, num_cycles, final_lr_ratio
    )


def create_linear_scheduler(optimizer, lr_config, num_warmup_steps, num_training_steps, is_main_process):
    """创建线性衰减调度器"""
    final_lr_ratio = lr_config.get('final_lr_ratio', 0.0)
    
    if final_lr_ratio != 0.0:
        return create_custom_linear_scheduler(optimizer, num_warmup_steps, num_training_steps, final_lr_ratio)
    else:
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )


def create_polynomial_scheduler(optimizer, lr_config, num_warmup_steps, num_training_steps, is_main_process):
    """创建多项式衰减调度器"""
    power = lr_config.get('power', 1.0)
    final_lr_ratio = lr_config.get('final_lr_ratio', 0.0)
    
    if final_lr_ratio != 0.0:
        return create_custom_polynomial_scheduler(optimizer, num_warmup_steps, num_training_steps, power, final_lr_ratio)
    else:
        return get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            power=power,
        )


def create_exponential_scheduler(optimizer, lr_config, num_warmup_steps, num_training_steps, is_main_process):
    """创建指数衰减调度器"""
    decay_rate = lr_config.get('decay_rate', 0.95)
    final_lr_ratio = lr_config.get('final_lr_ratio', None)
    
    # 如果指定了最终学习率比例，计算对应的衰减率
    if final_lr_ratio is not None:
        # final_lr_ratio = decay_rate^(num_training_steps - num_warmup_steps)
        decay_steps = num_training_steps - num_warmup_steps
        if decay_steps > 0:
            decay_rate = final_lr_ratio ** (1.0 / decay_steps)
    
    return create_custom_exponential_scheduler(optimizer, num_warmup_steps, num_training_steps, decay_rate)


def create_constant_scheduler(optimizer, lr_config, num_warmup_steps, num_training_steps, is_main_process):
    """创建常数调度器（warmup后保持学习率不变）"""
    return get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
    )


def create_cosine_restarts_scheduler(optimizer, lr_config, num_warmup_steps, num_training_steps, steps_per_epoch, is_main_process):
    """创建带重启的余弦调度器"""
    restart_period_epochs = lr_config.get('restart_period_epochs', 2)
    restart_period_steps = restart_period_epochs * steps_per_epoch
    final_lr_ratio = lr_config.get('final_lr_ratio', 0.1)
    
    return create_custom_cosine_restarts_scheduler(optimizer, num_warmup_steps, num_training_steps, restart_period_steps, final_lr_ratio)


# 自定义调度器实现

def create_custom_cosine_scheduler(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, final_lr_ratio=0.0):
    """自定义余弦调度器，支持设置最终学习率比例"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        
        # 在final_lr_ratio和1.0之间进行余弦衰减
        return final_lr_ratio + (1.0 - final_lr_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)


def create_custom_cosine_with_hold_scheduler(optimizer, num_warmup_steps, hold_steps, num_training_steps, num_cycles=0.5, final_lr_ratio=0.0):
    """自定义余弦+平稳期调度器"""
    def lr_lambda(current_step):
        # 阶段1: Warmup - 线性增长到目标学习率
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # 阶段2: Hold - 保持在目标学习率
        if current_step < num_warmup_steps + hold_steps:
            return 1.0
        
        # 阶段3: Cosine Decay - 余弦衰减
        decay_start_step = num_warmup_steps + hold_steps
        decay_steps = num_training_steps - decay_start_step
        
        if decay_steps <= 0:
            return final_lr_ratio
        
        progress = float(current_step - decay_start_step) / float(decay_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        
        # 在final_lr_ratio和1.0之间进行余弦衰减
        return final_lr_ratio + (1.0 - final_lr_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)


def create_custom_linear_scheduler(optimizer, num_warmup_steps, num_training_steps, final_lr_ratio=0.0):
    """自定义线性调度器，支持设置最终学习率比例"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return final_lr_ratio + (1.0 - final_lr_ratio) * (1.0 - progress)
    
    return LambdaLR(optimizer, lr_lambda)


def create_custom_polynomial_scheduler(optimizer, num_warmup_steps, num_training_steps, power=1.0, final_lr_ratio=0.0):
    """自定义多项式调度器，支持设置最终学习率比例"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        decay = (1.0 - progress) ** power
        return final_lr_ratio + (1.0 - final_lr_ratio) * decay
    
    return LambdaLR(optimizer, lr_lambda)


def create_custom_exponential_scheduler(optimizer, num_warmup_steps, num_training_steps, decay_rate=0.95):
    """自定义指数调度器"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        decay_steps = current_step - num_warmup_steps
        return decay_rate ** decay_steps
    
    return LambdaLR(optimizer, lr_lambda)


def create_custom_cosine_restarts_scheduler(optimizer, num_warmup_steps, num_training_steps, restart_period, final_lr_ratio=0.1):
    """自定义带重启的余弦调度器"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # 重启逻辑
        effective_step = current_step - num_warmup_steps
        cycle_step = effective_step % restart_period
        progress = float(cycle_step) / float(max(1, restart_period))
        
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr_ratio + (1.0 - final_lr_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)
