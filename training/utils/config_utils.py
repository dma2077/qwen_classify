"""
训练配置工具函数
"""

def get_effective_steps_per_epoch(config, train_loader):
    """计算正确的有效训练步数每epoch
    
    Args:
        config: 训练配置
        train_loader: 训练数据加载器
        
    Returns:
        int: 有效训练步数每epoch
    """
    import json
    
    # 获取DeepSpeed配置
    deepspeed_config_path = config.get('deepspeed', '')
    if not deepspeed_config_path:
        raise ValueError("DeepSpeed配置文件路径未设置")
    
    try:
        with open(deepspeed_config_path, 'r') as f:
            deepspeed_config = json.load(f)
    except Exception as e:
        raise ValueError(f"DeepSpeed配置文件解析失败: {e}")
    
    # 获取关键参数
    train_batch_size = deepspeed_config.get('train_batch_size', 256)
    dataset_size = len(train_loader.dataset)
    
    # 基于总批次大小计算有效步数
    effective_steps_per_epoch = dataset_size // train_batch_size
    if dataset_size % train_batch_size != 0:
        effective_steps_per_epoch += 1  # 向上取整
    
    print(f"📊 步数计算: 数据集大小={dataset_size:,}, 总批次大小={train_batch_size}, 有效步数={effective_steps_per_epoch}")
    
    return effective_steps_per_epoch

def get_total_effective_steps(config, train_loader):
    """计算总的有效训练步数（所有epochs）
    
    Args:
        config: 训练配置
        train_loader: 训练数据加载器
        
    Returns:
        int: 总的有效训练步数
    """
    import json
    
    # 获取DeepSpeed配置
    deepspeed_config_path = config.get('deepspeed', '')
    if not deepspeed_config_path:
        raise ValueError("DeepSpeed配置文件路径未设置")
    
    try:
        with open(deepspeed_config_path, 'r') as f:
            deepspeed_config = json.load(f)
    except Exception as e:
        raise ValueError(f"DeepSpeed配置文件解析失败: {e}")
    
    # 获取关键参数
    train_batch_size = deepspeed_config.get('train_batch_size', 256)
    micro_batch_size_per_gpu = deepspeed_config.get('train_micro_batch_size_per_gpu', 8)
    gradient_accumulation_steps = deepspeed_config.get('gradient_accumulation_steps', 4)
    dataset_size = len(train_loader.dataset)
    
    # 🔍 调试信息：显示所有关键参数
    print(f"🔍 DeepSpeed配置调试:")
    print(f"  • train_batch_size: {train_batch_size}")
    print(f"  • micro_batch_size_per_gpu: {micro_batch_size_per_gpu}")
    print(f"  • gradient_accumulation_steps: {gradient_accumulation_steps}")
    print(f"  • dataset_size: {dataset_size}")
    print(f"  • len(train_loader): {len(train_loader)}")
    
    # 获取epochs数
    training_config = config['training']
    num_epochs = training_config.get('epochs') or training_config.get('num_epochs')
    if isinstance(num_epochs, str):
        num_epochs = int(num_epochs)
    
    # 🔥 关键修复：确保使用正确的有效批次大小
    # DeepSpeed的train_batch_size已经是全局有效批次大小
    # 不应该再乘以任何梯度累积因子
    effective_steps_per_epoch = dataset_size // train_batch_size
    if dataset_size % train_batch_size != 0:
        effective_steps_per_epoch += 1  # 向上取整
    
    # 计算总的有效步数
    total_effective_steps = effective_steps_per_epoch * num_epochs
    
    print(f"🔍 步数计算调试:")
    print(f"  • 计算公式: {dataset_size} ÷ {train_batch_size} = {effective_steps_per_epoch}")
    print(f"  • 每epoch有效步数: {effective_steps_per_epoch}")
    print(f"  • 总epochs: {num_epochs}")
    print(f"  • 总有效步数: {total_effective_steps}")
    
    # 🔍 验证计算：与DataLoader步数对比
    dataloader_steps = len(train_loader)
    expected_ratio = dataloader_steps // effective_steps_per_epoch
    print(f"🔍 验证信息:")
    print(f"  • DataLoader步数: {dataloader_steps}")
    print(f"  • 预期比率(应该等于gradient_accumulation_steps): {expected_ratio}")
    print(f"  • gradient_accumulation_steps: {gradient_accumulation_steps}")
    
    return total_effective_steps

def prepare_config(config):
    """准备配置参数"""
    # 检查必要的配置
    required_keys = ['model', 'training', 'data']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"配置中缺少必要的键: {key}")
    
    # 设置training节点下的默认值
    training_config = config['training']
    training_config.setdefault('logging_steps', 50)
    training_config.setdefault('save_steps', 1000) 
    training_config.setdefault('eval_steps', 1000)
    training_config.setdefault('save_hf_format', True)
    training_config.setdefault('save_deepspeed_format', True)
    
    # 🔥 修复：处理字段名映射
    if 'epochs' in training_config and 'num_epochs' not in training_config:
        training_config['num_epochs'] = training_config['epochs']
        print(f"📋 配置映射: epochs -> num_epochs = {training_config['num_epochs']}")
    
    if 'lr' in training_config and 'learning_rate' not in training_config:
        training_config['learning_rate'] = training_config['lr']
        print(f"📋 配置映射: lr -> learning_rate = {training_config['learning_rate']}")
    
    # 将常用的配置项提升到根层级，方便访问
    config['logging_steps'] = training_config['logging_steps']
    config['save_steps'] = training_config['save_steps']
    config['eval_steps'] = training_config['eval_steps']
    config['save_hf_format'] = training_config['save_hf_format']
    config['save_deepspeed_format'] = training_config['save_deepspeed_format']
    
    # 确保output_dir在根层级
    if 'output_dir' not in config and 'output_dir' in training_config:
        config['output_dir'] = training_config['output_dir']
    
    # 🔥 修复：处理DeepSpeed配置结构
    deepspeed_config = config.get('deepspeed')
    if deepspeed_config is None:
        raise ValueError("DeepSpeed配置未找到！")
    
    # 处理不同的DeepSpeed配置格式
    if isinstance(deepspeed_config, dict):
        # 如果是字典格式，提取config_file
        if 'config_file' in deepspeed_config:
            config['deepspeed'] = deepspeed_config['config_file']
            print(f"📋 DeepSpeed配置: 从字典格式提取 -> {deepspeed_config['config_file']}")
        else:
            raise ValueError("DeepSpeed配置字典中缺少config_file字段")
    elif isinstance(deepspeed_config, str):
        # 如果已经是字符串路径，保持不变
        print(f"📋 DeepSpeed配置: 直接使用路径 -> {deepspeed_config}")
    else:
        raise ValueError("DeepSpeed配置必须是文件路径字符串或包含config_file的字典")
    
    # 🔥 新增：验证WandB配置
    wandb_config = config.get('wandb', {})
    if wandb_config.get('enabled', False):
        print(f"📋 WandB配置: 已启用")
        print(f"   📊 项目: {wandb_config.get('project', 'qwen_classification')}")
        print(f"   🏃 运行名称: {wandb_config.get('run_name', 'auto-generated')}")
    else:
        print(f"📋 WandB配置: 禁用")
    
    import os
    # 验证DeepSpeed配置文件是否存在
    deepspeed_config_path = config['deepspeed']
    if not os.path.exists(deepspeed_config_path):
        raise FileNotFoundError(f"DeepSpeed配置文件不存在: {deepspeed_config_path}")
    
    return config 