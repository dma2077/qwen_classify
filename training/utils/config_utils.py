"""
训练配置工具函数
"""

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
    
    if 'lr' in training_config and 'learning_rate' not in training_config:
        training_config['learning_rate'] = training_config['lr']
    
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
    
    import os
    # 验证DeepSpeed配置文件是否存在
    deepspeed_config_path = config['deepspeed']
    if not os.path.exists(deepspeed_config_path):
        raise FileNotFoundError(f"DeepSpeed配置文件不存在: {deepspeed_config_path}")
    
    return config 