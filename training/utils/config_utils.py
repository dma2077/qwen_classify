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
    
    # 验证DeepSpeed配置（现在总是通过命令行传入）
    if 'deepspeed' not in config:
        raise ValueError("DeepSpeed配置未找到！请使用--deepspeed_config参数指定配置文件")
    
    deepspeed_config = config['deepspeed']
    if not isinstance(deepspeed_config, str):
        raise ValueError("DeepSpeed配置必须是文件路径字符串")
    
    import os
    if not os.path.exists(deepspeed_config):
        raise FileNotFoundError(f"DeepSpeed配置文件不存在: {deepspeed_config}")
    
    return config 