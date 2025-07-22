from torch.optim import AdamW

def create_optimizer(model, config):
    """创建优化器"""
    # 从配置中获取参数，支持lr和learning_rate两种字段名
    training_config = config['training']
    lr = training_config.get('lr') or training_config.get('learning_rate')
    weight_decay = training_config['weight_decay']
    
    # 确保学习率是数字类型
    if isinstance(lr, str):
        lr = float(lr)
    if isinstance(weight_decay, str):
        weight_decay = float(weight_decay)
        
    no_decay = ["bias", "LayerNorm.weight"]
    
    # 分别收集参数，确保不为空
    decay_params = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad]
    no_decay_params = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad]
    
    grouped = []
    if decay_params:
        grouped.append({
            "params": decay_params,
            "lr": lr,
            "weight_decay": weight_decay,
        })
    if no_decay_params:
        grouped.append({
            "params": no_decay_params,
            "lr": lr,
            "weight_decay": 0.0,
        })
    
    # 如果没有找到任何参数组，使用所有可训练参数
    if not grouped:
        grouped = [{
            "params": [p for p in model.parameters() if p.requires_grad],
            "lr": lr,
            "weight_decay": weight_decay,
        }]
    
    optimizer = AdamW(grouped)
    
    return optimizer
