from torch.optim import AdamW

def create_optimizer(model, lr, weight_decay):
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
    
    return AdamW(grouped)
