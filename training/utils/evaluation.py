import torch
import time
from typing import Dict, Tuple

def evaluate_model(model, val_loader, device) -> Tuple[float, float]:
    """评估模型性能"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            try:
                # 移动数据到设备
                inputs = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                
                # 前向传播
                outputs = model(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels
                )
                
                # 计算损失
                loss = outputs.loss
                total_loss += loss.item()
                
                # 计算准确率
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
            except Exception as e:
                print(f"评估批次时出错: {e}")
                continue
    
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy

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
    
    # 参数名称映射和标准化
    print(f"原始训练配置: {training_config}")
    
    if 'epochs' in training_config and 'num_epochs' not in training_config:
        training_config['num_epochs'] = training_config['epochs']
        print(f"映射 epochs -> num_epochs: {training_config['num_epochs']}")
    
    if 'lr' in training_config and 'learning_rate' not in training_config:
        training_config['learning_rate'] = training_config['lr']
        print(f"映射 lr -> learning_rate: {training_config['learning_rate']}")
    
    # 将常用的配置项提升到根层级，方便访问
    config['logging_steps'] = training_config['logging_steps']
    config['save_steps'] = training_config['save_steps']
    config['eval_steps'] = training_config['eval_steps']
    config['save_hf_format'] = training_config['save_hf_format']
    config['save_deepspeed_format'] = training_config['save_deepspeed_format']
    
    # 确保output_dir在根层级
    if 'output_dir' not in config and 'output_dir' in training_config:
        config['output_dir'] = training_config['output_dir']
    
    print(f"处理后的训练配置: {config['training']}")
    
    return config 