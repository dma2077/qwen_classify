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