import torch
import time
from typing import Dict, Tuple
from tqdm import tqdm

def evaluate_model(model, val_loader, device) -> Tuple[float, float]:
    """评估模型性能"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # 创建评估进度条
    eval_pbar = tqdm(val_loader, desc="Evaluating", leave=False)
    
    batch_count = 0  # 用于计算平均损失
    
    with torch.no_grad():
        for batch in eval_pbar:
            try:
                batch_count += 1
                
                # 移动数据到设备
                inputs = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                
                # 前向传播
                forward_kwargs = {
                    "input_ids": inputs,
                    "attention_mask": attention_mask,
                    "pixel_values": pixel_values,
                    "labels": labels
                }
                
                # 检查并添加image_grid_thw参数
                if "image_grid_thw" in batch:
                    forward_kwargs["image_grid_thw"] = batch["image_grid_thw"].to(device)
                
                outputs = model(**forward_kwargs)
                
                # 计算损失
                loss = outputs.loss
                total_loss += loss.item()
                
                # 计算准确率
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                # 更新进度条显示
                current_accuracy = correct / total if total > 0 else 0
                current_avg_loss = total_loss / batch_count
                eval_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{current_avg_loss:.4f}',
                    'accuracy': f'{current_accuracy:.4f}'
                })
                
            except Exception as e:
                print(f"评估批次时出错: {e}")
                eval_pbar.set_postfix({'error': 'batch_failed'})
                continue
    
    # 关闭进度条
    eval_pbar.close()
    
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy 