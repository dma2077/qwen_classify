import torch
import time
from typing import Dict, Tuple
from tqdm import tqdm

def evaluate_model(model, val_loader, device) -> Tuple[float, float]:
    """评估模型性能 - 在分布式环境下正确聚合所有GPU的结果"""
    import torch.distributed as dist
    from training.utils.distributed import is_dist_initialized, get_rank
    
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # 只在主进程显示进度条
    show_progress = not is_dist_initialized() or get_rank() == 0
    eval_pbar = tqdm(val_loader, desc="Evaluating", leave=False, disable=not show_progress)
    
    batch_count = 0  # 用于计算平均损失
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_pbar):
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
                
                # 每10步或最后一步更新进度条显示
                if show_progress and (batch_idx % 10 == 0 or batch_idx == len(val_loader) - 1):
                    current_accuracy = correct / total if total > 0 else 0
                    current_avg_loss = total_loss / batch_count
                    eval_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{current_avg_loss:.4f}',
                        'accuracy': f'{current_accuracy:.4f}',
                        'samples': f'{total}'
                    })
                
            except Exception as e:
                if show_progress:
                    print(f"❌ 评估批次 {batch_idx} 出错: {e}")
                    eval_pbar.set_postfix({'error': f'batch_{batch_idx}_failed'})
                continue
    
    # 关闭进度条
    eval_pbar.close()
    
    # 在分布式环境下聚合所有GPU的结果
    if is_dist_initialized():
        # 转换为tensor进行聚合
        total_loss_tensor = torch.tensor(total_loss, dtype=torch.float32, device=device)
        correct_tensor = torch.tensor(correct, dtype=torch.long, device=device) 
        total_tensor = torch.tensor(total, dtype=torch.long, device=device)
        batch_count_tensor = torch.tensor(batch_count, dtype=torch.long, device=device)
        
        # 聚合所有GPU的结果
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(batch_count_tensor, op=dist.ReduceOp.SUM)
        
        # 计算全局平均值
        global_avg_loss = total_loss_tensor.item() / batch_count_tensor.item() if batch_count_tensor.item() > 0 else 0
        global_accuracy = correct_tensor.item() / total_tensor.item() if total_tensor.item() > 0 else 0
        
        # 只在主进程打印全局结果
        if get_rank() == 0:
            print("\n" + "="*80)
            print("📊 验证集评估结果 (全局聚合)")
            print("="*80)
            print(f"📈 Average Loss:     {global_avg_loss:.6f}")
            print(f"🎯 Accuracy:         {global_accuracy:.4f} ({global_accuracy*100:.2f}%)")
            print(f"📊 Total Samples:    {total_tensor.item():,}")
            print(f"✅ Correct Samples:  {correct_tensor.item():,}")
            print(f"❌ Wrong Samples:    {total_tensor.item() - correct_tensor.item():,}")
            print("="*80 + "\n")
        
        return global_avg_loss, global_accuracy
    else:
        # 单GPU模式
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        accuracy = correct / total if total > 0 else 0
        print("\n" + "="*80)
        print("📊 验证集评估结果")
        print("="*80)
        print(f"📈 Average Loss:     {avg_loss:.6f}")
        print(f"🎯 Accuracy:         {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"📊 Total Samples:    {total:,}")
        print(f"✅ Correct Samples:  {correct:,}")
        print(f"❌ Wrong Samples:    {total - correct:,}")
        print("="*80 + "\n")
        return avg_loss, accuracy 