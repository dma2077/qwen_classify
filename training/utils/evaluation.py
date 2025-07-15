import torch
import time
from typing import Dict, Tuple
from tqdm import tqdm
from collections import defaultdict

def evaluate_model(model, val_loader, device) -> Tuple[float, float]:
    """评估模型性能 - 在分布式环境下正确聚合所有GPU的结果"""
    import torch.distributed as dist
    
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # 检查分布式状态
    is_distributed = dist.is_available() and dist.is_initialized()
    if is_distributed:
        current_rank = dist.get_rank()
    else:
        current_rank = 0
    
    # 只在主进程显示进度条
    show_progress = not is_distributed or current_rank == 0
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
    if is_distributed:
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
        if current_rank == 0:
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

def evaluate_multi_dataset(model, val_loader, device, dataset_configs=None) -> Dict:
    """
    多数据集评估函数，支持按数据集分别统计指标
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        device: 设备
        dataset_configs: 数据集配置字典
        
    Returns:
        evaluation results dictionary containing overall and per-dataset metrics
    """
    import torch.distributed as dist
    
    model.eval()
    
    # 检查分布式状态
    is_distributed = dist.is_available() and dist.is_initialized()
    if is_distributed:
        current_rank = dist.get_rank()
    else:
        current_rank = 0
    
    # 只在主进程显示进度条
    show_progress = not is_distributed or current_rank == 0
    eval_pbar = tqdm(val_loader, desc="Multi-Dataset Evaluation", leave=False, disable=not show_progress)
    
    # 整体统计
    total_loss = 0
    correct = 0
    total = 0
    batch_count = 0
    
    # 按数据集统计
    dataset_stats = defaultdict(lambda: {
        'total_loss': 0.0,
        'correct': 0,
        'total': 0,
        'batch_count': 0
    })
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_pbar):
            try:
                batch_count += 1
                
                # 移动数据到设备
                inputs = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                
                # 获取数据集信息
                dataset_names = batch.get("dataset_names", [])
                num_classes_list = batch.get("num_classes_list", [])
                
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
                
                # 添加多数据集支持的参数
                if dataset_names:
                    forward_kwargs["dataset_names"] = dataset_names
                if num_classes_list:
                    forward_kwargs["num_classes_list"] = num_classes_list
                
                outputs = model(**forward_kwargs)
                
                # 计算损失和预测
                loss = outputs.loss
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                # 更新整体统计
                total_loss += loss.item()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                # 更新各数据集统计
                for i, dataset_name in enumerate(dataset_names):
                    if i >= len(labels) or i >= len(predictions):
                        continue
                        
                    label = labels[i].item()
                    pred = predictions[i].item()
                    
                    dataset_stats[dataset_name]['total_loss'] += loss.item() / len(dataset_names)
                    dataset_stats[dataset_name]['total'] += 1
                    dataset_stats[dataset_name]['batch_count'] += 1
                    
                    if pred == label:
                        dataset_stats[dataset_name]['correct'] += 1
                
                # 每10步或最后一步更新进度条显示
                if show_progress and (batch_idx % 10 == 0 or batch_idx == len(val_loader) - 1):
                    current_accuracy = correct / total if total > 0 else 0
                    current_avg_loss = total_loss / batch_count
                    eval_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{current_avg_loss:.4f}',
                        'accuracy': f'{current_accuracy:.4f}',
                        'datasets': f'{len(dataset_stats)}'
                    })
                
            except Exception as e:
                if show_progress:
                    print(f"❌ 多数据集评估批次 {batch_idx} 出错: {e}")
                    eval_pbar.set_postfix({'error': f'batch_{batch_idx}_failed'})
                continue
    
    # 关闭进度条
    eval_pbar.close()
    
    # 计算结果
    overall_avg_loss = total_loss / batch_count if batch_count > 0 else 0
    overall_accuracy = correct / total if total > 0 else 0
    
    # 计算各数据集的指标
    dataset_metrics = {}
    for dataset_name, stats in dataset_stats.items():
        if stats['total'] > 0:
            avg_loss = stats['total_loss'] / stats['batch_count'] if stats['batch_count'] > 0 else 0
            accuracy = stats['correct'] / stats['total']
            dataset_metrics[dataset_name] = {
                'loss': avg_loss,
                'accuracy': accuracy,
                'samples': stats['total'],
                'correct': stats['correct']
            }
    
    # 在分布式环境下聚合结果
    if is_distributed:
        # TODO: 添加分布式聚合逻辑
        # 这里可以添加类似单数据集评估的分布式聚合代码
        pass
    
    # 输出结果
    if show_progress:
        print("\n" + "="*80)
        print("📊 多数据集评估结果")
        print("="*80)
        print(f"📈 Overall Loss:     {overall_avg_loss:.6f}")
        print(f"🎯 Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        print(f"📊 Total Samples:    {total:,}")
        print(f"✅ Total Correct:    {correct:,}")
        print()
        
        for dataset_name, metrics in dataset_metrics.items():
            print(f"📂 {dataset_name}:")
            print(f"  • Loss:     {metrics['loss']:.6f}")
            print(f"  • Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print(f"  • Samples:  {metrics['samples']:,} (Correct: {metrics['correct']:,})")
            
        print("="*80 + "\n")
    
    return {
        'overall_loss': overall_avg_loss,
        'overall_accuracy': overall_accuracy,
        'dataset_metrics': dataset_metrics,
        'total_samples': total,
        'total_correct': correct
    } 