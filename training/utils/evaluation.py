import torch
import time
from typing import Dict, Tuple
from tqdm import tqdm
from collections import defaultdict
from .distributed import safe_all_reduce, safe_barrier, batch_all_reduce

def evaluate_model(model, val_loader, device) -> Tuple[float, float]:
    """评估模型性能 - 在分布式环境下正确聚合所有GPU的结果"""
    import torch.distributed as dist
    
    # 🔥 确保模型处于评估模式 - 兼容DeepSpeed包装
    model.eval()
    if hasattr(model, 'module'):
        model.module.eval()
        print(f"🔍 设置DeepSpeed包装模型为eval模式: model.training={model.training}, model.module.training={model.module.training}")
    else:
        print(f"🔍 设置模型为eval模式: model.training={model.training}")
    
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
    
    batch_count = 0
    
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
        
        # 批量聚合所有tensor，提高效率
        tensors_to_reduce = [total_loss_tensor, correct_tensor, total_tensor, batch_count_tensor]
        if not batch_all_reduce(tensors_to_reduce, op=dist.ReduceOp.SUM):
            print("⚠️  批量聚合失败，使用本地结果")
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            accuracy = correct / total if total > 0 else 0
            return avg_loss, accuracy
        
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
    """
    import torch.distributed as dist
    
    # 🔥 确保模型处于评估模式 - 兼容DeepSpeed包装
    model.eval()
    # 检查分布式状态
    is_distributed = dist.is_available() and dist.is_initialized()
    if is_distributed:
        current_rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        current_rank = 0
        world_size = 1
    
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
    
    # 在分布式环境下聚合结果
    if is_distributed:
        # 聚合整体统计
        total_loss_tensor = torch.tensor(total_loss, dtype=torch.float32, device=device)
        correct_tensor = torch.tensor(correct, dtype=torch.long, device=device) 
        total_tensor = torch.tensor(total, dtype=torch.long, device=device)
        batch_count_tensor = torch.tensor(batch_count, dtype=torch.long, device=device)
        
        overall_tensors = [total_loss_tensor, correct_tensor, total_tensor, batch_count_tensor]
        
        if batch_all_reduce(overall_tensors, op=dist.ReduceOp.SUM):
            # 使用聚合后的结果
            total_loss = total_loss_tensor.item()
            correct = correct_tensor.item()
            total = total_tensor.item()
            batch_count = batch_count_tensor.item()
        else:
            if show_progress:
                print("⚠️  整体统计聚合失败，使用本地结果")
        
        # 聚合数据集特定统计
        if dataset_stats:
            aggregated_dataset_stats = {}
            for dataset_name, stats in dataset_stats.items():
                # 创建tensor并尝试聚合
                dataset_tensors = [
                    torch.tensor(stats['total_loss'], dtype=torch.float32, device=device),
                    torch.tensor(stats['correct'], dtype=torch.long, device=device),
                    torch.tensor(stats['total'], dtype=torch.long, device=device),
                    torch.tensor(stats['batch_count'], dtype=torch.long, device=device)
                ]
                
                if batch_all_reduce(dataset_tensors, op=dist.ReduceOp.SUM):
                    aggregated_dataset_stats[dataset_name] = {
                        'total_loss': dataset_tensors[0].item(),
                        'correct': dataset_tensors[1].item(),
                        'total': dataset_tensors[2].item(),
                        'batch_count': dataset_tensors[3].item()
                    }
                else:
                    # 聚合失败，使用本地统计
                    aggregated_dataset_stats[dataset_name] = stats
            
            dataset_stats = aggregated_dataset_stats
    
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
    
    # 输出结果（只在主进程输出）
    if show_progress:
        print("\n" + "="*80)
        print("📊 多数据集评估结果")
        if is_distributed:
            print(f"   (分布式聚合结果, world_size={world_size})")
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

def evaluate_single_dataset_fast(model, val_loader, device) -> Tuple[float, float]:
    """优化的单数据集评估函数 - 大幅提升速度"""
    import torch.distributed as dist
    
    # 确保模型处于评估模式
    model.eval()
    if hasattr(model, 'module'):
        model.module.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    batch_count = 0
    
    # 检查分布式状态
    is_distributed = dist.is_available() and dist.is_initialized()
    if is_distributed:
        current_rank = dist.get_rank()
    else:
        current_rank = 0
    
    # 🔥 优化：减少进度条更新频率，只在主进程显示
    show_progress = not is_distributed or current_rank == 0
    eval_pbar = tqdm(val_loader, desc="Evaluating", leave=False, disable=not show_progress)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_pbar):
            try:
                batch_count += 1
                
                # 移动数据到设备 - 使用non_blocking加速
                inputs = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                
                # 前向传播
                forward_kwargs = {
                    "input_ids": inputs,
                    "attention_mask": attention_mask,
                    "pixel_values": pixel_values,
                    "labels": labels
                }
                
                # 检查并添加image_grid_thw参数
                if "image_grid_thw" in batch:
                    forward_kwargs["image_grid_thw"] = batch["image_grid_thw"].to(device, non_blocking=True)
                
                outputs = model(**forward_kwargs)
                
                # 计算损失和准确率
                loss = outputs.loss
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                # 更新统计
                total_loss += loss.item()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                # 🔥 优化：大幅减少进度条更新频率，只在关键步骤更新
                if show_progress and (batch_idx % 100 == 0 or batch_idx == len(val_loader) - 1):
                    current_accuracy = correct / total if total > 0 else 0
                    current_avg_loss = total_loss / batch_count
                    eval_pbar.set_postfix({
                        'loss': f'{current_avg_loss:.4f}',
                        'accuracy': f'{current_accuracy:.4f}',
                        'samples': f'{total}'
                    })
                
            except Exception as e:
                if show_progress:
                    print(f"❌ 评估批次 {batch_idx} 出错: {e}")
                continue
    
    # 关闭进度条
    eval_pbar.close()
    
    # 🔥 优化：简化分布式聚合，减少通信开销
    if is_distributed:
        # 只进行一次聚合，避免多次all_reduce
        total_loss_tensor = torch.tensor(total_loss, dtype=torch.float32, device=device)
        correct_tensor = torch.tensor(correct, dtype=torch.long, device=device) 
        total_tensor = torch.tensor(total, dtype=torch.long, device=device)
        
        # 批量聚合
        tensors_to_reduce = [total_loss_tensor, correct_tensor, total_tensor]
        if batch_all_reduce(tensors_to_reduce, op=dist.ReduceOp.SUM):
            total_loss = total_loss_tensor.item()
            correct = correct_tensor.item()
            total = total_tensor.item()
        
        # 只在主进程输出结果
        if current_rank == 0:
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            accuracy = correct / total if total > 0 else 0
            print(f"\n📊 评估结果: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return total_loss / batch_count if batch_count > 0 else 0, correct / total if total > 0 else 0
    else:
        # 单GPU模式
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        accuracy = correct / total if total > 0 else 0
        print(f"\n📊 评估结果: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f} ({accuracy*100:.2f}%)")
        return avg_loss, accuracy 