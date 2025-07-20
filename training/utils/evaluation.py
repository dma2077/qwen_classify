import torch
import time
from typing import Dict, Tuple
from tqdm import tqdm
from collections import defaultdict
from .distributed import safe_all_reduce, safe_barrier, batch_all_reduce

def evaluate_model(model, val_loader, device) -> Tuple[float, float]:
    """评估模型性能 - 增强分布式支持，防止死锁"""
    import torch.distributed as dist
    import signal
    
    # 检查分布式状态
    is_distributed = dist.is_available() and dist.is_initialized()
    if is_distributed:
        current_rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # 🔥 强制评估开始前同步
        try:
            dist.barrier()
            if current_rank == 0:
                print("🔄 评估开始前分布式同步完成")
        except Exception as e:
            print(f"⚠️ 评估开始前同步失败: {e}")
            return 999.0, 0.0
    else:
        current_rank = 0
        world_size = 1
    
    # 🔥 设置评估超时保护
    def timeout_handler(signum, frame):
        raise TimeoutError("评估过程超时")
    
    timeout_set = False
    if current_rank == 0:
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(600)  # 10分钟超时
            timeout_set = True
        except:
            pass  # 如果设置超时失败，继续执行
    
    try:
        # 🔥 确保模型处于评估模式 - 兼容DeepSpeed包装
        model.eval()
        if hasattr(model, 'module'):
            model.module.eval()
            if current_rank == 0:
                print(f"🔍 设置DeepSpeed包装模型为eval模式: model.training={model.training}, model.module.training={model.module.training}")
        else:
            if current_rank == 0:
                print(f"🔍 设置模型为eval模式: model.training={model.training}")
        
        total_loss = 0
        correct = 0
        total = 0
        
        # 只在主进程显示进度条
        show_progress = not is_distributed or current_rank == 0
        eval_pbar = tqdm(val_loader, desc="Evaluating", leave=False, disable=not show_progress)
        
        batch_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_pbar):
                try:
                    batch_count += 1
                    
                    # 🔥 每20个batch进行同步检查，减少同步频率
                    if is_distributed and batch_idx % 20 == 0:
                        try:
                            dist.barrier()
                        except Exception as sync_e:
                            if current_rank == 0:
                                print(f"⚠️ 第{batch_idx}批次同步失败: {sync_e}")
                    
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
                    
                except Exception as batch_e:
                    if show_progress:
                        print(f"❌ 评估批次 {batch_idx} 出错: {batch_e}")
                        eval_pbar.set_postfix({'error': f'batch_{batch_idx}_failed'})
                    continue
        
        # 关闭进度条
        eval_pbar.close()
        
        # 🔥 强制评估结束后同步
        if is_distributed:
            try:
                dist.barrier()
                if current_rank == 0:
                    print("🔄 评估结束后分布式同步完成")
            except Exception as e:
                print(f"⚠️ 评估结束后同步失败: {e}")
        
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
            
    except Exception as eval_e:
        # 🔥 评估异常处理
        if current_rank == 0:
            print(f"❌ 评估过程中发生异常: {eval_e}")
        
        # 尝试同步所有进程，避免死锁
        if is_distributed:
            try:
                dist.barrier()
                if current_rank == 0:
                    print("🔄 异常后分布式同步完成")
            except:
                if current_rank == 0:
                    print("⚠️ 异常后同步也失败")
        
        # 返回错误标识
        return 999.0, 0.0
        
    finally:
        # 🔥 清理超时设置
        if timeout_set and current_rank == 0:
            try:
                signal.alarm(0)
            except:
                pass


def evaluate_multi_dataset(model, val_loader, device, dataset_configs=None) -> Dict:
    """
    多数据集评估函数，支持按数据集分别统计指标 - 增强分布式支持
    """
    import torch.distributed as dist
    
    # 🔥 修复：添加分布式同步，确保所有进程同时开始评估
    is_distributed = dist.is_available() and dist.is_initialized()
    if is_distributed:
        current_rank = dist.get_rank()
        try:
            dist.barrier()
            if current_rank == 0:
                print("🔄 多数据集评估开始前分布式同步完成")
        except Exception as e:
            print(f"❌ 多数据集评估开始前同步失败: {e}")
            return {}
    else:
        current_rank = 0
    
    try:
        # 🔥 确保模型处于评估模式 - 兼容DeepSpeed包装
        model.eval()
        
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
                    
                    # 🔥 每20个batch进行同步检查
                    if is_distributed and batch_idx % 20 == 0:
                        try:
                            dist.barrier()
                        except Exception as sync_e:
                            if current_rank == 0:
                                print(f"⚠️ 多数据集评估第{batch_idx}批次同步失败: {sync_e}")
                    
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
                    
                    outputs = model(**forward_kwargs)
                    
                    # 计算损失和准确率
                    loss = outputs.loss
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    
                    # 整体统计更新
                    total_loss += loss.item()
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
                    
                    # 按数据集更新统计
                    if dataset_names:
                        batch_size = labels.size(0)
                        for i in range(batch_size):
                            if i < len(dataset_names):
                                dataset_name = dataset_names[i]
                                label_val = labels[i].item()
                                pred_val = predictions[i].item()
                                
                                dataset_stats[dataset_name]['total_loss'] += loss.item() / batch_size
                                dataset_stats[dataset_name]['total'] += 1
                                dataset_stats[dataset_name]['batch_count'] += 1 / batch_size
                                
                                if pred_val == label_val:
                                    dataset_stats[dataset_name]['correct'] += 1
                    
                    # 更新进度条
                    if show_progress and (batch_idx % 10 == 0 or batch_idx == len(val_loader) - 1):
                        current_accuracy = correct / total if total > 0 else 0
                        current_avg_loss = total_loss / batch_count
                        eval_pbar.set_postfix({
                            'loss': f'{current_avg_loss:.4f}',
                            'accuracy': f'{current_accuracy:.4f}',
                            'datasets': len(dataset_stats)
                        })
                
                except Exception as e:
                    if show_progress:
                        print(f"❌ 多数据集评估批次 {batch_idx} 出错: {e}")
                    continue
        
        # 关闭进度条
        eval_pbar.close()
        
        # 🔥 强制评估结束后同步
        if is_distributed:
            try:
                dist.barrier()
                if current_rank == 0:
                    print("🔄 多数据集评估结束后分布式同步完成")
            except Exception as e:
                print(f"⚠️ 多数据集评估结束后同步失败: {e}")
        
        # 计算整体指标
        overall_loss = total_loss / batch_count if batch_count > 0 else 0
        overall_accuracy = correct / total if total > 0 else 0
        
        # 构建结果字典
        results = {
            'overall_loss': overall_loss,
            'overall_accuracy': overall_accuracy,
            'total_samples': total,
            'total_correct': correct,
            'dataset_metrics': {}
        }
        
        # 计算每个数据集的指标
        for dataset_name, stats in dataset_stats.items():
            if stats['total'] > 0:
                dataset_loss = stats['total_loss'] / stats['batch_count'] if stats['batch_count'] > 0 else 0
                dataset_accuracy = stats['correct'] / stats['total']
                
                results['dataset_metrics'][dataset_name] = {
                    'loss': dataset_loss,
                    'accuracy': dataset_accuracy,
                    'samples': stats['total'],
                    'correct': stats['correct']
                }
        
        # 只在主进程打印结果
        if current_rank == 0:
            print(f"\n📊 多数据集评估完成:")
            print(f"  整体损失: {overall_loss:.4f}")
            print(f"  整体准确率: {overall_accuracy:.4f}")
            print(f"  数据集数量: {len(dataset_stats)}")
        
        return results
        
    except Exception as eval_e:
        # 🔥 评估异常处理
        if current_rank == 0:
            print(f"❌ 多数据集评估过程中发生异常: {eval_e}")
        
        # 尝试同步所有进程，避免死锁
        if is_distributed:
            try:
                dist.barrier()
                if current_rank == 0:
                    print("🔄 多数据集评估异常后分布式同步完成")
            except:
                if current_rank == 0:
                    print("⚠️ 多数据集评估异常后同步也失败")
        
        return {}


def evaluate_single_dataset_fast(model, val_loader, device) -> Tuple[float, float]:
    """优化的单数据集评估函数 - 增强分布式支持"""
    import torch.distributed as dist
    import signal
    
    # 检查分布式状态
    is_distributed = dist.is_available() and dist.is_initialized()
    if is_distributed:
        current_rank = dist.get_rank()
        # 🔥 强制评估开始前同步
        try:
            dist.barrier()
            if current_rank == 0:
                print("🔄 快速评估开始前分布式同步完成")
        except Exception as e:
            print(f"❌ 快速评估开始前同步失败: {e}")
            return 999.0, 0.0
    else:
        current_rank = 0
    
    # 🔥 设置评估超时保护
    def timeout_handler(signum, frame):
        raise TimeoutError("快速评估过程超时")
    
    timeout_set = False
    if current_rank == 0:
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(300)  # 5分钟超时
            timeout_set = True
        except:
            pass
    
    try:
        # 确保模型处于评估模式
        model.eval()
        if hasattr(model, 'module'):
            model.module.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        batch_count = 0
        
        # 🔥 优化：减少进度条更新频率，只在主进程显示
        show_progress = not is_distributed or current_rank == 0
        eval_pbar = tqdm(val_loader, desc="Evaluating", leave=False, disable=not show_progress)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_pbar):
                try:
                    batch_count += 1
                    
                    # 🔥 每20个batch进行同步检查
                    if is_distributed and batch_idx % 20 == 0:
                        try:
                            dist.barrier()
                        except Exception as sync_e:
                            if current_rank == 0:
                                print(f"⚠️ 第{batch_idx}批次同步失败: {sync_e}")
                    
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
        
        # 🔥 强制评估结束后同步
        if is_distributed:
            try:
                dist.barrier()
                if current_rank == 0:
                    print("🔄 快速评估结束后分布式同步完成")
            except Exception as e:
                print(f"⚠️ 快速评估结束后同步失败: {e}")
        
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
        
        # 计算最终结果
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        if current_rank == 0:
            print(f"📊 快速评估完成 - 损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}")
        
        return avg_loss, accuracy
        
    except Exception as eval_e:
        # 🔥 评估异常处理
        if current_rank == 0:
            print(f"❌ 快速评估过程中发生异常: {eval_e}")
        
        # 尝试同步所有进程，避免死锁
        if is_distributed:
            try:
                dist.barrier()
                if current_rank == 0:
                    print("🔄 快速评估异常后分布式同步完成")
            except:
                if current_rank == 0:
                    print("⚠️ 快速评估异常后同步也失败")
        
        # 返回错误标识
        return 999.0, 0.0
        
    finally:
        # 🔥 清理超时设置
        if timeout_set and current_rank == 0:
            try:
                signal.alarm(0)
            except:
                pass 