import os
import time
import json
import torch
import deepspeed
from tqdm import tqdm
from transformers import AutoProcessor
from collections import defaultdict
from .utils.model_utils import save_hf_model
from .utils.distributed import DistributedContext
from .utils.evaluation import evaluate_multi_dataset
from .utils.monitor import TrainingMonitor, make_json_serializable, calculate_mfu
from data.dataloader import create_full_eval_dataloader

class DeepSpeedTrainer:
    def __init__(self, config):
        # 假设配置已经通过prepare_config处理过
        self.config = config
        self.dist_ctx = DistributedContext()
        
        # 设置NCCL超时保护（在分布式训练时）
        if self.dist_ctx.world_size > 1:
            from .utils.distributed import setup_nccl_timeout_env
            setup_nccl_timeout_env()
        
        # 只在主进程创建完整的TrainingMonitor，非主进程使用DummyMonitor
        if self.dist_ctx.is_main_process:
            from training.utils.monitor import TrainingMonitor
            self.monitor = TrainingMonitor(self.config['output_dir'], config)
            print("✅ 主进程：创建完整TrainingMonitor（包含wandb）")
        else:
            from training.utils.monitor import DummyMonitor  
            self.monitor = DummyMonitor(self.config['output_dir'], config)
            print(f"ℹ️  进程 rank {self.dist_ctx.rank}：使用DummyMonitor（无wandb）")
        
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.lr_scheduler = None
        self.current_step = 0
        self.current_epoch = 0
        
        # 多数据集支持
        self.dataset_configs = self.config.get('datasets', {}).get('dataset_configs', {})
        self.enable_dataset_metrics = self.config.get('wandb', {}).get('log_dataset_metrics', True)
        
        # 用于跟踪各数据集的指标
        self.dataset_metrics = defaultdict(lambda: {
            'total_loss': 0.0,
            'total_samples': 0,
            'correct_samples': 0,
            'step_count': 0
        })
        
        # 最佳模型追踪
        self.best_model_config = self.config.get('training', {}).get('best_model_tracking', {})
        self.best_model_enabled = self.best_model_config.get('enabled', True)
        self.best_metric_name = self.best_model_config.get('metric', 'overall_accuracy')
        self.best_metric_mode = self.best_model_config.get('mode', 'max')  # 'max' or 'min'
        self.save_best_only = self.best_model_config.get('save_best_only', True)
        
        # 初始化最佳指标
        if self.best_metric_mode == 'max':
            self.best_metric_value = float('-inf')
        else:
            self.best_metric_value = float('inf')
        
        self.best_model_step = 0
        self.best_model_path = None
        
        # 评估配置
        self.eval_config = self.config.get('training', {}).get('evaluation', {})
        self.partial_eval_during_training = self.eval_config.get('partial_eval_during_training', True)
        self.full_eval_at_end = self.eval_config.get('full_eval_at_end', True)
        self.eval_best_model_only = self.eval_config.get('eval_best_model_only', True)
        
    def setup_model(self, model, train_loader, val_loader, optimizer, lr_scheduler):
        """设置模型和相关组件"""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        # 初始化DeepSpeed
        self.model, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=self.config['deepspeed']
        )
        
        self.dist_ctx.print_info()
        self.dist_ctx.print_main(f"模型初始化完成，设备: {self.dist_ctx.device}")
        
        # 设置monitor的model引用用于MFU计算
        self.monitor.set_model_ref(self.model)
        
    def save_checkpoint(self, step, is_best=False):
        """保存检查点"""
        if is_best:
            checkpoint_dir = os.path.join(self.config['output_dir'], f"best-model-step-{step}")
        else:
            checkpoint_dir = os.path.join(self.config['output_dir'], f"checkpoint-{step}")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存训练信息
        training_info = {
            'step': step,
            'epoch': self.current_epoch,
            'config': self.config,
            'dataset_metrics': dict(self.dataset_metrics),  # 保存数据集指标
            'is_best_model': is_best,
            'best_metric_value': self.best_metric_value if is_best else None,
            'timestamp': time.time()
        }
        
        with open(os.path.join(checkpoint_dir, 'training_info.json'), 'w') as f:
            # 使用make_json_serializable确保所有数据都可以序列化
            json.dump(make_json_serializable(training_info), f, indent=2)
        
        # 保存DeepSpeed格式（可选）
        if self.config.get('save_deepspeed_format', True):
            deepspeed_dir = os.path.join(checkpoint_dir, 'deepspeed')
            self.model.save_checkpoint(deepspeed_dir)
            if is_best:
                self.dist_ctx.print_main(f"🏆 最佳模型DeepSpeed检查点保存到: {deepspeed_dir}")
            else:
                self.dist_ctx.print_main(f"DeepSpeed检查点保存到: {deepspeed_dir}")
        
        # 保存HuggingFace格式（可选）
        if self.config.get('save_hf_format', True):
            if self.dist_ctx.is_main_process:
                hf_dir = save_hf_model(self.model, self.config, checkpoint_dir)
                if hf_dir:
                    if is_best:
                        self.dist_ctx.print_main(f"🏆 最佳模型HuggingFace检查点保存到: {hf_dir}")
                    else:
                        self.dist_ctx.print_main(f"HuggingFace检查点保存到: {hf_dir}")
        
        if is_best:
            self.best_model_path = checkpoint_dir
        
        self.dist_ctx.barrier()
        return checkpoint_dir
    
    def _is_better_metric(self, current_value, best_value):
        """判断当前指标是否更好"""
        if self.best_metric_mode == 'max':
            return current_value > best_value
        else:
            return current_value < best_value
    
    def _cleanup_old_best_models(self):
        """清理所有检查点，只保留最新的最佳模型"""
        if not self.save_best_only:
            return
            
        try:
            import glob
            import shutil
            
            # 查找所有检查点目录
            best_model_pattern = os.path.join(self.config['output_dir'], "best-model-step-*")
            checkpoint_pattern = os.path.join(self.config['output_dir'], "checkpoint-*")
            
            best_model_dirs = glob.glob(best_model_pattern)
            checkpoint_dirs = glob.glob(checkpoint_pattern)
            
            dirs_to_remove = []
            
            # 1. 删除所有常规检查点（checkpoint-*）
            dirs_to_remove.extend(checkpoint_dirs)
            
            # 2. 删除除最新之外的所有最佳模型检查点
            if len(best_model_dirs) > 1:
                def extract_step(path):
                    try:
                        return int(os.path.basename(path).split('-')[-1])
                    except:
                        return 0
                
                best_model_dirs.sort(key=extract_step)
                dirs_to_remove.extend(best_model_dirs[:-1])  # 保留最后一个（最新的）
            
            # 执行清理
            total_removed = 0
            for dir_path in dirs_to_remove:
                if os.path.exists(dir_path):
                    dir_name = os.path.basename(dir_path)
                    self.dist_ctx.print_main(f"🗑️  删除检查点: {dir_name}")
                    shutil.rmtree(dir_path)
                    total_removed += 1
            
            # 显示清理结果
            if total_removed > 0:
                self.dist_ctx.print_main(f"✅ 清理完成，删除了 {total_removed} 个检查点")
                
                # 显示保留的最佳模型
                remaining_best = glob.glob(best_model_pattern)
                if remaining_best:
                    remaining_best.sort(key=lambda x: int(os.path.basename(x).split('-')[-1]))
                    self.dist_ctx.print_main(f"🏆 保留最佳模型: {os.path.basename(remaining_best[-1])}")
            else:
                self.dist_ctx.print_main("✅ 无需清理，目录已经很干净")
                
        except Exception as e:
            self.dist_ctx.print_main(f"⚠️  清理检查点时出错: {e}")

    def _update_best_model(self, eval_results, step):
        """更新最佳模型"""
        if not self.best_model_enabled:
            return False
        
        # 获取当前指标值
        if self.best_metric_name == 'overall_accuracy':
            current_value = eval_results.get('overall_accuracy', 0.0)
        elif self.best_metric_name == 'overall_loss':
            current_value = eval_results.get('overall_loss', float('inf'))
        else:
            # 支持数据集特定指标，如 'food101_accuracy'
            if 'dataset_metrics' in eval_results:
                for dataset_name, metrics in eval_results['dataset_metrics'].items():
                    metric_key = self.best_metric_name.replace(f'{dataset_name}_', '')
                    if self.best_metric_name.startswith(dataset_name) and metric_key in metrics:
                        current_value = metrics[metric_key]
                        break
                else:
                    current_value = eval_results.get('overall_accuracy', 0.0)  # 默认使用overall_accuracy
            else:
                current_value = eval_results.get('overall_accuracy', 0.0)
        
        # 检查是否是最佳模型
        if self._is_better_metric(current_value, self.best_metric_value):
            self.best_metric_value = current_value
            self.best_model_step = step
            
            # 保存最佳模型
            self.save_checkpoint(step, is_best=True)
            
            # 清理旧的最佳模型（如果启用了仅保存最佳模型）
            if self.save_best_only:
                self._cleanup_old_best_models()
            
            # 记录到wandb
            self.monitor.log_metrics({
                'best_model_step': step,
                f'best_{self.best_metric_name}': current_value
            }, step)
            
            self.dist_ctx.print_main(
                f"🏆 发现更好模型! {self.best_metric_name}: {current_value:.4f} "
                f"(步骤 {step})"
            )
            return True
        
        return False
    
    def _aggregate_loss(self, loss):
        """在分布式训练中聚合loss"""
        if self.dist_ctx.world_size <= 1:
            return loss.item()
        
        try:
            import torch.distributed as dist
            # 将当前GPU的loss广播到所有进程并求平均
            loss_tensor = torch.tensor(loss.item(), dtype=torch.float32, device=self.dist_ctx.device)
            
            # 使用all_reduce来计算所有GPU的平均loss
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            aggregated_loss = loss_tensor.item() / self.dist_ctx.world_size
            
            return aggregated_loss
            
        except Exception as e:
            # 如果聚合失败，返回当前GPU的loss
            print(f"⚠️  Loss聚合失败，使用当前GPU loss: {e}")
            return loss.item()
    
    def _update_dataset_metrics(self, batch, outputs, aggregated_loss):
        """更新各数据集的指标 - 优化版本，减少计算开销"""
        if not self.enable_dataset_metrics:
            return
            
        dataset_names = batch.get("dataset_names", [])
        labels = batch.get("labels")
        
        if not dataset_names or labels is None or outputs.logits is None:
            return
        
        # 只在必要时计算预测结果（避免每次都计算）
        predictions = None
        
        # 按数据集统计指标 - 简化循环和计算
        dataset_count = len(dataset_names)
        if dataset_count == 0:
            return
            
        # 批量更新基础指标，避免逐个更新
        avg_loss_per_sample = aggregated_loss / dataset_count
        
        for i, dataset_name in enumerate(dataset_names):
            if i >= len(labels):
                continue
            
            # 延迟计算预测结果，只在需要时计算
            if predictions is None:
                predictions = torch.argmax(outputs.logits, dim=-1)
            
            if i >= len(predictions):
                continue
                
            # 简化指标更新，减少重复计算
            metrics = self.dataset_metrics[dataset_name]
            metrics['total_loss'] += avg_loss_per_sample
            metrics['total_samples'] += 1
            metrics['step_count'] += 1
            
            # 只在需要时进行tensor转换
            if predictions[i].item() == labels[i].item():
                metrics['correct_samples'] += 1
    
    def _log_dataset_metrics(self, step, is_eval=False):
        """记录各数据集的指标"""
        if not self.enable_dataset_metrics or not self.dataset_metrics:
            return
            
        # 计算并输出各数据集的指标
        dataset_log_data = {}
        overall_samples = 0
        overall_correct = 0
        
        # 根据是否为评估模式选择指标组
        metric_group = "eval" if is_eval else "training"
        
        for dataset_name, metrics in self.dataset_metrics.items():
            if metrics['total_samples'] == 0:
                continue
                
            avg_loss = metrics['total_loss'] / metrics['step_count'] if metrics['step_count'] > 0 else 0
            accuracy = metrics['correct_samples'] / metrics['total_samples']
            
            dataset_log_data[f"{metric_group}/{dataset_name}_loss"] = avg_loss
            dataset_log_data[f"{metric_group}/{dataset_name}_accuracy"] = accuracy
            dataset_log_data[f"{metric_group}/{dataset_name}_samples"] = metrics['total_samples']
            
            # 累计整体指标
            overall_samples += metrics['total_samples']
            overall_correct += metrics['correct_samples']
            
            # 只在主进程输出详细信息
            if self.dist_ctx.is_main_process:
                prefix = "EVAL" if is_eval else "TRAIN"
                print(f"📊 {prefix} - {dataset_name}: "
                      f"Loss={avg_loss:.4f}, Acc={accuracy:.4f} ({accuracy*100:.2f}%), "
                      f"Samples={metrics['total_samples']}")
        
        # 添加整体指标
        if overall_samples > 0:
            overall_accuracy = overall_correct / overall_samples
            dataset_log_data[f"{metric_group}/overall_accuracy"] = overall_accuracy
            dataset_log_data[f"{metric_group}/overall_samples"] = overall_samples
            dataset_log_data[f"{metric_group}/overall_correct"] = overall_correct
            
            if self.dist_ctx.is_main_process:
                prefix = "EVAL" if is_eval else "TRAIN"
                print(f"📊 {prefix} - OVERALL: "
                      f"Acc={overall_accuracy:.4f} ({overall_accuracy*100:.2f}%), "
                      f"Samples={overall_samples}")
        
        # 记录到wandb
        if dataset_log_data:
            self.monitor.log_metrics(dataset_log_data, step, commit=False)
            
        # 如果不是eval模式，重置训练指标
        if not is_eval:
            self.dataset_metrics.clear()
    
    def _forward_backward_with_profiling(self, forward_kwargs):
        """在前向+反向传播过程中实时测量FLOPs"""
        try:
            total_flops = 0.0
            outputs = None
            loss = None
            
            # 检查PyTorch是否支持FLOPs profiling
            try:
                # 使用profiler包装完整的前向+反向传播过程
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_flops=True,
                    profile_memory=False
                ) as prof:
                    # 前向传播
                    outputs = self.model(**forward_kwargs)
                    loss = outputs.loss
                    
                    # 反向传播
                    self.model.backward(loss)
                
                # 收集FLOPs统计
                for event in prof.events():
                    if hasattr(event, 'flops') and event.flops > 0:
                        total_flops += event.flops
                
                return outputs, loss, float(total_flops)
                
            except (AttributeError, TypeError) as e:
                # 如果profiler不支持with_flops，回退到正常执行
                print(f"⚠️  Profiler不支持FLOPs测量，使用正常模式: {e}")
                outputs = self.model(**forward_kwargs)
                loss = outputs.loss
                self.model.backward(loss)
                return outputs, loss, 0.0
                
        except Exception as e:
            print(f"❌ 实时FLOPs测量失败: {e}")
            # 发生错误时执行正常的前向+反向传播
            outputs = self.model(**forward_kwargs)
            loss = outputs.loss
            self.model.backward(loss)
            return outputs, loss, 0.0
        
    def evaluate(self, step=None):
        """评估模型，统一使用多数据集评估逻辑
        
        Args:
            step: 当前步数，如果提供则用于最佳模型保存；否则使用self.current_step
        """
        self.dist_ctx.print_main("🔍 开始评估...")
        
        # 统一使用多数据集评估函数
        eval_results = evaluate_multi_dataset(self.model, self.val_loader, self.dist_ctx.device, self.dataset_configs)
        
        # 添加调试信息
        self.dist_ctx.print_main(f"🔍 eval_results 原始数据: {eval_results}")
        
        # 准备wandb记录数据
        eval_log_data = {}
        overall_samples = 0
        overall_correct = 0
        
        # 处理数据集指标（如果存在）
        if eval_results and 'dataset_metrics' in eval_results and eval_results['dataset_metrics']:
            self.dist_ctx.print_main(f"📊 检测到多数据集评估结果:")
            # 多数据集情况：记录每个数据集的指标
            for dataset_name, metrics in eval_results['dataset_metrics'].items():
                eval_log_data[f"eval/{dataset_name}_loss"] = metrics['loss']
                eval_log_data[f"eval/{dataset_name}_accuracy"] = metrics['accuracy']
                eval_log_data[f"eval/{dataset_name}_samples"] = metrics['samples']
                
                overall_samples += metrics['samples']
                overall_correct += metrics['correct']
                
                # 打印每个数据集的详细结果
                self.dist_ctx.print_main(f"  📂 {dataset_name}:")
                self.dist_ctx.print_main(f"     Loss: {metrics['loss']:.6f}")
                self.dist_ctx.print_main(f"     Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
                self.dist_ctx.print_main(f"     Samples: {metrics['samples']:,} (Correct: {metrics['correct']:,})")
        else:
            self.dist_ctx.print_main(f"📊 检测到单数据集评估结果:")
            # 单数据集情况：使用overall指标作为主要指标
            eval_log_data["eval/loss"] = eval_results.get('overall_loss', 0)
            eval_log_data["eval/accuracy"] = eval_results.get('overall_accuracy', 0)
            overall_samples = eval_results.get('total_samples', 0)
            overall_correct = eval_results.get('total_correct', 0)
            
            self.dist_ctx.print_main(f"  📈 Loss: {eval_results.get('overall_loss', 0):.6f}")
            self.dist_ctx.print_main(f"  🎯 Accuracy: {eval_results.get('overall_accuracy', 0):.4f} ({eval_results.get('overall_accuracy', 0)*100:.2f}%)")
            self.dist_ctx.print_main(f"  📊 Samples: {overall_samples:,} (Correct: {overall_correct:,})")
        
        # 添加整体指标（适用于单数据集和多数据集）
        if overall_samples > 0:
            overall_accuracy = overall_correct / overall_samples
        else:
            overall_accuracy = eval_results.get('overall_accuracy', 0)
            overall_samples = eval_results.get('total_samples', 0)
            overall_correct = eval_results.get('total_correct', 0)
        
        # 总是添加整体指标
        overall_loss = eval_results.get('overall_loss', 0)
        eval_log_data["eval/overall_loss"] = overall_loss
        eval_log_data["eval/overall_accuracy"] = overall_accuracy
        eval_log_data["eval/overall_samples"] = overall_samples
        eval_log_data["eval/overall_correct"] = overall_correct
        
        # 记录到wandb
        current_step = step if step is not None else self.current_step
        
        # 打印综合评估结果
        self.dist_ctx.print_main("=" * 80)
        self.dist_ctx.print_main(f"📊 评估完成 (Step {current_step})")
        self.dist_ctx.print_main("=" * 80)
        self.dist_ctx.print_main(f"🎯 整体准确率: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        self.dist_ctx.print_main(f"📈 整体损失:   {overall_loss:.6f}")
        self.dist_ctx.print_main(f"📊 总样本数:   {overall_samples:,}")
        self.dist_ctx.print_main(f"✅ 正确样本:   {overall_correct:,}")
        self.dist_ctx.print_main("=" * 80)
        
        # 添加调试信息
        self.dist_ctx.print_main(f"🔍 准备记录eval指标到wandb (step={current_step}):")
        for key, value in eval_log_data.items():
            self.dist_ctx.print_main(f"   • {key}: {value}")
        
        self.monitor.log_metrics(eval_log_data, current_step)
        
        # 更新最佳模型
        eval_results_for_best = {
            'overall_loss': overall_loss,
            'overall_accuracy': overall_accuracy
        }
        self._update_best_model(eval_results_for_best, current_step)
        
        # 返回整体指标
        self.dist_ctx.print_main(f"✅ 评估结束 - 验证损失: {overall_loss:.4f}, 准确率: {overall_accuracy:.4f}")
        return overall_loss, overall_accuracy
    
    def full_evaluation_on_best_model(self):
        """在最佳模型上进行完整评估"""
        if not self.full_eval_at_end or not self.best_model_path:
            return
        
        self.dist_ctx.print_main("\n" + "="*80)
        self.dist_ctx.print_main("🔍 开始对最佳模型进行完整评估")
        self.dist_ctx.print_main("="*80)
        
        # 创建完整评估数据加载器
        # 安全地获取processor，避免DeepSpeed包装导致的属性访问错误
        try:
            processor = self.model.module.processor
        except AttributeError:
            try:
                processor = self.model.processor
            except AttributeError:
                processor = None
                self.dist_ctx.print_main("⚠️ 无法获取模型processor，将从配置中重新加载")
        
        full_eval_loader = create_full_eval_dataloader(self.config, processor)
        
        if full_eval_loader is None:
            self.dist_ctx.print_main("⚠️ 无法创建完整评估数据加载器，跳过完整评估")
            return
        
        # 统一使用多数据集评估函数
        eval_results = evaluate_multi_dataset(self.model, full_eval_loader, self.dist_ctx.device, self.dataset_configs)
        
        # 准备wandb记录数据
        eval_log_data = {}
        overall_samples = 0
        overall_correct = 0
        
        # 处理数据集指标（如果存在）
        if eval_results and 'dataset_metrics' in eval_results and eval_results['dataset_metrics']:
            # 多数据集情况：记录每个数据集的指标
            for dataset_name, metrics in eval_results['dataset_metrics'].items():
                eval_log_data[f"eval/final_{dataset_name}_loss"] = metrics['loss']
                eval_log_data[f"eval/final_{dataset_name}_accuracy"] = metrics['accuracy']
                eval_log_data[f"eval/final_{dataset_name}_samples"] = metrics['samples']
                
                overall_samples += metrics['samples']
                overall_correct += metrics['correct']
        else:
            # 单数据集情况：使用overall指标作为主要指标
            eval_log_data["eval/final_loss"] = eval_results.get('overall_loss', 0)
            eval_log_data["eval/final_accuracy"] = eval_results.get('overall_accuracy', 0)
            overall_samples = eval_results.get('total_samples', 0)
            overall_correct = eval_results.get('total_correct', 0)
        
        # 添加整体指标（适用于单数据集和多数据集）
        if overall_samples > 0:
            overall_accuracy = overall_correct / overall_samples
        else:
            overall_accuracy = eval_results.get('overall_accuracy', 0)
            overall_samples = eval_results.get('total_samples', 0)
            overall_correct = eval_results.get('total_correct', 0)
        
        # 总是添加整体指标
        eval_log_data["eval/final_overall_loss"] = eval_results.get('overall_loss', 0)
        eval_log_data["eval/final_overall_accuracy"] = overall_accuracy
        eval_log_data["eval/final_overall_samples"] = overall_samples
        eval_log_data["eval/final_overall_correct"] = overall_correct
        
        # 记录到wandb
        self.monitor.log_metrics(eval_log_data, self.best_model_step)
        
        # 显示结果
        self.dist_ctx.print_main(f"\n🎯 最佳模型完整评估结果:")
        self.dist_ctx.print_main(f"   • 整体准确率: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        self.dist_ctx.print_main(f"   • 总样本数: {overall_samples:,}")
        self.dist_ctx.print_main(f"   • 正确样本数: {overall_correct:,}")
        
        self.dist_ctx.print_main("="*80)
        
        return {
            'overall_loss': eval_results.get('overall_loss', 0),
            'overall_accuracy': overall_accuracy,
            'dataset_metrics': eval_results.get('dataset_metrics', {})
        }
        
    def train(self):
        """训练模型"""
        self.dist_ctx.print_main("开始训练...")
        self.monitor.start_training()
        
        num_epochs = self.config['training']['num_epochs']
        logging_steps = self.config['logging_steps']
        save_steps = self.config['save_steps']
        eval_steps = self.config['eval_steps']
        
        # 计算有效训练步数（考虑DeepSpeed的分布式训练和梯度累积）
        deepspeed_config = self.config.get('deepspeed', {})
        if isinstance(deepspeed_config, str):
            with open(deepspeed_config, 'r') as f:
                deepspeed_config = json.load(f)
        
        # 获取DeepSpeed参数
        micro_batch_size_per_gpu = deepspeed_config.get('train_micro_batch_size_per_gpu', 1)
        gradient_accumulation_steps = deepspeed_config.get('gradient_accumulation_steps', 1)
        train_batch_size = deepspeed_config.get('train_batch_size', 32)
        
        # 计算有效训练步数（基于实际的DataLoader长度）
        dataloader_steps_per_epoch = len(self.train_loader)
        effective_steps_per_epoch = dataloader_steps_per_epoch // gradient_accumulation_steps
        total_effective_steps = effective_steps_per_epoch * num_epochs
        
        # 创建进度条（基于有效训练步数）
        pbar = tqdm(total=total_effective_steps, desc="Training Steps", disable=not self.dist_ctx.is_main_process)
        
        # 计算验证信息
        dataset_size = len(self.train_loader.dataset)
        samples_per_gpu = dataloader_steps_per_epoch * micro_batch_size_per_gpu
        
        # 使用更清晰的格式输出训练配置信息
        if self.dist_ctx.is_main_process:
            print("="*80)
            print("🚀 训练配置信息")
            print("="*80)
            print(f"📊 数据集配置:")
            print(f"  • 总数据集大小: {dataset_size:,}")
            print(f"  • 每GPU处理样本数: {samples_per_gpu:,}")
            print(f"📦 批次配置:")
            print(f"  • 每GPU微批次大小: {micro_batch_size_per_gpu}")
            print(f"  • 梯度累积步数: {gradient_accumulation_steps}")
            print(f"  • 总有效批次大小: {train_batch_size}")
            print(f"📈 步数统计:")
            print(f"  • 每GPU DataLoader步数: {dataloader_steps_per_epoch:,}")
            print(f"  • 有效训练步数每epoch: {effective_steps_per_epoch:,}")
            print(f"  • 总有效训练步数: {total_effective_steps:,}")
            print("="*80)
        
        effective_step = 0  # 用于跟踪有效步数
        flops_profiled = False  # 标记是否已经测量过FLOPs
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.model.train()
            
            # 为分布式采样器设置epoch（确保每个epoch的shuffle正确）
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            epoch_loss = 0
            epoch_start_time = time.time()
            
            for batch_idx, batch in enumerate(self.train_loader):
                self.current_step += 1
                
                # 前向传播
                inputs = batch["input_ids"].to(self.dist_ctx.device)
                attention_mask = batch["attention_mask"].to(self.dist_ctx.device)
                pixel_values = batch["pixel_values"].to(self.dist_ctx.device)
                labels = batch["labels"].to(self.dist_ctx.device)
                
                # 添加image_grid_thw参数（如果存在）
                forward_kwargs = {
                    "input_ids": inputs,
                    "attention_mask": attention_mask,
                    "pixel_values": pixel_values,
                    "labels": labels
                }
                
                # 检查并添加image_grid_thw参数
                if "image_grid_thw" in batch:
                    forward_kwargs["image_grid_thw"] = batch["image_grid_thw"].to(self.dist_ctx.device)
                
                # 添加多数据集支持的参数
                if "dataset_names" in batch:
                    forward_kwargs["dataset_names"] = batch["dataset_names"]
                if "num_classes_list" in batch:
                    forward_kwargs["num_classes_list"] = batch["num_classes_list"]
                
                # 决定是否进行实时FLOPs测量 - 大幅减少频率以避免性能开销
                should_measure_flops = (
                    not flops_profiled or  # 仅第一次测量
                    (effective_step > 0 and effective_step % 500 == 0)  # 每500个有效步骤重新测量（而不是50步）
                )
                
                # 实时FLOPs测量和模型前向+反向传播
                if should_measure_flops and self.dist_ctx.is_main_process:
                    # 在主进程中进行实时FLOPs测量（仅限必要时）
                    outputs, loss, real_time_flops = self._forward_backward_with_profiling(forward_kwargs)
                    
                    # 更新FLOPs信息
                    if real_time_flops > 0:
                        self.monitor.set_actual_flops(real_time_flops, attention_mask.size(1))
                        if not flops_profiled:
                            print(f"✅ 实时测量FLOPs: {real_time_flops:.2e}")
                        flops_profiled = True
                else:
                    # 正常的前向+反向传播（无profiling开销） - 大多数步骤使用这个路径
                    outputs = self.model(**forward_kwargs)
                    loss = outputs.loss
                    self.model.backward(loss)
                    real_time_flops = None
                
                # 简化分布式FLOPs同步 - 仅在首次测量时同步，避免每次都广播
                if should_measure_flops and real_time_flops is not None and self.dist_ctx.world_size > 1 and not flops_profiled:
                    import torch.distributed as dist
                    
                    # 仅在首次成功测量FLOPs时进行一次同步
                    flops_tensor = torch.tensor(real_time_flops, dtype=torch.float32, device=self.dist_ctx.device)
                    seq_tensor = torch.tensor(attention_mask.size(1), dtype=torch.float32, device=self.dist_ctx.device)
                    
                    try:
                        dist.broadcast(flops_tensor, src=0)
                        dist.broadcast(seq_tensor, src=0)
                        
                        # 所有进程更新FLOPs信息
                        if not self.dist_ctx.is_main_process:
                            self.monitor.set_actual_flops(flops_tensor.item(), int(seq_tensor.item()))
                    except Exception as e:
                        print(f"⚠️  FLOPs同步失败: {e}")
                
                # 聚合多卡loss（在分布式训练中）
                aggregated_loss = self._aggregate_loss(loss)
                epoch_loss += aggregated_loss
                
                # 优化数据集指标更新 - 降低频率以减少开销
                if self.enable_dataset_metrics and (self.current_step % 10 == 0):  # 每10步更新一次而不是每步
                    self._update_dataset_metrics(batch, outputs, aggregated_loss)
                
                # 获取梯度范数
                grad_norm = self.model.get_global_grad_norm()
                
                # 更新参数
                self.model.step()
                
                # 记录训练指标（准备数据）
                current_lr = self.optimizer.param_groups[0]['lr']
                # 确保grad_norm是float类型，避免JSON序列化错误
                # 处理grad_norm可能为None的情况
                if grad_norm is None:
                    grad_norm_value = 0.0
                elif hasattr(grad_norm, 'item'):
                    grad_norm_value = float(grad_norm.item())
                else:
                    grad_norm_value = float(grad_norm)
                
                # 检查是否是有效步骤（完成了梯度累积）
                is_effective_step = self.current_step % gradient_accumulation_steps == 0
                
                if is_effective_step:
                    effective_step += 1
                    
                    # 降低进度条更新频率以减少开销（每10个有效步骤更新一次）
                    if effective_step % 10 == 0:
                        pbar.update(10)  # 一次更新10步
                        pbar.set_postfix({
                            'loss': f'{aggregated_loss:.4f}',
                            'lr': f'{current_lr:.2e}',
                            'epoch': f'{epoch + batch_idx/len(self.train_loader):.2f}'
                        })
                    
                    # 优化监控记录 - 仅在必要时传递real_time_flops
                    step_real_time_flops = real_time_flops if should_measure_flops and real_time_flops is not None else None
                    self.monitor.log_step(effective_step, epoch, aggregated_loss, grad_norm_value, current_lr, attention_mask, step_real_time_flops)
                
                    # 详细日志记录（基于有效步数判断输出频率）
                    if effective_step % logging_steps == 0:
                        # 记录各数据集的指标
                        self._log_dataset_metrics(effective_step, is_eval=False)
                        
                        # 基础日志信息
                        log_message = (
                            f"Step {effective_step:,} | "
                            f"Loss: {aggregated_loss:.4f} | "
                            f"Grad Norm: {grad_norm_value:.4f} | "
                            f"LR: {current_lr:.2e} | "
                            f"Epoch: {epoch + batch_idx/len(self.train_loader):.2f}"
                        )
                        
                        # 如果进行了实时FLOPs测量，添加MFU信息
                        if should_measure_flops and hasattr(self.monitor, 'actual_flops') and self.monitor.actual_flops:
                            # 计算当前步骤的时间（从上次记录到现在）
                            current_time = time.time()
                            actual_step_time = current_time - self.monitor.step_start_time
                            
                            current_seq_length = self.monitor._calculate_actual_seq_length(attention_mask)
                            # 使用实际的批次大小（考虑分布式训练）
                            actual_batch_size = inputs.size(0) * self.dist_ctx.world_size
                            current_mfu = calculate_mfu(self.model, actual_batch_size, current_seq_length, 
                                                      actual_step_time, self.monitor.actual_flops)
                            log_message += f" | MFU: {current_mfu:.1%}"
                            
                            if should_measure_flops:
                                log_message += " [📊实时测量]"
                        
                        # 打印日志信息
                        if self.dist_ctx.is_main_process:
                            pbar.write(log_message)
                    
                    # 定期评估（基于有效步数）
                    if effective_step > 0 and effective_step % eval_steps == 0:
                        # 暂时刷新进度条以避免输出冲突
                        pbar.clear()
                        
                        # 添加评估异常处理，避免NCCL超时导致训练中断
                        try:
                            eval_loss, eval_accuracy = self.evaluate(step=effective_step)
                            # 注意：评估结果已经在evaluate方法中记录到wandb了，无需重复记录
                        except Exception as eval_error:
                            if self.dist_ctx.is_main_process:
                                print(f"⚠️  评估过程出错: {eval_error}")
                                print("⚠️  跳过本次评估，继续训练...")
                            # 记录一个占位符的eval结果，避免wandb图表中断
                            try:
                                placeholder_eval_data = {
                                    "eval/overall_loss": 999.0,  # 使用明显的占位符值
                                    "eval/overall_accuracy": 0.0,
                                    "eval/evaluation_failed": 1.0,  # 标记评估失败
                                }
                                self.monitor.log_metrics(placeholder_eval_data, effective_step)
                            except:
                                pass  # 如果连记录都失败，就完全跳过
                        
                        self.model.train()
                        # 重新显示进度条
                        pbar.refresh()
                    
                    # 定期保存检查点（基于有效步数）
                    if effective_step > 0 and effective_step % save_steps == 0:
                        if not self.save_best_only:  # 只有在未启用"仅保存最佳模型"时才保存常规检查点
                            pbar.clear()
                            self.save_checkpoint(effective_step)
                            pbar.refresh()
                        elif self.dist_ctx.is_main_process:  # 如果启用了仅保存最佳模型，只显示信息
                            pbar.write(f"💡 仅保存最佳模型模式已启用，跳过步骤 {effective_step} 的常规检查点保存")
            
            # Epoch结束统计
            epoch_time = time.time() - epoch_start_time
            avg_loss = epoch_loss / len(self.train_loader)
            self.monitor.log_epoch(epoch, avg_loss, epoch_time, effective_step)
            
            # 使用tqdm.write()输出epoch统计信息
            epoch_message = (
                f"📊 Epoch {epoch+1}/{num_epochs} 完成 | "
                f"平均损失: {avg_loss:.4f} | "
                f"耗时: {epoch_time:.2f}秒 | "
                f"有效步数: {effective_step:,}"
            )
            if self.dist_ctx.is_main_process:
                pbar.write(epoch_message)
        
        pbar.close()
        
        # 训练结束前进行最终评估
        if self.dist_ctx.is_main_process:
            print("\n🎯 训练即将完成，进行最终评估...")
        eval_loss, eval_accuracy = self.evaluate(step=effective_step)
        
        # 保存最终检查点（如果未启用仅保存最佳模型）
        if not self.save_best_only:
            if self.dist_ctx.is_main_process:
                print(f"💾 保存最终检查点...")
            self.save_checkpoint(effective_step)
        elif self.dist_ctx.is_main_process:
            print(f"💡 仅保存最佳模型模式已启用，跳过最终检查点保存")
        
        # 进行完整评估（在最佳模型上）
        if self.full_eval_at_end:
            self.full_evaluation_on_best_model()
        
        if self.dist_ctx.is_main_process:
            print("🎉 训练完成！")
            print(f"📊 最终评估结果 - 损失: {eval_loss:.4f}, 准确率: {eval_accuracy:.4f}")
            if self.best_model_enabled:
                print(f"🏆 最佳模型 - {self.best_metric_name}: {self.best_metric_value:.4f} (步骤 {self.best_model_step})")
                print(f"🏆 最佳模型路径: {self.best_model_path}")
        
        # 记录最终评估结果并结束wandb run
        # 注意：如果是多数据集评估，结果已经在evaluate方法中记录了
        # 只有单数据集情况下才需要这里记录
        if not (self.dataset_configs and self.enable_dataset_metrics):
            self.monitor.log_evaluation(effective_step, eval_loss, eval_accuracy)
        self.monitor.save_logs()
        
        # 训练结束后进行最终清理
        if self.save_best_only and self.dist_ctx.is_main_process:
            self.dist_ctx.print_main("🧹 进行最终检查点清理...")
            self._cleanup_old_best_models()
        
        self.monitor.finish_training()
        
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if os.path.exists(checkpoint_path):
            self.model.load_checkpoint(checkpoint_path)
            self.dist_ctx.print_main(f"检查点加载成功: {checkpoint_path}")
        else:
            self.dist_ctx.print_main(f"检查点不存在: {checkpoint_path}")
            
    def get_training_stats(self):
        """获取训练统计信息"""
        return self.monitor.get_avg_metrics() 