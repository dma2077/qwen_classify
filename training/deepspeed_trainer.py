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
from .utils.monitor import TrainingMonitor, make_json_serializable
from data.dataloader import create_full_eval_dataloader

# 新增导入
from .utils.flops_calculate import MFUStats

class DeepSpeedTrainer:
    def __init__(self, config):
        # 假设配置已经通过prepare_config处理过
        self.config = config
        
        # 🔥 修复：设置端口配置，避免端口冲突
        import os
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29501'  # 使用29501端口，避免29500冲突
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        
        self.dist_ctx = DistributedContext()
        
        # 设置NCCL超时保护（在分布式训练时）
        if self.dist_ctx.world_size > 1:
            from .utils.distributed import setup_nccl_timeout_env
            setup_nccl_timeout_env()
            if self.dist_ctx.is_main_process:
                print("✅ 已设置NCCL超时保护")
        
        # 内存优化配置
        self.enable_gradient_checkpointing = config.get('training', {}).get('gradient_checkpointing', True)
        
        # 只在主进程创建完整的TrainingMonitor，非主进程使用DummyMonitor
        if self.dist_ctx.is_main_process:
            from training.utils.monitor import TrainingMonitor
            # 不再硬编码flops_profile_freq，让TrainingMonitor从配置文件中读取
            self.monitor = TrainingMonitor(self.config['output_dir'], config)
        else:
            from training.utils.monitor import DummyMonitor  
            self.monitor = DummyMonitor(self.config['output_dir'], config)
        
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
        
        # 缓存MFU计算结果，避免重复计算
        self._mfu_cache = {}
        
        # 新增：初始化MFU统计器，替换Profiler-based的MFU计算
        self.mfu_stats = None  # 延迟初始化，等获取到模型配置路径后再初始化
        
        # 🔥 新增：性能监控
        self.performance_stats = {
            'total_training_time': 0.0,
            'total_eval_time': 0.0,
            'memory_usage': [],
            'gpu_utilization': []
        }
        
    def setup_model(self, model, train_loader, val_loader, optimizer=None, lr_scheduler=None):
        """设置模型和相关组件"""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 🔥 新增：应用内存优化设置
        self._apply_memory_optimizations()
        
        # 创建优化器和调度器（如果未提供）
        if optimizer is None:
            from optimizer.optimizer import create_optimizer
            optimizer = create_optimizer(model, self.config)
        
        if lr_scheduler is None:
            from training.lr_scheduler import create_lr_scheduler
            # 计算steps_per_epoch - 基于总批次大小
            deepspeed_config = self._get_deepspeed_config()
            train_batch_size = deepspeed_config.get('train_batch_size', 256)
            dataset_size = len(train_loader.dataset)
            steps_per_epoch = dataset_size // train_batch_size
            if dataset_size % train_batch_size != 0:
                steps_per_epoch += 1  # 向上取整
            lr_scheduler = create_lr_scheduler(optimizer, self.config, steps_per_epoch)
        
        # 获取DeepSpeed配置
        deepspeed_config = self._get_deepspeed_config()
        
        # 初始化DeepSpeed
        self.model, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=deepspeed_config
        )
        
        if self.dist_ctx.is_main_process:
            print(f"✅ 模型初始化完成")
        
        # 设置monitor的model引用用于MFU计算
        self.monitor.set_model_ref(self.model)
        
    def _apply_memory_optimizations(self):
        """应用内存优化设置"""
        # 1. 梯度检查点
        if self.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # 2. 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def _get_deepspeed_config(self):
        """获取DeepSpeed配置"""
        deepspeed_config_path = self.config.get('deepspeed', '')
        
        # 验证配置文件路径
        if not deepspeed_config_path:
            raise ValueError("DeepSpeed配置文件路径未设置")
        
        if not os.path.exists(deepspeed_config_path):
            raise FileNotFoundError(f"DeepSpeed配置文件不存在: {deepspeed_config_path}")
        
        # 加载配置文件
        try:
            with open(deepspeed_config_path, 'r') as f:
                deepspeed_config = json.load(f)
        except Exception as e:
            raise ValueError(f"DeepSpeed配置文件解析失败: {e}")
        
        # 验证必要字段
        required_fields = ['train_batch_size', 'train_micro_batch_size_per_gpu']
        missing_fields = [field for field in required_fields if field not in deepspeed_config]
        if missing_fields:
            raise ValueError(f"DeepSpeed配置文件缺少必要字段: {missing_fields}")
        
        # 打印配置信息（仅主进程）
        if self.dist_ctx.is_main_process:
            print(f"🔧 DeepSpeed配置: {deepspeed_config['train_micro_batch_size_per_gpu']} x {deepspeed_config.get('gradient_accumulation_steps', 1)} = {deepspeed_config['train_batch_size']}")
        
        return deepspeed_config
        
    def _calculate_training_stats(self):
        """计算训练统计信息"""
        deepspeed_config = self._get_deepspeed_config()
        
        micro_batch_size_per_gpu = deepspeed_config.get('train_micro_batch_size_per_gpu', 1)
        gradient_accumulation_steps = deepspeed_config.get('gradient_accumulation_steps', 1)
        train_batch_size = deepspeed_config.get('train_batch_size', 32)
        
        dataset_size = len(self.train_loader.dataset)
        
        # 正确计算每epoch的有效步数：基于总批次大小
        effective_steps_per_epoch = dataset_size // train_batch_size
        if dataset_size % train_batch_size != 0:
            effective_steps_per_epoch += 1  # 向上取整
        
        total_effective_steps = effective_steps_per_epoch * self.config['training']['num_epochs']
        
        # DataLoader的步数（用于调试信息）
        dataloader_steps_per_epoch = len(self.train_loader)
        samples_per_gpu = dataloader_steps_per_epoch * micro_batch_size_per_gpu
        
        return {
            'micro_batch_size_per_gpu': micro_batch_size_per_gpu,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'train_batch_size': train_batch_size,
            'dataloader_steps_per_epoch': dataloader_steps_per_epoch,
            'effective_steps_per_epoch': effective_steps_per_epoch,
            'total_effective_steps': total_effective_steps,
            'dataset_size': dataset_size,
            'samples_per_gpu': samples_per_gpu
        }
        
    def _print_training_config(self, stats):
        """打印训练配置信息"""
        if not self.dist_ctx.is_main_process:
            return
            
        print("🚀 训练配置:")
        print(f"  • 数据集: {stats['dataset_size']:,} 样本")
        print(f"  • 批次: {stats['micro_batch_size_per_gpu']} x {stats['gradient_accumulation_steps']} = {stats['train_batch_size']}")
        print(f"  • 步数: {stats['effective_steps_per_epoch']:,} 步/epoch, 总计 {stats['total_effective_steps']:,} 步")
        
    def _prepare_batch_data(self, batch):
        """准备批次数据 - 优化版本"""
        # 🔥 优化：使用pin_memory和non_blocking加速数据传输
        device = self.dist_ctx.device
        
        inputs = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        
        forward_kwargs = {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels
        }
        
        # 检查并添加image_grid_thw参数
        if "image_grid_thw" in batch:
            forward_kwargs["image_grid_thw"] = batch["image_grid_thw"].to(device, non_blocking=True)
        
        # 添加多数据集支持的参数
        if "dataset_names" in batch:
            forward_kwargs["dataset_names"] = batch["dataset_names"]
        if "num_classes_list" in batch:
            forward_kwargs["num_classes_list"] = batch["num_classes_list"]
            
        return forward_kwargs, inputs, attention_mask, labels
        
    def _optimize_dataloader(self):
        """优化数据加载器设置"""
        # 设置DataLoader的优化参数
        if hasattr(self.train_loader, 'pin_memory'):
            self.train_loader.pin_memory = True
        
        if hasattr(self.train_loader, 'num_workers'):
            # 根据CPU核心数优化worker数量
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            optimal_workers = min(cpu_count, 16)  # 提高上限到16个workers
            self.train_loader.num_workers = optimal_workers
        
        # 设置预取因子
        if hasattr(self.train_loader, 'prefetch_factor'):
            self.train_loader.prefetch_factor = 2
        
    def _init_mfu_stats(self):
        """初始化MFU统计器"""
        if self.mfu_stats is not None:
            return True
            
        try:
            # 创建args对象，包含MFU计算所需的参数
            import argparse
            args = argparse.Namespace()
            args.logging_per_step = self.config.get('logging_steps', 20)
            
            # 智能查找模型配置文件
            config_path = None
            possible_model_dirs = [
                self.config.get('model', {}).get('model_path', ''),
                self.config.get('model', {}).get('model_name_or_path', ''),
                self.config.get('output_dir', './output'),
                './models',
                './checkpoints',
                '.'
            ]
            
            # 过滤空路径
            possible_model_dirs = [path for path in possible_model_dirs if path]
            
            for model_dir in possible_model_dirs:
                if os.path.exists(model_dir):
                    test_config_path = os.path.join(model_dir, "config.json")
                    if os.path.exists(test_config_path):
                        config_path = test_config_path
                        args.model_dir = model_dir
                        break
            
            if config_path is None:
                if self.dist_ctx.is_main_process:
                    print(f"⚠️ 未找到模型配置文件 config.json")
                    print("📁 搜索路径:")
                    for path in possible_model_dirs:
                        if os.path.exists(path):
                            try:
                                files = os.listdir(path)
                                print(f"  - {path}: {files[:5]}{'...' if len(files) > 5 else ''}")
                            except:
                                print(f"  - {path}: 无法访问")
                        else:
                            print(f"  - {path}: 不存在")
                return False
            
            self.mfu_stats = MFUStats(args)
            if self.dist_ctx.is_main_process:
                print(f"✅ MFU统计器初始化成功，使用配置: {config_path}")
            return True
            
        except Exception as e:
            if self.dist_ctx.is_main_process:
                print(f"❌ MFU统计器初始化失败: {e}")
                import traceback
                traceback.print_exc()
            return False
    
    def _collect_mfu_data(self, batch, inputs, attention_mask):
        """收集MFU计算所需的数据"""
        try:
            # 计算图像token数量
            num_image_tokens = 0
            if "pixel_values" in batch and batch["pixel_values"] is not None:
                # 估算图像token数量，根据具体模型可能需要调整
                pixel_values = batch["pixel_values"]
                if pixel_values.dim() >= 3:
                    # 假设每张图像产生固定数量的token，这里使用常见的配置
                    # 可以根据实际模型配置调整
                    batch_size = pixel_values.size(0)
                    # 一般VLM模型每张图像产生256-1024个token
                    tokens_per_image = 256  # 这个值可以根据实际模型调整
                    num_image_tokens = batch_size * tokens_per_image
            
            # 获取文本token数量
            num_tokens = attention_mask.sum().item() if attention_mask is not None else inputs.numel()
            
            # 样本数量
            num_samples = inputs.size(0)
            
            # 图像数量
            num_images = batch.get("num_images", 0)
            if num_images == 0 and "pixel_values" in batch and batch["pixel_values"] is not None:
                num_images = batch["pixel_values"].size(0)
            
            # 调用MFU统计器的set方法
            self.mfu_stats.set(
                num_image_tokens=num_image_tokens,
                num_tokens=num_tokens,
                num_samples=num_samples,
                num_images=num_images
            )
            
        except Exception as e:
            if self.dist_ctx.is_main_process:
                print(f"⚠️ 收集MFU数据失败: {e}")
        
    def _build_training_metrics(self, effective_step, epoch, aggregated_loss, current_lr, grad_norm_value, 
                               inputs, attention_mask, step_time):
        """构建训练指标"""
        training_data = {
            "training/loss": float(aggregated_loss),
            "training/lr": float(current_lr), 
            "training/epoch": float(epoch),
            "training/grad_norm": float(grad_norm_value),
            "step": int(effective_step)
        }
        
        # 🔥 修复：降低性能指标记录频率，确保能看到perf指标
        should_log_perf = (effective_step % 20 == 0)  # 每20步记录一次性能指标
        
        if should_log_perf:
            if step_time > 0:
                training_data.update({
                    "perf/step_time": float(step_time),
                    "perf/steps_per_second": float(1.0 / step_time),
                })
                
                # 使用新的MFU计算方式
                if self.mfu_stats is not None:
                    try:
                        # 检查是否有足够的数据进行MFU计算
                        tokens_for_mfu = self.mfu_stats.tokens_for_mfu
                        has_sufficient_data = (
                            tokens_for_mfu["num_tokens"] > 0 and 
                            tokens_for_mfu["num_samples"] > 0 and
                            effective_step >= self.mfu_stats.args.logging_per_step
                        )
                        
                        if not has_sufficient_data:
                            if self.dist_ctx.is_main_process and effective_step % 50 == 0:
                                print(f"🔄 MFU数据收集中 (step={effective_step}): "
                                      f"tokens={tokens_for_mfu['num_tokens']}, "
                                      f"samples={tokens_for_mfu['num_samples']}, "
                                      f"images={tokens_for_mfu['num_images']}")
                            return training_data
                        
                        # 获取MFU日志数据
                        mfu_log_dict = self.mfu_stats.mfu(step_time, effective_step)
                        
                        # 调试：检查MFU数据的完整性
                        if self.dist_ctx.is_main_process and effective_step % 50 == 0:
                            print(f"🔍 MFU原始数据 (step={effective_step}):")
                            for key, value in mfu_log_dict.items():
                                print(f"  {key}: {value}")
                        
                        # 确保所有MFU指标都是有效的数值
                        valid_mfu_data = {}
                        for key, value in mfu_log_dict.items():
                            if isinstance(value, (int, float)) and not (isinstance(value, float) and (value != value or value == float('inf') or value == float('-inf'))):
                                valid_mfu_data[key] = float(value)
                            else:
                                if self.dist_ctx.is_main_process:
                                    print(f"⚠️ 跳过无效MFU指标: {key}={value}")
                        
                        training_data.update(valid_mfu_data)
                        
                        # 添加额外的性能指标
                        current_seq_length = attention_mask.sum(dim=1).float().mean().item() if attention_mask is not None else 0
                        actual_batch_size = inputs.size(0) * self.dist_ctx.world_size
                        
                        training_data.update({
                            "perf/tokens_per_second": float(actual_batch_size * current_seq_length / step_time),
                            "perf/samples_per_second": float(actual_batch_size / step_time),
                            "perf/actual_seq_length": float(current_seq_length),
                            "perf/actual_batch_size": float(actual_batch_size),
                        })
                        
                        if self.dist_ctx.is_main_process and effective_step % 100 == 0:
                            print(f"📊 MFU指标摘要 (step={effective_step}): "
                                  f"MFU={valid_mfu_data.get('perf/mfu_per_step_per_gpu', 0):.4f}, "
                                  f"VIT_FLOPs={valid_mfu_data.get('perf/vit_flops_per_step_per_gpu', 0):.2f}T, "
                                  f"LLM_FLOPs={valid_mfu_data.get('perf/llm_flops_per_step_per_gpu', 0):.2f}T")
                                  
                    except Exception as mfu_error:
                        if self.dist_ctx.is_main_process:
                            print(f"⚠️ MFU计算失败 (step={effective_step}): {mfu_error}")
                            import traceback
                            traceback.print_exc()
                else:
                    if self.dist_ctx.is_main_process and effective_step % 100 == 0:
                        print(f"⚠️ MFU统计器未初始化 (step={effective_step})")
            else:
                # 如果步骤时间为0或负数，记录警告
                if self.dist_ctx.is_main_process:
                    print(f"⚠️ 步骤时间异常，跳过性能指标记录 (step={effective_step}, step_time={step_time})")
        else:
            # 调试信息：为什么跳过性能指标
            if effective_step % 100 == 0:  # 每100步输出一次
                print(f"⏭️  跳过性能指标记录 (step={effective_step}): 频率检查 {effective_step} % 20 != 0")
                
        # 验证MFU指标是否包含在training_data中
        if self.dist_ctx.is_main_process and effective_step % 100 == 0:
            mfu_metrics = [k for k in training_data.keys() if 'mfu' in k.lower() or 'flops' in k.lower()]
            if mfu_metrics:
                print(f"✅ MFU指标将记录到WandB (step={effective_step}): {mfu_metrics}")
            else:
                print(f"⚠️ 未找到MFU指标 (step={effective_step})")
                
        return training_data
        
    def _handle_effective_step(self, effective_step, epoch, batch_idx, aggregated_loss, current_lr, 
                              grad_norm_value, inputs, attention_mask, step_time, is_eval_step):
        """处理有效步骤的逻辑"""
        # 降低进度条更新频率以减少开销（每10个有效步骤更新一次）
        if effective_step % 10 == 0:
            self._update_progress_bar(effective_step, aggregated_loss, current_lr, epoch, batch_idx)
        
        # 记录训练指标到本地日志
        self.monitor.log_step(effective_step, epoch, aggregated_loss, grad_norm_value, current_lr, attention_mask, skip_wandb=is_eval_step)
        
        # 🔥 修复：确保所有步骤都记录training和perf指标到WandB
        if self.dist_ctx.is_main_process:
            # 检查是否需要记录training指标
            should_log_training = (effective_step % self.monitor.freq['training_log_freq'] == 0)
            
            if should_log_training:
                training_data = self._build_training_metrics(effective_step, epoch, aggregated_loss, current_lr, 
                                                           grad_norm_value, inputs, attention_mask, step_time)
                
                # 🔥 修复：简化记录逻辑，每个step都记录并commit
                self.monitor.log_metrics(training_data, effective_step, commit=True)
                
                # 添加调试输出
                training_metrics_list = [k for k in training_data.keys() if k.startswith('training/')]
                perf_metrics_list = [k for k in training_data.keys() if k.startswith('perf/')]
                if training_metrics_list or perf_metrics_list:
                    print(f"📊 Training指标已记录 (step={effective_step}): "
                          f"training={len(training_metrics_list)}, perf={len(perf_metrics_list)}")
                    if training_metrics_list:
                        print(f"   📈 Training指标: {training_metrics_list}")
                    if perf_metrics_list:
                        print(f"   ⚡ Perf指标: {perf_metrics_list}")
            else:
                # 调试输出：显示为什么跳过记录
                if effective_step % 100 == 0:  # 每100步输出一次，避免日志过多
                    print(f"⏭️  跳过training指标记录 (step={effective_step}): "
                          f"频率={self.monitor.freq['training_log_freq']}")
                
    def _update_progress_bar(self, effective_step, aggregated_loss, current_lr, epoch, batch_idx):
        """更新进度条"""
        if hasattr(self, 'pbar'):
            self.pbar.update(10)  # 一次更新10步
            self.pbar.set_postfix({
                'loss': f'{aggregated_loss:.4f}',
                'lr': f'{current_lr:.2e}',
                'epoch': f'{epoch + batch_idx/len(self.train_loader):.2f}'
            })
            
    def _handle_evaluation_step(self, effective_step, epoch, aggregated_loss, current_lr, grad_norm_value, 
                               inputs, attention_mask, step_time):
        """处理评估步骤"""
        # 暂时刷新进度条以避免输出冲突
        if hasattr(self, 'pbar'):
            self.pbar.clear()
        
        print(f"🔍 开始处理评估步骤 (step={effective_step})")
        
        # 添加评估异常处理，避免NCCL超时导致训练中断
        try:
            print(f"🔄 调用evaluate方法...")
            # 获取eval数据但不让evaluate方法记录到wandb
            eval_loss, eval_accuracy, eval_results = self.evaluate(step=effective_step, log_to_wandb=False, return_results=True)
            
            print(f"✅ Evaluate方法完成: eval_loss={eval_loss:.4f}, eval_accuracy={eval_accuracy:.4f}")
            
            # 构建完整的training数据（包括性能指标）
            current_training_data = self._build_training_metrics(effective_step, epoch, aggregated_loss, current_lr, 
                                                               grad_norm_value, inputs, attention_mask, step_time)
            
            # 准备eval数据
            eval_data = self._build_eval_metrics(eval_loss, eval_accuracy, eval_results)
            
            print(f"📊 构建的eval数据: {list(eval_data.keys())}")
            print(f"📊 Eval数据详情: {eval_data}")
            
            # 🔥 修复：确保eval指标正确记录到WandB
            if self.dist_ctx.is_main_process:
                print(f"🔧 开始记录eval指标到WandB...")
                # 记录eval指标，强制commit确保数据同步
                self.monitor.log_metrics(eval_data, effective_step, commit=True)
                
                # 输出详细的记录信息
                eval_metrics_list = [k for k in eval_data.keys() if k.startswith('eval/')]
                
                print(f"✅ Eval指标已记录到WandB (step={effective_step})")
                print(f"   📊 记录的eval指标: {eval_metrics_list}")
                print(f"   📈 整体准确率: {eval_accuracy:.4f}")
                print(f"   📉 整体损失: {eval_loss:.6f}")
                print(f"   🔢 eval指标数量: {len(eval_data)}")
                
                # 验证eval指标记录
                if eval_metrics_list:
                    print(f"   ✅ Eval指标记录成功")
                else:
                    print(f"   ⚠️ 没有找到eval指标")
            else:
                print(f"⚠️ 非主进程，跳过eval指标记录")
                
        except Exception as eval_error:
            if self.dist_ctx.is_main_process:
                print(f"❌ 评估过程出错: {eval_error}")
                print(f"   effective_step: {effective_step}")
                print(f"   epoch: {epoch}")
                print(f"   aggregated_loss: {aggregated_loss}")
                print(f"   current_lr: {current_lr}")
                print("⚠️  跳过本次评估，继续训练...")
                import traceback
                traceback.print_exc()
            # 记录一个占位符的eval结果，避免wandb图表中断
            self._log_placeholder_eval(effective_step, aggregated_loss, current_lr)
        
        self.model.train()
        # 重新显示进度条
        if hasattr(self, 'pbar'):
            self.pbar.refresh()
            
    def _build_eval_metrics(self, eval_loss, eval_accuracy, eval_results):
        """构建评估指标 - 确保包含所有必要的eval指标"""
        eval_data = {
            "eval/overall_loss": float(eval_loss),
            "eval/overall_accuracy": float(eval_accuracy),
        }
        
        # 添加整体样本数和正确数（如果eval_results中有）
        if eval_results:
            overall_samples = eval_results.get('total_samples', 0)
            overall_correct = eval_results.get('total_correct', 0)
            if overall_samples > 0:
                eval_data["eval/overall_samples"] = int(overall_samples)
                eval_data["eval/overall_correct"] = int(overall_correct)
        
        # 添加每个数据集的详细指标（如果存在）
        if eval_results and 'dataset_metrics' in eval_results and eval_results['dataset_metrics']:
            for dataset_name, metrics in eval_results['dataset_metrics'].items():
                eval_data[f"eval/{dataset_name}_loss"] = float(metrics['loss'])
                eval_data[f"eval/{dataset_name}_accuracy"] = float(metrics['accuracy'])
                eval_data[f"eval/{dataset_name}_samples"] = int(metrics['samples'])
                eval_data[f"eval/{dataset_name}_correct"] = int(metrics['correct'])
                
        return eval_data
        
    def _log_placeholder_eval(self, effective_step, aggregated_loss, current_lr):
        """记录占位符评估结果"""
        try:
            placeholder_eval_data = {
                "training/loss": float(aggregated_loss),
                "training/lr": float(current_lr),
                "eval/overall_loss": 999.0,  # 使用明显的占位符值
                "eval/overall_accuracy": 0.0,
                "eval/evaluation_failed": 1.0,  # 标记评估失败
                "step": int(effective_step)
            }
            self.monitor.log_metrics(placeholder_eval_data, effective_step)
        except Exception as placeholder_error:
            print(f"❌ 记录占位符eval结果失败: {placeholder_error}")
            print(f"   effective_step: {effective_step}")
            print(f"   aggregated_loss: {aggregated_loss}")
            print(f"   current_lr: {current_lr}")
            import traceback
            traceback.print_exc()
            
    def _handle_logging_step(self, effective_step, aggregated_loss, grad_norm_value, current_lr, epoch, batch_idx, inputs, attention_mask):
        """处理日志记录步骤"""
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
        if hasattr(self.monitor, 'actual_flops') and self.monitor.actual_flops:
            current_time = time.time()
            step_start_time = getattr(self.monitor, 'step_start_time', None)
            if step_start_time is not None:
                actual_step_time = current_time - step_start_time
                
                current_mfu = self._calculate_mfu(effective_step, inputs, attention_mask, actual_step_time)
                if current_mfu is not None:
                    log_message += f" | MFU: {current_mfu:.1%}"
                    log_message += " [📊实时测量]"
        
        # 打印日志信息
        if self.dist_ctx.is_main_process and hasattr(self, 'pbar'):
            self.pbar.write(log_message)
            
    def _handle_save_step(self, effective_step):
        """处理保存步骤"""
        if not self.save_best_only:  # 只有在未启用"仅保存最佳模型"时才保存常规检查点
            if hasattr(self, 'pbar'):
                self.pbar.clear()
            self.save_checkpoint(effective_step)
            if hasattr(self, 'pbar'):
                self.pbar.refresh()
        elif self.dist_ctx.is_main_process:  # 如果启用了仅保存最佳模型，只显示信息
            if hasattr(self, 'pbar'):
                self.pbar.write(f"💡 仅保存最佳模型模式已启用，跳过步骤 {effective_step} 的常规检查点保存")
                
    def _train_epoch(self, epoch, stats):
        """训练一个epoch - 优化版本"""
        self.current_epoch = epoch
        self.model.train()
        
        # 🔥 新增：优化数据加载器
        if epoch == 0:  # 只在第一个epoch优化
            self._optimize_dataloader()
        
        # 为分布式采样器设置epoch（确保每个epoch的shuffle正确）
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)
        
        epoch_loss = 0
        epoch_start_time = time.time()
        effective_step = epoch * stats['effective_steps_per_epoch']
        
        # 🔥 新增：性能监控
        epoch_performance = {
            'forward_time': 0.0,
            'backward_time': 0.0,
            'optimizer_time': 0.0,
            'data_loading_time': 0.0,
            'memory_usage': []
        }
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch_start_time = time.time()
            self.current_step += 1
            
            # 🔥 新增：数据加载时间监控
            data_loading_time = time.time() - batch_start_time
            epoch_performance['data_loading_time'] += data_loading_time
            
            # 准备批次数据
            forward_kwargs, inputs, attention_mask, labels = self._prepare_batch_data(batch)
            
            # 🔥 新增：前向传播时间监控
            forward_start = time.time()
            outputs = self.model(**forward_kwargs)
            loss = outputs.loss
            epoch_performance['forward_time'] += time.time() - forward_start
            
            # 🔥 新增：反向传播时间监控
            backward_start = time.time()
            self.model.backward(loss)
            epoch_performance['backward_time'] += time.time() - backward_start
            
            # 聚合多卡loss（在分布式训练中）
            aggregated_loss = self._aggregate_loss(loss)
            epoch_loss += aggregated_loss
            
            # 优化数据集指标更新 - 降低频率以减少开销
            if self.enable_dataset_metrics and (self.current_step % 10 == 0):
                self._update_dataset_metrics(batch, outputs, aggregated_loss)
            
            # 收集MFU统计数据
            if self.mfu_stats is not None:
                self._collect_mfu_data(batch, inputs, attention_mask)
            
            # 🔥 新增：优化器时间监控
            optimizer_start = time.time()
            grad_norm = self.model.get_global_grad_norm()
            self.model.step()
            epoch_performance['optimizer_time'] += time.time() - optimizer_start
            
            # 处理梯度范数
            grad_norm_value = self._process_grad_norm(grad_norm)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 检查是否是有效步骤（完成了梯度累积）
            is_effective_step = self.current_step % stats['gradient_accumulation_steps'] == 0
            
            if is_effective_step:
                effective_step += 1
                
                # 计算步骤时间 - 修复None值问题
                current_time = time.time()
                step_start_time = getattr(self.monitor, 'step_start_time', None)
                if step_start_time is not None:
                    step_time = current_time - step_start_time
                else:
                    step_time = 0.0
                
                # 判断是否为评估步骤
                is_eval_step = (effective_step % self.config['eval_steps'] == 0)
                
                # 处理有效步骤
                self._handle_effective_step(effective_step, epoch, batch_idx, aggregated_loss, current_lr, 
                                          grad_norm_value, inputs, attention_mask, step_time, is_eval_step)
                
                # 详细日志记录
                if effective_step % self.config['logging_steps'] == 0:
                    self._handle_logging_step(effective_step, aggregated_loss, grad_norm_value, current_lr, 
                                            epoch, batch_idx, inputs, attention_mask)
                
                # 定期评估
                if effective_step > 0 and effective_step % self.config['eval_steps'] == 0:
                    if self.dist_ctx.is_main_process:
                        print(f"\n🎯 触发评估步骤 (step={effective_step}, eval_steps={self.config['eval_steps']})")
                    self._handle_evaluation_step(effective_step, epoch, aggregated_loss, current_lr, 
                                               grad_norm_value, inputs, attention_mask, step_time)
                
                # 定期保存检查点
                if effective_step > 0 and effective_step % self.config['save_steps'] == 0:
                    self._handle_save_step(effective_step)
            
            # 🔥 新增：定期内存清理
            if batch_idx % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # 🔥 新增：记录内存使用情况
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                epoch_performance['memory_usage'].append(memory_allocated)
        
        # Epoch结束统计
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / len(self.train_loader)
        self.monitor.log_epoch(epoch, avg_loss, epoch_time, effective_step)
        
        # 🔥 新增：记录性能统计
        self._log_performance_stats(epoch, epoch_performance, epoch_time)
        
        # 输出epoch统计信息
        epoch_message = (
            f"📊 Epoch {epoch+1}/{self.config['training']['num_epochs']} 完成 | "
            f"平均损失: {avg_loss:.4f} | "
            f"耗时: {epoch_time:.2f}秒 | "
            f"有效步数: {effective_step:,}"
        )
        if self.dist_ctx.is_main_process and hasattr(self, 'pbar'):
            self.pbar.write(epoch_message)
            
        return effective_step
        
    def _log_performance_stats(self, epoch, performance, total_time):
        """记录性能统计信息"""
        if not self.dist_ctx.is_main_process:
            return
            
        # 计算平均内存使用
        avg_memory = sum(performance['memory_usage']) / len(performance['memory_usage']) if performance['memory_usage'] else 0
        
        # 计算各阶段时间占比
        total_compute_time = performance['forward_time'] + performance['backward_time'] + performance['optimizer_time']
        data_loading_ratio = performance['data_loading_time'] / total_time * 100
        compute_ratio = total_compute_time / total_time * 100
        
        performance_data = {
            f"perf/epoch_{epoch}_total_time": total_time,
            f"perf/epoch_{epoch}_forward_time": performance['forward_time'],
            f"perf/epoch_{epoch}_backward_time": performance['backward_time'],
            f"perf/epoch_{epoch}_optimizer_time": performance['optimizer_time'],
            f"perf/epoch_{epoch}_data_loading_time": performance['data_loading_time'],
            f"perf/epoch_{epoch}_avg_memory_gb": avg_memory,
            f"perf/epoch_{epoch}_data_loading_ratio": data_loading_ratio,
            f"perf/epoch_{epoch}_compute_ratio": compute_ratio,
            "step": epoch * len(self.train_loader)
        }
        
        # 🔥 修复：暂时禁用性能统计记录，避免step冲突
        # self.monitor.log_metrics(performance_data, epoch * len(self.train_loader), commit=True)
        
        if self.dist_ctx.is_main_process:
            print(f"🔧 Epoch {epoch} 性能统计:")
            print(f"  • 总耗时: {total_time:.2f}s")
            print(f"  • 前向传播: {performance['forward_time']:.2f}s")
            print(f"  • 反向传播: {performance['backward_time']:.2f}s")
            print(f"  • 优化器: {performance['optimizer_time']:.2f}s")
            print(f"  • 数据加载: {performance['data_loading_time']:.2f}s ({data_loading_ratio:.1f}%)")
            print(f"  • 平均内存: {avg_memory:.2f}GB")
            print(f"  • 计算占比: {compute_ratio:.1f}%")
            
    def _process_grad_norm(self, grad_norm):
        """处理梯度范数"""
        if grad_norm is None:
            return 0.0
        elif hasattr(grad_norm, 'item'):
            return float(grad_norm.item())
        else:
            return float(grad_norm)
            
    def train(self):
        """训练模型 - 优化版本"""
        self.dist_ctx.print_main("开始训练...")
        self.monitor.start_training()
        
        # 计算训练统计信息
        stats = self._calculate_training_stats()
        
        # 打印训练配置信息
        self._print_training_config(stats)
        
        # 🔥 初始化FLOPs profiling，确保MFU能够正确记录
        if self.dist_ctx.is_main_process:
            self.dist_ctx.print_main("🔍 初始化FLOPs profiling...")
            try:
                # 获取第一个batch进行FLOPs profiling
                first_batch = next(iter(self.train_loader))
                forward_kwargs, inputs, attention_mask, labels = self._prepare_batch_data(first_batch)
                
                # 进行FLOPs profiling
                batch_example = {
                    "input_ids": inputs,
                    "attention_mask": attention_mask,
                    "pixel_values": forward_kwargs.get("pixel_values"),
                    "labels": labels
                }
                
                self.monitor.profile_model_flops(batch_example)
                self.dist_ctx.print_main("✅ FLOPs profiling完成，MFU计算已启用")
                
            except Exception as flops_error:
                self.dist_ctx.print_main(f"❌ FLOPs profiling失败: {flops_error}")
                self.dist_ctx.print_main(f"   first_batch类型: {type(first_batch)}")
                self.dist_ctx.print_main(f"   batch_example: {batch_example}")
                self.dist_ctx.print_main("⚠️ MFU计算将被禁用")
                import traceback
                traceback.print_exc()
        
        # 🔥 新增：初始化MFU统计器
        self.dist_ctx.print_main("🔧 初始化MFU统计器...")
        mfu_init_success = self._init_mfu_stats()
        
        # 在分布式环境中同步初始化状态
        if hasattr(self.dist_ctx, 'world_size') and self.dist_ctx.world_size > 1:
            import torch.distributed as dist
            # 广播初始化状态
            if dist.is_initialized():
                success_tensor = torch.tensor([1 if mfu_init_success else 0], dtype=torch.int, device=torch.cuda.current_device())
                dist.broadcast(success_tensor, src=0)
                mfu_init_success = bool(success_tensor.item())
        
        if mfu_init_success:
            self.dist_ctx.print_main("✅ MFU统计器初始化成功")
        else:
            self.dist_ctx.print_main("❌ MFU统计器初始化失败，将使用传统方法")
        
        # 创建进度条（基于有效训练步数）
        self.pbar = tqdm(total=stats['total_effective_steps'], desc="Training Steps", disable=not self.dist_ctx.is_main_process)
        
        # 训练循环
        try:
            for epoch in range(self.config['training']['num_epochs']):
                effective_step = self._train_epoch(epoch, stats)
        except KeyboardInterrupt:
            self.dist_ctx.print_main("⚠️ 训练被用户中断")
        except Exception as training_error:
            self.dist_ctx.print_main(f"❌ 训练过程中发生错误: {training_error}")
            raise training_error
        finally:
            self.pbar.close()
        
        # 训练结束处理
        self._finish_training(effective_step)
        


    def _finish_training(self, effective_step):
        """完成训练"""
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
        
        # 确保最终评估结果被记录到WandB
        self._log_final_evaluation(effective_step, eval_loss, eval_accuracy)
        
        # 训练结束后进行最终清理
        if self.save_best_only and self.dist_ctx.is_main_process:
            self.dist_ctx.print_main("🧹 进行最终检查点清理...")
            self._cleanup_old_best_models()
        
        self.monitor.finish_training()
        
    def _log_final_evaluation(self, effective_step, eval_loss, eval_accuracy):
        """记录最终评估结果"""
        try:
            final_eval_data = {
                "eval/final_overall_loss": eval_loss,
                "eval/final_overall_accuracy": eval_accuracy,
                "eval/final_evaluation": 1.0  # 标记这是最终评估
            }
            self.monitor.log_metrics(final_eval_data, effective_step, commit=True)
            self.dist_ctx.print_main(f"✅ 最终评估结果已记录到WandB")
        except Exception as final_eval_error:
            self.dist_ctx.print_main(f"❌ 最终评估WandB记录失败: {final_eval_error}")
            self.dist_ctx.print_main(f"   effective_step: {effective_step}")
            self.dist_ctx.print_main(f"   eval_loss: {eval_loss}")
            self.dist_ctx.print_main(f"   eval_accuracy: {eval_accuracy}")
            import traceback
            traceback.print_exc()
        self.monitor.save_logs()

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
            
            # 🔥 修复：暂时禁用最佳模型记录，避免step冲突
            # 记录到wandb
            # self.monitor.log_metrics({
            #     'best_model_step': step,
            #     f'best_{self.best_metric_name}': current_value
            # }, step)
            
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
        """记录各数据集的指标 - 优化版本，减少WandB记录频率"""
        if not self.enable_dataset_metrics or not self.dataset_metrics:
            return
        
        # 大幅降低数据集指标记录频率，避免WandB step冲突
        should_log_dataset = (step % 200 == 0)  # 每200步记录一次
        if not should_log_dataset:
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
            
            # 只在主进程输出详细信息（降低输出频率）
            if self.dist_ctx.is_main_process and (step % 500 == 0):  # 每500步输出一次
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
            
            if self.dist_ctx.is_main_process and (step % 500 == 0):  # 每500步输出一次
                prefix = "EVAL" if is_eval else "TRAIN"
                print(f"📊 {prefix} - OVERALL: "
                      f"Acc={overall_accuracy:.4f} ({overall_accuracy*100:.2f}%), "
                      f"Samples={overall_samples}")
        
        # 🔥 修复：暂时禁用数据集指标记录，避免step冲突
        # 记录到wandb时使用commit=True，确保数据同步
        # if dataset_log_data:
        #     self.monitor.log_metrics(dataset_log_data, step, commit=True)
            
        # 如果不是eval模式，重置训练指标
        if not is_eval:
            self.dataset_metrics.clear()
    

        
    def evaluate(self, step=None, log_to_wandb=True, return_results=False):
        """评估模型，根据数据集数量选择最优评估策略
        
        Args:
            step: 当前步数，如果提供则用于最佳模型保存；否则使用self.current_step
            log_to_wandb: 是否记录到WandB，默认为True
            return_results: 是否返回详细的评估结果，默认为False
        """
        current_step = step if step is not None else self.current_step
        
        try:
            self.dist_ctx.print_main("🔍 开始评估...")
            
            # 添加评估前的barrier，确保所有进程同步
            from .utils.distributed import safe_barrier
            if not safe_barrier():
                self.dist_ctx.print_main("⚠️  评估前同步失败，跳过本次评估")
                return 0.0, 0.0
            
            # 🔥 优化：根据数据集数量选择评估策略
            dataset_count = len(self.dataset_configs) if self.dataset_configs else 0
            
            if dataset_count <= 1:
                # 单数据集：使用快速评估函数
                self.dist_ctx.print_main("🚀 使用快速单数据集评估")
                from .utils.evaluation import evaluate_single_dataset_fast
                eval_loss, eval_accuracy = evaluate_single_dataset_fast(self.model, self.val_loader, self.dist_ctx.device)
                
                # 构造兼容的结果格式
                eval_results = {
                    'overall_loss': eval_loss,
                    'overall_accuracy': eval_accuracy,
                    'dataset_metrics': {},
                    'total_samples': len(self.val_loader.dataset),
                    'total_correct': int(eval_accuracy * len(self.val_loader.dataset))
                }
            else:
                # 多数据集：使用完整的多数据集评估函数
                self.dist_ctx.print_main("📊 使用多数据集评估")
                eval_results = evaluate_multi_dataset(self.model, self.val_loader, self.dist_ctx.device, self.dataset_configs)
            
            # 检查评估结果是否有效
            if eval_results is None or not eval_results:
                self.dist_ctx.print_main("⚠️  评估结果为空，跳过本次评估")
                return 0.0, 0.0
            
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
                # 单数据集情况：使用整体指标
                self.dist_ctx.print_main(f"📊 检测到单数据集评估结果")
                overall_samples = eval_results.get('total_samples', 0)
                overall_correct = eval_results.get('total_correct', 0)
            
            # 计算整体指标
            overall_accuracy = overall_correct / overall_samples if overall_samples > 0 else 0
            overall_loss = eval_results.get('overall_loss', 0)
            
            # 添加整体指标到wandb数据
            eval_log_data["eval/overall_loss"] = overall_loss
            eval_log_data["eval/overall_accuracy"] = overall_accuracy
            eval_log_data["eval/overall_samples"] = overall_samples
            eval_log_data["eval/overall_correct"] = overall_correct
            
            # 输出整体结果
            self.dist_ctx.print_main("=" * 80)
            self.dist_ctx.print_main(f"🎯 整体准确率: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
            self.dist_ctx.print_main(f"📈 整体损失:   {overall_loss:.6f}")
            self.dist_ctx.print_main(f"📊 总样本数:   {overall_samples:,}")
            self.dist_ctx.print_main(f"✅ 正确样本:   {overall_correct:,}")
            self.dist_ctx.print_main("=" * 80)
            
            # 记录到WandB - 🔥 修复：确保eval指标正确记录
            if current_step is not None and log_to_wandb:
                try:
                    # 确保所有eval指标都有正确的step字段
                    eval_log_data_with_step = eval_log_data.copy()
                    eval_log_data_with_step["step"] = current_step
                    
                    # 一次性记录所有eval指标，避免step冲突
                    self.monitor.log_metrics(eval_log_data, current_step, commit=True)
                    
                    # 输出详细的记录信息
                    eval_metrics_list = [k for k in eval_log_data.keys() if k.startswith('eval/')]
                    self.dist_ctx.print_main(f"✅ 评估指标已记录到WandB (step={current_step})")
                    self.dist_ctx.print_main(f"   📊 记录的eval指标: {eval_metrics_list}")
                    self.dist_ctx.print_main(f"   📈 整体准确率: {overall_accuracy:.4f}")
                    self.dist_ctx.print_main(f"   📉 整体损失: {overall_loss:.6f}")
                    
                except Exception as wandb_error:
                    self.dist_ctx.print_main(f"⚠️  WandB记录失败: {wandb_error}")
                    import traceback
                    traceback.print_exc()
            elif current_step is not None and not log_to_wandb:
                # 静默模式，不输出额外信息
                pass
            else:
                self.dist_ctx.print_main(f"📊 评估完成但未记录到WandB (step=None)")
            
            # 更新最佳模型 - 只在step不为None时更新
            if current_step is not None:
                try:
                    eval_results_for_best = {
                        'overall_loss': overall_loss,
                        'overall_accuracy': overall_accuracy
                    }
                    self._update_best_model(eval_results_for_best, current_step)
                except Exception as best_model_error:
                    self.dist_ctx.print_main(f"⚠️  最佳模型更新失败: {best_model_error}")
            else:
                self.dist_ctx.print_main(f"📊 跳过最佳模型更新 (step=None)")
            
            # 返回整体指标
            self.dist_ctx.print_main(f"✅ 评估结束 - 验证损失: {overall_loss:.4f}, 准确率: {overall_accuracy:.4f}")
            
            if return_results:
                return overall_loss, overall_accuracy, eval_results
            else:
                return overall_loss, overall_accuracy
            
        except Exception as eval_error:
            # 简化的错误处理
            self.dist_ctx.print_main(f"❌ 评估过程出错: {eval_error}")
            self.dist_ctx.print_main("🔄 跳过本次评估，继续训练...")
            return 0.0, 0.0
    
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