import time
import json
import os
import torch
import psutil
from typing import Dict, List, Optional

# 添加wandb支持
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. pip install wandb to enable logging.")

def make_json_serializable(obj):
    """确保对象可以JSON序列化"""
    if isinstance(obj, torch.Tensor):
        return float(obj.item()) if obj.numel() == 1 else obj.tolist()
    elif torch.is_tensor(obj):
        # 处理其他torch tensor类型
        return float(obj.item()) if obj.numel() == 1 else obj.tolist()
    elif hasattr(obj, 'dtype') and 'torch' in str(type(obj)):
        # 处理torch的标量类型
        if 'float' in str(obj.dtype):
            return float(obj)
        elif 'int' in str(obj.dtype):
            return int(obj)
        else:
            return float(obj)
    elif isinstance(obj, (float, int)):
        return obj
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

def get_gpu_peak_flops():
    """获取GPU峰值FLOPs性能"""
    try:
        if not torch.cuda.is_available():
            return 312e12  # 默认值
        
        # 获取GPU名称
        gpu_name = torch.cuda.get_device_name(0).upper()
        
        # 不同GPU的峰值性能 (TFLOPs for FP16/BF16)
        gpu_peak_flops = {
            # NVIDIA A100系列
            'A100': 312e12,    # A100 80GB
            'A100-SXM': 312e12,
            'A100-PCIE': 312e12,
            # NVIDIA A800系列 (针对中国市场的A100变体)
            'A800': 280e12,    # A800 80GB (稍低于A100)
            'A800-SXM': 280e12,
            'A800-PCIE': 280e12,
            # NVIDIA H100系列  
            'H100': 989e12,    # H100 80GB
            'H100-SXM': 989e12,
            'H100-PCIE': 756e12,
            # NVIDIA H800系列 (针对中国市场的H100变体)
            'H800': 850e12,    # H800 80GB (稍低于H100)
            'H800-SXM': 850e12,
            'H800-PCIE': 700e12,
            # NVIDIA V100系列
            'V100': 112e12,    # V100 32GB
            'V100-SXM': 112e12,
            'V100-PCIE': 112e12,
            # NVIDIA RTX系列
            'RTX 4090': 165e12,
            'RTX 4080': 112e12,
            'RTX 3090': 71e12,
            'RTX 3080': 58e12,
            # NVIDIA T4
            'T4': 65e12,
            # NVIDIA L4
            'L4': 121e12,
        }
        
        # 查找匹配的GPU
        for gpu_model, peak_flops in gpu_peak_flops.items():
            if gpu_model in gpu_name:
                # 只在第一次识别时打印（避免频繁输出）
                if not hasattr(get_gpu_peak_flops, '_gpu_identified'):
                    print(f"✅ 识别GPU: {gpu_name} -> {gpu_model} ({peak_flops/1e12:.0f} TFLOPs)")
                    get_gpu_peak_flops._gpu_identified = True
                return peak_flops
        
        # 如果没有找到匹配的GPU，使用默认值
        print(f"未识别的GPU类型: {gpu_name}，使用默认峰值性能 (A100: 312 TFLOPs)")
        return 312e12  # 默认使用A100的性能
        
    except Exception as e:
        print(f"获取GPU峰值性能错误: {e}")
        return 312e12  # 默认值

def calculate_mfu(model, batch_size: int, seq_length: int, step_time: float, actual_flops: float = None) -> float:
    """计算MFU (Model FLOPs Utilization)
    
    MFU = 实际FLOPs/s / GPU峰值FLOPs/s
    
    参数:
        model: 模型实例
        batch_size: 批次大小
        seq_length: 实际序列长度（包含visual tokens + text tokens）
        step_time: 步骤耗时（秒）
        actual_flops: 实际测量的FLOPs（包含前向+反向传播）
    
    返回:
        mfu: Model FLOPs Utilization (0-1之间的值)
    """
    try:
        if actual_flops is None:
            # 如果没有提供实际FLOPs，返回0
            return 0.0
        
        # 计算实际FLOPs/s
        actual_flops_per_second = actual_flops / step_time
        
        # 动态获取GPU峰值性能
        peak_flops_per_second = get_gpu_peak_flops()
        
        # 计算MFU
        mfu = actual_flops_per_second / peak_flops_per_second
        return min(mfu, 1.0)  # 限制在100%以内
        
    except Exception as e:
        print(f"MFU计算错误: {e}")
        return 0.0

def profile_model_flops(model, batch_example: Dict) -> float:
    """使用PyTorch profiler获取模型实际FLOPs（包括前向+反向传播）"""
    try:
        # 确保模型在训练模式
        model.train()
        
        # 获取实际的序列长度（包括visual tokens + text tokens）
        actual_seq_length = _get_actual_sequence_length(model, batch_example)
        
        # 分别测量前向传播和反向传播的FLOPs
        forward_flops = _profile_forward_flops(model, batch_example)
        backward_flops = _profile_backward_flops(model, batch_example)
        
        total_flops = forward_flops + backward_flops
        
        if total_flops > 0:
            print(f"文本tokens长度: {batch_example['input_ids'].size(1)}")
            print(f"实际序列长度(包含visual tokens): {actual_seq_length}")
            print(f"前向传播FLOPs: {forward_flops:.2e}")
            print(f"反向传播FLOPs: {backward_flops:.2e}")
            print(f"总FLOPs: {total_flops:.2e}")
        else:
            print("无法通过profiler测量FLOPs，使用估算方法")
            total_flops = _estimate_flops_fallback(model, batch_example, actual_seq_length)
        
        return float(total_flops)
        
    except Exception as e:
        print(f"FLOPs profiling错误: {e}")
        # 尝试获取序列长度用于估算
        try:
            actual_seq_length = _get_actual_sequence_length(model, batch_example)
            return _estimate_flops_fallback(model, batch_example, actual_seq_length)
        except:
            return _estimate_flops_fallback(model, batch_example)

def _profile_forward_flops(model, batch_example: Dict) -> float:
    """测量前向传播的FLOPs"""
    try:
        model.eval()  # 使用eval模式避免dropout等影响FLOPs计算
        
        # 检查PyTorch版本是否支持with_flops
        try:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,
                profile_memory=False
            ) as prof:
                with torch.no_grad():
                    # 仅执行前向传播
                    outputs = model(**batch_example)
            
            # 获取FLOPs统计
            flops = 0
            for event in prof.events():
                if hasattr(event, 'flops') and event.flops > 0:
                    flops += event.flops
            
            return float(flops)
            
        except (AttributeError, TypeError) as e:
            print(f"PyTorch profiler不支持with_flops参数: {e}")
            return 0.0
        
    except Exception as e:
        print(f"前向传播FLOPs测量错误: {e}")
        return 0.0

def _profile_backward_flops(model, batch_example: Dict) -> float:
    """测量反向传播的FLOPs"""
    try:
        model.train()  # 训练模式
        
        # 先执行前向传播（不在profiler中）
        outputs = model(**batch_example)
        loss = outputs.loss
        
        # 检查PyTorch版本是否支持with_flops
        try:
            # 测量反向传播的FLOPs
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,
                profile_memory=False
            ) as prof:
                # 仅执行反向传播
                loss.backward()
            
            # 清理梯度
            model.zero_grad()
            
            # 获取FLOPs统计
            flops = 0
            for event in prof.events():
                if hasattr(event, 'flops') and event.flops > 0:
                    flops += event.flops
            
            return float(flops)
            
        except (AttributeError, TypeError) as e:
            print(f"PyTorch profiler不支持with_flops参数: {e}")
            model.zero_grad()  # 确保清理梯度
            return 0.0
        
    except Exception as e:
        print(f"反向传播FLOPs测量错误: {e}")
        return 0.0

def _get_actual_sequence_length(model, batch_example: Dict) -> int:
    """获取实际的序列长度（包括visual tokens + text tokens）"""
    try:
        # 临时设置模型为eval模式以获取输出shape
        model.eval()
        
        with torch.no_grad():
            # 执行前向传播获取输出
            outputs = model(**batch_example)
            # 获取实际的序列长度
            actual_seq_length = outputs.last_hidden_state.size(1)
        
        # 恢复训练模式
        model.train()
        
        return actual_seq_length
        
    except Exception as e:
        print(f"获取实际序列长度错误: {e}")
        # 如果无法获取，使用文本长度作为近似
        return batch_example['input_ids'].size(1)

def _estimate_visual_tokens_count(batch_example: Dict) -> int:
    """估算视觉tokens的数量"""
    try:
        # 对于Qwen2.5-VL，visual tokens数量通常基于图像分辨率
        # 如果有image_grid_thw参数，使用它来计算
        if 'image_grid_thw' in batch_example:
            # image_grid_thw的形状通常是 [batch_size, 3]，其中3代表 [tiles, height, width]
            grid_thw = batch_example['image_grid_thw']
            if grid_thw.dim() == 2 and grid_thw.size(1) == 3:
                # 计算每个图像的visual tokens数量 (tiles * height * width)
                visual_tokens_per_image = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
                # 返回batch中第一个图像的visual tokens数量（假设batch内所有图像相同）
                return int(visual_tokens_per_image[0].item())
        
        # 如果没有image_grid_thw，使用默认估算
        # 对于标准的VL模型，一般一张图片会产生几百到几千个visual tokens
        # 这里使用一个保守的估算值
        if 'pixel_values' in batch_example and batch_example['pixel_values'] is not None:
            # 基于像素值的形状进行估算
            pixel_values = batch_example['pixel_values']
            if pixel_values.dim() >= 3:
                # 一般来说，visual tokens数量与图像分辨率相关
                # 这里使用一个简化的估算公式
                return 576  # 常见的vision transformer patch数量 (24*24)
        
        return 0  # 如果没有视觉输入，返回0
        
    except Exception as e:
        print(f"估算visual tokens数量错误: {e}")
        return 0

def _estimate_flops_fallback(model, batch_example: Dict, actual_seq_length: int = None) -> float:
    """备选的FLOPs估算方法（前向+反向传播）"""
    try:
        # 获取模型参数数量
        if hasattr(model, 'module'):
            param_count = sum(p.numel() for p in model.module.parameters())
        else:
            param_count = sum(p.numel() for p in model.parameters())
        
        # 获取batch size
        batch_size = batch_example['input_ids'].size(0)
        
        # 使用实际序列长度，如果没有提供则估算
        if actual_seq_length is not None:
            seq_length = actual_seq_length
        else:
            # 估算序列长度 = 文本tokens + visual tokens
            text_length = batch_example['input_ids'].size(1)
            visual_tokens = _estimate_visual_tokens_count(batch_example)
            seq_length = text_length + visual_tokens
        
        # 更准确的FLOPs估算（基于Transformer架构）
        # 前向传播FLOPs估算
        forward_flops = _estimate_forward_flops(param_count, batch_size, seq_length)
        
        # 反向传播FLOPs估算（通常是前向传播的2倍）
        backward_flops = 2 * forward_flops
        
        total_flops = forward_flops + backward_flops
        
        print(f"使用估算方法:")
        print(f"  参数数量: {param_count:.2e}")
        print(f"  批次大小: {batch_size}")
        print(f"  文本tokens长度: {batch_example['input_ids'].size(1)}")
        if actual_seq_length is not None:
            print(f"  实际序列长度: {seq_length}")
        else:
            estimated_visual = _estimate_visual_tokens_count(batch_example)
            print(f"  估算visual tokens: {estimated_visual}")
            print(f"  估算总序列长度: {seq_length}")
        print(f"  前向传播FLOPs: {forward_flops:.2e}")
        print(f"  反向传播FLOPs: {backward_flops:.2e}")
        print(f"  总FLOPs: {total_flops:.2e}")
        
        return float(total_flops)
        
    except Exception as e:
        print(f"FLOPs估算错误: {e}")
        return 0.0

def _estimate_forward_flops(param_count: int, batch_size: int, seq_length: int) -> float:
    """估算前向传播的FLOPs"""
    # 对于Transformer模型，前向传播的FLOPs主要来自：
    # 1. 矩阵乘法（线性层）
    # 2. 注意力机制
    # 3. 激活函数等
    
    # 简化估算：每个参数在前向传播中大约参与2次乘法运算
    # 对于multimodal模型，考虑视觉和文本的交互，使用稍高的系数
    flops_per_token = 2.5 * param_count
    total_flops = flops_per_token * batch_size * seq_length
    
    return float(total_flops)

def get_gpu_stats():
    """获取GPU状态信息"""
    gpu_stats = {}
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                # 获取GPU内存使用情况
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)    # GB
                memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                
                # 获取GPU利用率 (近似值，基于内存使用)
                memory_utilization = (memory_allocated / memory_total) * 100
                
                gpu_stats[f'gpu_{i}'] = {
                    'memory_allocated_gb': memory_allocated,
                    'memory_reserved_gb': memory_reserved,
                    'memory_total_gb': memory_total,
                    'memory_utilization_percent': memory_utilization
                }
                
                # 尝试获取GPU温度和功耗 (如果可用)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # W
                    gpu_stats[f'gpu_{i}']['temperature_c'] = temp
                    gpu_stats[f'gpu_{i}']['power_usage_w'] = power
                except:
                    pass  # pynvml不可用时跳过
                    
    except Exception as e:
        print(f"GPU状态获取错误: {e}")
    
    return gpu_stats

class TrainingMonitor:
    """训练监控器（支持wandb）"""
    
    def __init__(self, output_dir: str, config: Dict = None, log_file: str = "training_log.json"):
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, log_file)
        self.step_logs = []
        self.epoch_logs = []
        self.start_time = None
        self.step_start_time = None
        self.config = config or {}
        
        # 创建日志目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化wandb
        self.use_wandb = False
        self._init_wandb()
        
        # MFU计算相关参数
        self.model_ref = None
        self.seq_length = config.get('model', {}).get('max_sequence_length', 512)
        self.batch_size = config.get('train_batch_size', 32)
        self.actual_flops = None  # 存储实际测量的FLOPs
        self.actual_seq_length = None  # 存储实际的序列长度（包含visual tokens）
    
    def _is_main_process(self):
        """检查是否是主进程"""
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank() == 0
            return True  # 非分布式训练时默认为主进程
        except ImportError:
            return True
    
    def _init_wandb(self):
        """初始化wandb（仅在主进程中）"""
        if not WANDB_AVAILABLE:
            return
        
        wandb_config = self.config.get('wandb', {})
        if not wandb_config.get('enabled', False):
            print("wandb logging disabled in config")
            return
        
        # 检查是否是分布式训练，如果是则只在主进程中初始化wandb
        is_main_process = True
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                # 分布式训练中，只有rank 0进程初始化wandb
                is_main_process = dist.get_rank() == 0
                if not is_main_process:
                    print(f"进程 rank {dist.get_rank()}: 跳过wandb初始化（非主进程）")
                    return
        except ImportError:
            # 如果torch.distributed不可用，默认为主进程
            pass
        
        if not is_main_process:
            return
        
        try:
            # 只在主进程中初始化wandb
            wandb.init(
                project=wandb_config.get('project', 'qwen_classification'),
                name=wandb_config.get('run_name'),
                tags=wandb_config.get('tags', []),
                notes=wandb_config.get('notes'),
                config=self.config,
                resume="allow"  # 允许恢复
            )
            
            # 记录模型和训练配置
            try:
                if 'model' in self.config:
                    wandb.config.update({
                        'model_name': self.config['model'].get('pretrained_name', 'unknown'),
                        'num_labels': self.config['model'].get('num_labels', 'unknown')
                    })
                
                if 'training' in self.config:
                    wandb.config.update({
                        'learning_rate': self.config['training'].get('learning_rate', 'unknown'),
                        'num_epochs': self.config['training'].get('num_epochs', 'unknown'),
                        'batch_size': self.config.get('train_batch_size', 'unknown')
                    })
            except Exception as config_error:
                print(f"⚠️  wandb配置更新失败: {config_error}")
            
            self.use_wandb = True
            print("✅ wandb initialized successfully")
            
            # 显示wandb链接信息
            try:
                if wandb.run is not None:
                    print(f"📊 wandb project: {wandb.run.project}")
                    print(f"🔗 wandb run: {wandb.run.name}")
                    print(f"🚀 View run at: {wandb.run.url}")
                    
                    # 构建项目链接
                    if hasattr(wandb.run, 'entity') and hasattr(wandb.run, 'project'):
                        project_url = f"https://wandb.ai/{wandb.run.entity}/{wandb.run.project}"
                        print(f"⭐ View project at: {project_url}")
            except Exception as display_error:
                print(f"⚠️  wandb链接显示失败: {display_error}")
            
        except Exception as e:
            print(f"❌ Failed to initialize wandb: {e}")
            self.use_wandb = False
    
    def set_model_ref(self, model):
        """设置模型引用，用于MFU计算"""
        self.model_ref = model
    
    def profile_model_flops(self, batch_example: Dict):
        """测量模型的实际FLOPs"""
        if self.model_ref is None:
            print("模型引用未设置，无法测量FLOPs")
            return
        
        print("正在测量模型实际FLOPs...")
        self.actual_flops = profile_model_flops(self.model_ref, batch_example)
        
        # 同时获取实际序列长度
        try:
            self.actual_seq_length = _get_actual_sequence_length(self.model_ref, batch_example)
            print(f"✅ 实际序列长度: {self.actual_seq_length}")
        except Exception as e:
            print(f"❌ 获取序列长度失败: {e}")
        
        if self.actual_flops > 0:
            print(f"✅ 模型实际FLOPs: {self.actual_flops:.2e}")
            
            # 显示MFU计算相关信息
            self._show_mfu_calculation_info()
        else:
            print("❌ FLOPs测量失败，MFU计算将被禁用")
    
    def _show_mfu_calculation_info(self):
        """显示MFU计算的详细信息"""
        try:
            if self.actual_flops is None or self.actual_flops <= 0:
                return
            
            print(f"\n📊 MFU计算配置信息:")
            print(f"  • 批次大小: {self.batch_size}")
            print(f"  • 实际序列长度: {self.actual_seq_length}")
            print(f"  • 实际FLOPs: {self.actual_flops:.2e}")
            
            # 获取GPU峰值性能
            peak_flops = get_gpu_peak_flops()
            print(f"  • GPU峰值性能: {peak_flops:.2e} FLOPs/s")
            
            # 估算一个样本的MFU (假设1秒的step time)
            sample_step_time = 1.0
            sample_mfu = calculate_mfu(self.model_ref, self.batch_size, self.actual_seq_length, sample_step_time, self.actual_flops)
            print(f"  • 理论最大MFU (1秒/步): {sample_mfu:.4f} ({sample_mfu*100:.2f}%)")
            
            # 计算达到目标MFU所需的步骤时间
            target_mfus = [0.1, 0.2, 0.3, 0.5]
            print(f"  • 达到目标MFU所需的步骤时间:")
            for target_mfu in target_mfus:
                required_time = self.actual_flops / (target_mfu * peak_flops)
                print(f"    - {target_mfu*100:.0f}% MFU: {required_time:.3f}秒/步")
                
        except Exception as e:
            print(f"显示MFU信息错误: {e}")
    
    def set_actual_flops(self, flops: float, seq_length: int = None):
        """设置实际FLOPs（用于分布式训练中的同步）"""
        self.actual_flops = flops
        if seq_length is not None:
            self.actual_seq_length = seq_length
    
    def _calculate_actual_seq_length(self, attention_mask):
        """动态计算当前batch的实际序列长度"""
        try:
            if attention_mask is None:
                return self.seq_length
            
            # 计算每个样本的有效长度
            valid_lengths = attention_mask.sum(dim=1)  # [batch_size]
            
            # 使用批次中的平均有效长度（或最大长度）
            # 这里使用平均值，因为MFU通常关心的是整体吞吐量
            avg_seq_length = valid_lengths.float().mean().item()
            
            return int(avg_seq_length)
            
        except Exception as e:
            print(f"计算实际序列长度错误: {e}")
            return self.actual_seq_length if self.actual_seq_length is not None else self.seq_length
    
    def start_training(self):
        """开始训练"""
        self.start_time = time.time()
        self.step_start_time = time.time()
        
        if self.use_wandb and self._is_main_process():
            wandb.log({"training/started": True, "training/start_time": self.start_time})
    
    def log_step(self, step: int, epoch: int, loss: float, grad_norm: float, learning_rate: float, attention_mask=None, real_time_flops=None):
        """记录训练步骤"""
        current_time = time.time()
        step_time = current_time - self.step_start_time
        
        # 如果提供了实时FLOPs，更新当前FLOPs值
        if real_time_flops is not None and real_time_flops > 0:
            self.actual_flops = real_time_flops
        
        log_entry = {
            'step': int(step),
            'epoch': int(epoch),
            'loss': float(loss),
            'grad_norm': float(grad_norm),
            'learning_rate': float(learning_rate),
            'step_time': float(step_time),
            'timestamp': float(current_time)
        }
        
        # 如果有实时FLOPs，也记录到日志中
        if real_time_flops is not None:
            log_entry['real_time_flops'] = float(real_time_flops)
        
        self.step_logs.append(log_entry)
        
        # 记录到wandb（仅主进程）
        if self.use_wandb and self._is_main_process():
            # Training组 - 包含loss、grad_norm、lr
            wandb.log({
                "training/loss": float(loss),
                "training/grad_norm": float(grad_norm),
                "training/lr": float(learning_rate),
                "training/epoch": float(epoch),
                "global_step": int(step)
            }, step=int(step))
            
            # Perf组 - 包含MFU（使用实时数据）
            if self.model_ref is not None and self.actual_flops is not None:
                # 优先使用当前batch的实际序列长度
                if attention_mask is not None:
                    # 动态计算当前batch的实际序列长度
                    current_seq_length = self._calculate_actual_seq_length(attention_mask)
                elif self.actual_seq_length is not None:
                    # 使用之前测量的序列长度
                    current_seq_length = self.actual_seq_length
                else:
                    # 使用配置中的默认值
                    current_seq_length = self.seq_length
                
                # 使用最新的FLOPs值计算MFU
                current_flops = real_time_flops if real_time_flops is not None else self.actual_flops
                mfu = calculate_mfu(self.model_ref, self.batch_size, current_seq_length, step_time, current_flops)
                
                perf_logs = {
                    "perf/mfu": float(mfu),
                    "perf/step_time": float(step_time),
                    "perf/tokens_per_second": float(self.batch_size * current_seq_length / step_time),
                    "perf/actual_flops": float(current_flops),
                    "perf/actual_seq_length": float(current_seq_length)
                }
                
                # 如果有实时FLOPs，标记出来
                if real_time_flops is not None:
                    perf_logs["perf/real_time_measurement"] = 1.0
                    perf_logs["perf/flops_per_second"] = float(current_flops / step_time)
                else:
                    perf_logs["perf/real_time_measurement"] = 0.0
                
                wandb.log(perf_logs, step=int(step))
            
            # System组 - GPU状态 (每10步记录一次) - 已禁用单GPU指标，避免冗余
            # 注释掉单个GPU指标，减少WandB中的冗余信息
            # if step % 10 == 0:
            #     gpu_stats = get_gpu_stats()
            #     if gpu_stats:
            #         system_logs = {}
            #         for gpu_id, stats in gpu_stats.items():
            #             # 只记录GPU内存分配和利用率
            #             system_logs[f"system/{gpu_id}_memory_allocated_percent"] = stats['memory_utilization_percent']
            #             system_logs[f"system/{gpu_id}_memory_allocated_gb"] = stats['memory_allocated_gb']
            #         
            #         if system_logs:  # 只有当有有效数据时才记录
            #             wandb.log(system_logs, step=int(step))
        
        self.step_start_time = current_time
        
        # 定期保存本地日志
        if step % 100 == 0:
            self.save_logs()
    
    def log_epoch(self, epoch: int, avg_loss: float, elapsed_time: float):
        """记录epoch统计"""
        log_entry = {
            'epoch': int(epoch),
            'avg_loss': float(avg_loss),
            'elapsed_time': float(elapsed_time),
            'timestamp': float(time.time())
        }
        
        self.epoch_logs.append(log_entry)
        
        # 记录到wandb（仅主进程）
        if self.use_wandb and self._is_main_process():
            wandb.log({
                "training/epoch_avg_loss": float(avg_loss),
                "training/epoch_time": float(elapsed_time),
                "training/epoch_number": int(epoch)
            }, step=int(epoch))
        
        self.save_logs()
    
    def log_evaluation(self, step: int, eval_loss: float, eval_accuracy: float):
        """记录评估结果 - 在training组中显示accuracy"""
        if self.use_wandb and self._is_main_process():
            wandb.log({
                "training/eval_loss": float(eval_loss),
                "training/eval_accuracy": float(eval_accuracy),
                "global_step": int(step)
            }, step=int(step))
    
    def save_logs(self):
        """保存日志到文件"""
        try:
            logs = {
                'step_logs': self.step_logs,
                'epoch_logs': self.epoch_logs,
                'total_training_time': time.time() - self.start_time if self.start_time else 0
            }
            
            # 确保所有数据都可以JSON序列化
            serializable_logs = make_json_serializable(logs)
            
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_logs, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存日志失败: {e}")
    
    def finish_training(self):
        """结束训练"""
        if self.use_wandb and self._is_main_process():
            wandb.log({"training/finished": True, "training/total_time": time.time() - self.start_time})
            wandb.finish()
            print("📊 wandb run finished")
    
    def get_latest_metrics(self) -> Optional[Dict]:
        """获取最新的训练指标"""
        if self.step_logs:
            return self.step_logs[-1]
        return None
    
    def get_avg_metrics(self, last_n_steps: int = 100) -> Dict:
        """获取最近N步的平均指标"""
        if not self.step_logs:
            return {}
        
        recent_logs = self.step_logs[-last_n_steps:]
        
        if not recent_logs:
            return {}
        
        avg_loss = sum(log['loss'] for log in recent_logs) / len(recent_logs)
        avg_grad_norm = sum(log['grad_norm'] for log in recent_logs) / len(recent_logs)
        avg_step_time = sum(log['step_time'] for log in recent_logs) / len(recent_logs)
        
        return {
            'avg_loss': avg_loss,
            'avg_grad_norm': avg_grad_norm,
            'avg_step_time': avg_step_time,
            'num_steps': len(recent_logs)
        } 