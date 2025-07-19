import time
import json
import os
import torch
import psutil
from typing import Dict, List, Optional

# 监控频率配置 - 统一使用all_freq设置

# 添加wandb支持
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. pip install wandb to enable logging.")

# 全局缓存GPU峰值性能，避免重复识别
_GPU_PEAK_FLOPS_CACHE = None

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
    """获取GPU峰值FLOPs性能 - 优化版本，仅首次识别"""
    global _GPU_PEAK_FLOPS_CACHE
    
    # 如果已经缓存，直接返回
    if _GPU_PEAK_FLOPS_CACHE is not None:
        return _GPU_PEAK_FLOPS_CACHE
    
    try:
        if not torch.cuda.is_available():
            _GPU_PEAK_FLOPS_CACHE = 312e12  # 默认值
            return _GPU_PEAK_FLOPS_CACHE
        
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
                # 仅首次识别时打印
                print(f"✅ 识别GPU: {gpu_name} -> {gpu_model} ({peak_flops/1e12:.0f} TFLOPs)")
                _GPU_PEAK_FLOPS_CACHE = peak_flops
                return _GPU_PEAK_FLOPS_CACHE
        
        # 如果没有找到匹配的GPU，使用默认值
        print(f"⚠️  未识别的GPU类型: {gpu_name}，使用默认峰值性能 (A100: 312 TFLOPs)")
        _GPU_PEAK_FLOPS_CACHE = 312e12
        return _GPU_PEAK_FLOPS_CACHE
        
    except Exception as e:
        print(f"获取GPU峰值性能错误: {e}")
        _GPU_PEAK_FLOPS_CACHE = 312e12  # 默认值
        return _GPU_PEAK_FLOPS_CACHE

def calculate_mfu_with_profiler(model, batch_size: int, seq_length: int, step_time: float) -> float:
    """使用PyTorch Profiler计算MFU (Model FLOPs Utilization)
    
    MFU = 实际FLOPs/s / GPU峰值FLOPs/s
    
    参数:
        model: 模型实例
        batch_size: 批次大小
        seq_length: 实际序列长度
        step_time: 步骤耗时（秒）
    
    返回:
        mfu: Model FLOPs Utilization (0-1之间的值)
    """
    try:
        # 使用profiler测量FLOPs
        actual_flops = _measure_flops_with_profiler(model, batch_size, seq_length)
        
        if actual_flops <= 0:
            print("⚠️  Profiler无法测量FLOPs，返回0")
            return 0.0
        
        # 计算实际FLOPs/s
        actual_flops_per_second = actual_flops / step_time
        
        # 获取GPU峰值性能
        peak_flops_per_second = get_gpu_peak_flops()
        
        # 计算MFU
        mfu = actual_flops_per_second / peak_flops_per_second
        return min(mfu, 1.0)  # 限制在100%以内
        
    except Exception as e:
        print(f"Profiler MFU计算错误: {e}")
        return 0.0

def _measure_flops_with_profiler(model, batch_size: int, seq_length: int) -> float:
    """使用PyTorch Profiler测量FLOPs"""
    try:
        # 创建模拟的batch用于profiling
        device = next(model.parameters()).device
        dummy_batch = _create_dummy_batch_for_profiling(batch_size, seq_length, device)
        
        if not dummy_batch:
            print("⚠️  无法创建虚拟batch，跳过FLOPs测量")
            return 0.0
        
        model.eval()
        
        # 检查PyTorch版本和CUDA版本兼容性
        torch_version = torch.__version__
        cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"
        print(f"🔍 PyTorch版本: {torch_version}, CUDA版本: {cuda_version}")
        
        # 尝试使用profiler测量FLOPs
        try:
            # 首先尝试不使用with_flops参数
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=False
            ) as prof:
                with torch.no_grad():
                    _ = model(**dummy_batch)
            
            # 检查profiler是否正常工作
            try:
                events = prof.events()
                if events is not None and len(events) > 0:
                    print(f"✅ Profiler正常工作，获取到 {len(events)} 个事件")
                    
                    # 尝试获取FLOPs信息
                    total_flops = 0
                    flops_events = 0
                    
                    # 🔥 修复：安全地迭代events，避免TypeError
                    try:
                        events_list = list(events)  # 确保events是可迭代的
                        for event in events_list:
                            if hasattr(event, 'flops') and event.flops > 0:
                                total_flops += event.flops
                                flops_events += 1
                    except (TypeError, AttributeError) as iter_error:
                        print(f"⚠️  迭代events失败: {iter_error}")
                        print("🔧 使用估算方法")
                        return _estimate_flops_fallback(model, dummy_batch, seq_length)
                    
                    if total_flops > 0:
                        print(f"✅ 成功获取FLOPs: {total_flops:.2e} (来自 {flops_events} 个事件)")
                        return float(total_flops)
                    else:
                        print("⚠️  Profiler未检测到FLOPs，使用估算方法")
                else:
                    print("⚠️  Profiler events为空或None，使用估算方法")
            except Exception as events_error:
                print(f"⚠️  获取profiler events失败: {events_error}")
                print("🔧 使用估算方法")
                
        except Exception as profiler_error:
            print(f"Profiler执行错误: {profiler_error}")
            print("🔧 尝试使用估算方法")
        
        # 如果profiler失败，使用估算方法
        return _estimate_flops_fallback(model, dummy_batch, seq_length)
        
    except Exception as e:
        print(f"FLOPs测量完全失败: {e}")
        return 0.0
 
def _create_dummy_batch_for_profiling(batch_size: int, seq_length: int, device: torch.device) -> Dict:
    """创建用于profiling的虚拟batch"""
    try:
        # 创建虚拟的输入数据
        input_ids = torch.randint(0, 1000, (batch_size, seq_length), device=device)
        attention_mask = torch.ones(batch_size, seq_length, device=device)
        pixel_values = torch.randn(batch_size, 3, 224, 224, device=device)  # 假设图像尺寸
        labels = torch.randint(0, 10, (batch_size,), device=device)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels
        }
        
    except Exception as e:
        print(f"创建虚拟batch错误: {e}")
        return {}

def profile_model_flops(model, batch_example: Dict) -> float:
    """使用PyTorch profiler获取模型实际FLOPs（包括前向+反向传播）"""
    try:
        # 确保模型在训练模式
        model.train()
        
        # 获取实际的序列长度（包括visual tokens + text tokens）
        try:
            actual_seq_length = _get_actual_sequence_length(model, batch_example)
        except Exception as e:
            print(f"⚠️  获取序列长度失败: {e}，使用输入长度")
            actual_seq_length = batch_example['input_ids'].size(1)
        
        print(f"🔍 开始FLOPs profiling: 序列长度={actual_seq_length}")
        
        # 尝试使用profiler测量FLOPs
        try:
            # 测量前向传播FLOPs
            forward_flops = _profile_forward_flops(model, batch_example)
            
            # 测量反向传播FLOPs
            backward_flops = _profile_backward_flops(model, batch_example)
            
            total_flops = forward_flops + backward_flops
            
            if total_flops > 0:
                print(f"✅ Profiler FLOPs测量成功:")
                print(f"  前向传播FLOPs: {forward_flops:.2e}")
                print(f"  反向传播FLOPs: {backward_flops:.2e}")
                print(f"  总FLOPs: {total_flops:.2e}")
                return float(total_flops)
            else:
                print("⚠️  Profiler未获取到FLOPs，使用估算方法")
                
        except Exception as profiler_error:
            print(f"Profiler测量失败: {profiler_error}")
            print("🔧 使用估算方法作为备选")
        
        # 如果profiler失败，使用估算方法
        print("🔧 使用估算方法测量FLOPs")
        total_flops = _estimate_flops_fallback(model, batch_example, actual_seq_length)
        
        if total_flops > 0:
            print(f"文本tokens长度: {batch_example['input_ids'].size(1)}")
            print(f"实际序列长度(包含visual tokens): {actual_seq_length}")
            print(f"估算总FLOPs: {total_flops:.2e}")
        
        return float(total_flops)
        
    except Exception as e:
        print(f"FLOPs测量完全失败: {e}")
        # 最后的回退：使用最基本的估算
        try:
            return _estimate_flops_fallback(model, batch_example)
        except:
            print("❌ 所有FLOPs测量方法都失败，返回0")
            return 0.0

def _profile_forward_flops(model, batch_example: Dict) -> float:
    """测量前向传播的FLOPs"""
    try:
        model.eval()  # 使用eval模式避免dropout等影响FLOPs计算
        
        # 检查PyTorch版本兼容性
        torch_version = torch.__version__
        print(f"🔍 前向传播Profiler - PyTorch版本: {torch_version}")
        
        try:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=False
            ) as prof:
                with torch.no_grad():
                    # 仅执行前向传播
                    outputs = model(**batch_example)
            
            # 获取FLOPs统计
            flops = 0
            try:
                events = prof.events()
                print(f"🔍 前向传播Profiler - events类型: {type(events)}")
                
                if events is not None:
                    try:
                        events_length = len(events)
                        print(f"🔍 前向传播Profiler - events长度: {events_length}")
                        
                        if events_length > 0:
                            # 🔥 修复：安全地迭代events，避免TypeError
                            try:
                                events_list = list(events)  # 确保events是可迭代的
                                print(f"🔍 前向传播Profiler - 成功转换为list，长度: {len(events_list)}")
                                
                                for i, event in enumerate(events_list):
                                    if hasattr(event, 'flops') and event.flops > 0:
                                        flops += event.flops
                                        if i < 5:  # 只打印前5个有FLOPs的事件
                                            print(f"  Event {i}: flops={event.flops}")
                                
                                if flops > 0:
                                    print(f"✅ 前向传播FLOPs: {flops:.2e}")
                                    return float(flops)
                                else:
                                    print("⚠️  前向传播Profiler未检测到FLOPs")
                            except (TypeError, AttributeError) as iter_error:
                                print(f"⚠️  迭代前向传播events失败: {iter_error}")
                                print(f"  events类型: {type(events)}")
                                print(f"  events内容: {events}")
                                return 0.0
                        else:
                            print("⚠️  前向传播Profiler events为空")
                    except Exception as len_error:
                        print(f"⚠️  获取events长度失败: {len_error}")
                        print(f"  events类型: {type(events)}")
                        return 0.0
                else:
                    print("⚠️  前向传播Profiler events为None")
            except Exception as events_error:
                print(f"⚠️  获取前向传播profiler events失败: {events_error}")
                import traceback
                traceback.print_exc()
            
            return 0.0
            
        except Exception as e:
            print(f"前向传播Profiler错误: {e}")
            return 0.0
        
    except Exception as e:
        print(f"前向传播FLOPs测量错误: {e}")
        return 0.0

def _profile_backward_flops(model, batch_example: Dict) -> float:
    """测量反向传播的FLOPs"""
    try:
        model.train()  # 训练模式
        
        # 检查PyTorch版本兼容性
        torch_version = torch.__version__
        print(f"🔍 反向传播Profiler - PyTorch版本: {torch_version}")
        
        # 先执行前向传播（不在profiler中）
        outputs = model(**batch_example)
        loss = outputs.loss
        
        try:
            # 测量反向传播的FLOPs
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=False
            ) as prof:
                # 仅执行反向传播
                loss.backward()
            
            # 清理梯度
            model.zero_grad()
            
            # 获取FLOPs统计
            flops = 0
            try:
                events = prof.events()
                print(f"🔍 反向传播Profiler - events类型: {type(events)}")
                
                if events is not None:
                    try:
                        events_length = len(events)
                        print(f"🔍 反向传播Profiler - events长度: {events_length}")
                        
                        if events_length > 0:
                            # 🔥 修复：安全地迭代events，避免TypeError
                            try:
                                events_list = list(events)  # 确保events是可迭代的
                                print(f"🔍 反向传播Profiler - 成功转换为list，长度: {len(events_list)}")
                                
                                for i, event in enumerate(events_list):
                                    if hasattr(event, 'flops') and event.flops > 0:
                                        flops += event.flops
                                        if i < 5:  # 只打印前5个有FLOPs的事件
                                            print(f"  Event {i}: flops={event.flops}")
                                
                                if flops > 0:
                                    print(f"✅ 反向传播FLOPs: {flops:.2e}")
                                    return float(flops)
                                else:
                                    print("⚠️  反向传播Profiler未检测到FLOPs")
                            except (TypeError, AttributeError) as iter_error:
                                print(f"⚠️  迭代反向传播events失败: {iter_error}")
                                print(f"  events类型: {type(events)}")
                                print(f"  events内容: {events}")
                                return 0.0
                        else:
                            print("⚠️  反向传播Profiler events为空")
                    except Exception as len_error:
                        print(f"⚠️  获取events长度失败: {len_error}")
                        print(f"  events类型: {type(events)}")
                        return 0.0
                else:
                    print("⚠️  反向传播Profiler events为None")
            except Exception as events_error:
                print(f"⚠️  获取反向传播profiler events失败: {events_error}")
                import traceback
                traceback.print_exc()
            
            return 0.0
            
        except Exception as e:
            print(f"反向传播Profiler错误: {e}")
            model.zero_grad()  # 确保清理梯度
            return 0.0
        
    except Exception as e:
        print(f"反向传播FLOPs测量错误: {e}")
        return 0.0

def _get_actual_sequence_length(model, batch_example: Dict) -> int:
    """获取实际的序列长度（包括visual tokens + text tokens）"""
    try:
        # 🔥 修复：直接通过attention_mask计算实际序列长度
        # 这是最准确的方法，因为attention_mask覆盖了完整的序列（visual + text tokens）
        if 'attention_mask' in batch_example and batch_example['attention_mask'] is not None:
            attention_mask = batch_example['attention_mask']
            # 计算每个样本的有效长度，然后取平均值
            valid_lengths = attention_mask.sum(dim=1)  # [batch_size]
            actual_seq_length = int(valid_lengths.float().mean().item())
            return actual_seq_length
        else:
            # 如果没有attention_mask，使用输入长度作为近似
            actual_seq_length = batch_example['input_ids'].size(1)
            print(f"⚠️  没有attention_mask，使用输入长度: {actual_seq_length}")
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
    
    def __init__(self, output_dir: str, config: Dict = None, log_file: str = "training_log.json", flops_profile_freq: int = None):
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, log_file)
        self.step_logs = []
        self.epoch_logs = []
        self.start_time = None
        self.step_start_time = None
        self.config = config or {}
        
        # FLOPs profiling频率配置 - 如果未提供则从配置文件读取
        self.flops_profile_freq = flops_profile_freq
        
        # 初始化监控频率配置
        self._init_monitor_frequencies()
        
        # 创建日志目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化wandb
        self.use_wandb = False
        self._init_wandb()
        
        # MFU计算相关参数
        self.model_ref = None
        self.seq_length = config.get('model', {}).get('max_sequence_length', 512)
        
        # 正确获取batch_size - 优先从DeepSpeed配置获取
        self.batch_size = self._get_effective_batch_size(config)
        
        self.actual_flops = None  # 存储实际测量的FLOPs
        self.actual_seq_length = None  # 存储实际的序列长度（包含visual tokens）
        
        # 设置主进程标识 - 修复：不在这里设置，保持为方法
        
        print(f"📊 TrainingMonitor初始化: batch_size={self.batch_size}, flops_profile_freq={self.flops_profile_freq}")
    
    def _init_monitor_frequencies(self):
        """初始化监控频率配置 - 支持多种配置结构"""
        # 从config中获取monitor频率配置
        monitor_config = self.config.get('monitor', {})
        
        # 🔥 修复：支持多种配置结构
        # 1. 优先使用 'freq' 配置
        freq_config = monitor_config.get('freq', {})
        if not freq_config:
            # 2. 如果没有 'freq'，尝试使用 'all_freq'
            freq_config = monitor_config.get('all_freq', {})
        
        # 🔥 所有频率都从monitor.freq中独立设置
        self.freq = {
            'training_log_freq': freq_config.get('training_log_freq', 10),           # 训练指标记录频率
            'perf_log_freq': freq_config.get('perf_log_freq', 10),                   # 性能指标记录频率（降低到10步）
            'gpu_log_freq': freq_config.get('gpu_log_freq', 50),                     # GPU监控频率
            'local_save_freq': freq_config.get('local_save_freq', 200),              # 本地保存频率
            'progress_update_freq': freq_config.get('progress_update_freq', 10),     # 进度更新频率
            'eval_log_freq': freq_config.get('eval_log_freq', 1),                    # 评估指标记录频率
        }
        
        # flops_profile_freq配置 - 优先从配置文件读取，如果没有则使用构造函数传入的值或默认值
        config_flops_profile_freq = freq_config.get('flops_profile_freq')
        if config_flops_profile_freq is not None:
            # 配置文件中有设置，使用配置文件的值
            self.flops_profile_freq = config_flops_profile_freq
        elif self.flops_profile_freq is not None:
            # 构造函数传入了值，保持不变
            pass
        else:
            # 都没有设置，使用默认值
            self.flops_profile_freq = 500
        
        # 只在主进程输出关键监控配置
        if self._is_main_process():
            print(f"📊 监控频率: 训练{self.freq['training_log_freq']}步, 性能{self.freq['perf_log_freq']}步, 评估{self.freq['eval_log_freq']}步")
            print(f"   📂 配置来源: {'freq' if monitor_config.get('freq') else 'all_freq' if monitor_config.get('all_freq') else '默认值'}")
    
    def _get_effective_batch_size(self, config: Dict) -> int:
        """正确获取有效的batch size"""
        try:
            # 首先尝试从DeepSpeed配置获取
            deepspeed_config = config.get('deepspeed', {})
            if isinstance(deepspeed_config, str):
                # 如果是文件路径，读取文件
                import json
                with open(deepspeed_config, 'r') as f:
                    deepspeed_config = json.load(f)
            
            # 优先使用DeepSpeed的train_batch_size（这是真正的有效批次大小）
            if 'train_batch_size' in deepspeed_config:
                return deepspeed_config['train_batch_size']
            
            # 备选方案：从train_micro_batch_size_per_gpu计算
            if 'train_micro_batch_size_per_gpu' in deepspeed_config:
                micro_batch = deepspeed_config['train_micro_batch_size_per_gpu']
                gradient_accumulation = deepspeed_config.get('gradient_accumulation_steps', 1)
                
                # 计算世界大小
                try:
                    import torch.distributed as dist
                    if dist.is_available() and dist.is_initialized():
                        world_size = dist.get_world_size()
                    else:
                        world_size = 1
                except:
                    world_size = 1
                
                effective_batch_size = micro_batch * gradient_accumulation * world_size
                return effective_batch_size
            
            # 最后的备选方案：从根配置获取
            if 'train_batch_size' in config:
                return config['train_batch_size']
            
            # 默认值
            return 32
            
        except Exception as e:
            print(f"⚠️  获取batch_size失败: {e}，使用默认值32")
            return 32
    
    def _is_main_process(self):
        """检查是否是主进程"""
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                is_main = rank == 0
                return is_main
            else:
                return True  # 非分布式训练时默认为主进程
        except ImportError:
            return True
    
    def _init_wandb(self):
        """初始化WandB - 修复版本，避免step冲突"""
        try:
            # 🔥 修复：首先检查配置中是否启用了WandB
            wandb_config = self.config.get('wandb', {})
            wandb_enabled = wandb_config.get('enabled', False)
            
            # 🔥 关键修复：根据配置设置use_wandb标志
            if wandb_enabled and WANDB_AVAILABLE and self._is_main_process():
                self.use_wandb = True
            else:
                self.use_wandb = False
                if not wandb_enabled:
                    print("⚠️ WandB在配置中被禁用")
                elif not WANDB_AVAILABLE:
                    print("⚠️ WandB未安装，跳过WandB初始化")
                elif not self._is_main_process():
                    print("⚠️ 非主进程，跳过WandB初始化")
                return
            
            import wandb
            
            # 🔥 重要：检查是否已经有活跃的WandB运行
            if wandb.run is not None:
                print(f"⚠️ 检测到已存在的WandB运行: {wandb.run.name}")
                print(f"   🔗 URL: {wandb.run.url}")
                print(f"   📊 当前step: {getattr(wandb.run, 'step', 0)}")
                
                # 决定是否复用现有运行
                choice = "reuse"  # 默认复用，避免step冲突
                
                if choice == "reuse":
                    print("✅ 复用现有WandB运行")
                    # 仍需定义指标
                    self._define_eval_metrics()
                    return
                else:
                    print("🔄 结束现有运行，创建新运行")
                    wandb.finish()
            
            # 获取配置参数
            project = wandb_config.get('project', 'qwen_classification')
            run_name = wandb_config.get('run_name')
            if run_name is None:
                run_name = f'run_{int(time.time())}'
            tags = wandb_config.get('tags', [])
            notes = wandb_config.get('notes', '')
            
            print(f"🔧 开始初始化WandB...")
            print(f"   📊 项目: {project}")
            print(f"   🏃 运行名称: {run_name}")
            print(f"   🏷️ 标签: {tags}")
            
            # 🔥 关键修复：使用reinit=True确保干净的初始化
            wandb.init(
                project=project,
                name=run_name,
                tags=tags,
                notes=notes,
                config=self.config,
                dir=self.output_dir,
                reinit=True  # 确保干净的初始化
            )
            
            print(f"✅ WandB初始化成功")
            print(f"   🔗 URL: {wandb.run.url}")
            print(f"   🆔 Run ID: {wandb.run.id}")
            print(f"   📊 初始step: {getattr(wandb.run, 'step', 0)}")
            
            # 定义指标（这是安全的，不会影响step）
            self._define_eval_metrics()
            
            # 🔥 关键修复：完全避免记录任何初始数据
            print("✅ WandB初始化完成，等待真实数据（不记录初始数据）")
            
        except Exception as e:
            print(f"❌ WandB初始化失败: {e}")
            print(f"   配置: {self.config.get('wandb', {})}")
            print(f"   输出目录: {self.output_dir}")
            print(f"   WANDB_AVAILABLE: {WANDB_AVAILABLE}")
            print(f"   is_main_process: {self._is_main_process()}")
            import traceback
            traceback.print_exc()
            self.use_wandb = False
    
    def set_model_ref(self, model):
        """设置模型引用，用于MFU计算"""
        self.model_ref = model
    
    def _define_eval_metrics(self):
        """定义eval指标，确保wandb正确识别和显示 - 改进版本"""
        try:
            if not self.use_wandb or not self._is_main_process():
                return
            
            import wandb
            if wandb.run is None:
                return
            
            # 🔥 关键修复：分别定义training和eval指标，使用统一的x轴
            wandb.define_metric("step")
            
            # 定义训练指标组
            wandb.define_metric("training/loss", step_metric="step", summary="min")
            wandb.define_metric("training/lr", step_metric="step", summary="last")
            wandb.define_metric("training/epoch", step_metric="step", summary="last")
            wandb.define_metric("training/grad_norm", step_metric="step", summary="last")
            
            # 定义评估指标组 - 🔥 确保所有eval指标都被定义
            wandb.define_metric("eval/overall_loss", step_metric="step", summary="min")
            wandb.define_metric("eval/overall_accuracy", step_metric="step", summary="max")
            wandb.define_metric("eval/overall_samples", step_metric="step", summary="last")
            wandb.define_metric("eval/overall_correct", step_metric="step", summary="last")
            
            # 定义数据集特定的eval指标
            dataset_configs = self.config.get('datasets', {}).get('dataset_configs', {})
            for dataset_name in dataset_configs.keys():
                wandb.define_metric(f"eval/{dataset_name}_loss", step_metric="step", summary="min")
                wandb.define_metric(f"eval/{dataset_name}_accuracy", step_metric="step", summary="max")
                wandb.define_metric(f"eval/{dataset_name}_samples", step_metric="step", summary="last")
            
            # 定义最终评估指标
            wandb.define_metric("eval/final_overall_loss", step_metric="step", summary="min")
            wandb.define_metric("eval/final_overall_accuracy", step_metric="step", summary="max")
            wandb.define_metric("eval/final_overall_samples", step_metric="step", summary="last")
            wandb.define_metric("eval/final_overall_correct", step_metric="step", summary="last")
            
            # 定义性能指标组
            wandb.define_metric("perf/step_time", step_metric="step", summary="mean")
            wandb.define_metric("perf/steps_per_second", step_metric="step", summary="mean")
            wandb.define_metric("perf/mfu", step_metric="step", summary="mean")
            wandb.define_metric("perf/mfu_percent", step_metric="step", summary="mean")
            wandb.define_metric("perf/tokens_per_second", step_metric="step", summary="mean")
            wandb.define_metric("perf/samples_per_second", step_metric="step", summary="mean")
            wandb.define_metric("perf/actual_flops", step_metric="step", summary="last")
            wandb.define_metric("perf/actual_seq_length", step_metric="step", summary="last")
            wandb.define_metric("perf/flops_per_second", step_metric="step", summary="mean")
            
            # 使用通配符定义其他可能的指标
            wandb.define_metric("training/*", step_metric="step")
            wandb.define_metric("eval/*", step_metric="step")
            wandb.define_metric("perf/*", step_metric="step")
            
            print("✅ 已定义详细指标分组：training/*, eval/*, perf/* 指标使用统一的'step'轴")
            print(f"   📊 已定义的具体eval指标: overall_loss, overall_accuracy, overall_samples, overall_correct")
            if dataset_configs:
                print(f"   📂 已定义的数据集eval指标: {list(dataset_configs.keys())}")
            
        except Exception as e:
            print(f"⚠️  定义eval指标失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_eval_charts(self):
        """强制确保eval图表在wandb界面中可见"""
        try:
            if not self.use_wandb or not self._is_main_process():
                return
            
            import wandb
            if wandb.run is None:
                return
            
            # 🔥 新策略：不再强制记录占位符数据，而是依赖metric定义和首次真实eval数据
            # 这样可以避免图表中出现不相关的初始值
            
            # 准备eval指标列表（用于日志输出）
            eval_metrics_list = [
                "eval/overall_loss",
                "eval/overall_accuracy", 
                "eval/overall_samples",
                "eval/overall_correct"
            ]
            
            # 如果有多数据集配置，也添加对应的指标
            dataset_configs = self.config.get('datasets', {}).get('dataset_configs', {})
            for dataset_name in dataset_configs.keys():
                eval_metrics_list.extend([
                    f"eval/{dataset_name}_loss",
                    f"eval/{dataset_name}_accuracy",
                    f"eval/{dataset_name}_samples"
                ])
            
            print(f"📊 eval图表已准备就绪，等待首次评估数据 - 指标: {eval_metrics_list}")
            
        except Exception as e:
            print(f"⚠️  创建eval图表失败: {e}")
    
    def _force_create_eval_charts(self):
        """强制创建eval图表，确保eval指标能够显示"""
        try:
            if not self.use_wandb or not self._is_main_process():
                return
            
            import wandb
            if wandb.run is None:
                return
            
            print("🔧 强制创建eval图表...")
            
            # 记录一个初始的eval数据点，强制创建eval图表
            initial_eval_data = {
                "eval/overall_loss": 999.0,  # 使用明显的初始值
                "eval/overall_accuracy": 0.0,
                "eval/overall_samples": 0,
                "eval/overall_correct": 0
            }
            
            # 添加数据集特定的eval指标
            dataset_configs = self.config.get('datasets', {}).get('dataset_configs', {})
            for dataset_name in dataset_configs.keys():
                initial_eval_data[f"eval/{dataset_name}_loss"] = 999.0
                initial_eval_data[f"eval/{dataset_name}_accuracy"] = 0.0
                initial_eval_data[f"eval/{dataset_name}_samples"] = 0
            
            # 记录初始eval数据点
            wandb.log(initial_eval_data, step=0, commit=True)
            print("✅ 已记录初始eval数据点，强制创建eval图表")
            
            # 强制同步到云端
            try:
                wandb.run.sync()
                print("🔄 已强制同步初始eval数据到WandB云端")
            except Exception as sync_error:
                print(f"⚠️ 初始eval数据同步失败: {sync_error}")
            
        except Exception as e:
            print(f"⚠️ 强制创建eval图表失败: {e}")
    
    def _log_initial_data_points(self):
        """记录初始数据点，确保所有图表都能显示"""
        try:
            if not self.use_wandb or not self._is_main_process():
                return
            
            import wandb
            if wandb.run is None:
                return
            
            print("🔧 记录初始数据点...")
            
            # 记录初始training数据点
            initial_training_data = {
                "training/loss": 999.0,
                "training/lr": 0.0,
                "training/epoch": 0.0,
                "training/grad_norm": 0.0
            }
            
            # 记录初始perf数据点
            initial_perf_data = {
                "perf/step_time": 0.0,
                "perf/mfu": 0.0,
                "perf/tokens_per_second": 0.0
            }
            
            # 记录所有初始数据点
            all_initial_data = {**initial_training_data, **initial_perf_data}
            wandb.log(all_initial_data, step=0, commit=True)
            
            print("✅ 已记录初始数据点，确保所有图表都能显示")
            
            # 强制同步到云端
            try:
                wandb.run.sync()
                print("🔄 已强制同步初始数据到WandB云端")
            except Exception as sync_error:
                print(f"⚠️ 初始数据同步失败: {sync_error}")
            
        except Exception as e:
            print(f"⚠️ 记录初始数据点失败: {e}")

    def _create_detailed_charts(self):
        """创建详细的训练和评估指标图表 - 优化版本，不记录初始数据"""
        try:
            if not self.use_wandb or not self._is_main_process():
                return
            
            # 移除初始数据记录，避免step=0的问题
            # wandb会在第一次真实数据记录时自动创建图表
            print("✅ 图表将在实际数据记录时自动创建")
            
        except Exception as e:
            print(f"图表准备失败: {e}")

    def _ensure_eval_charts_visible(self):
        """确保eval图表在wandb界面中可见 - 优化版本，减少额外记录"""
        try:
            if not self.use_wandb or not self._is_main_process():
                return
            
            # 移除额外的chart_visibility_check记录，避免step冲突
            print("✅ eval图表将在第一次评估时自动显示")
            
        except Exception as e:
            print(f"⚠️  确保eval图表可见性失败: {e}")
    
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
            sample_mfu = calculate_mfu_with_profiler(self.model_ref, self.batch_size, self.actual_seq_length, sample_step_time)
            print(f"  • 理论最大MFU (1秒/步): {sample_mfu:.4f} ({sample_mfu*100:.2f}%)")
            
            # 计算达到目标MFU所需的步骤时间
            target_mfus = [0.1, 0.2, 0.3, 0.5]
            print(f"  • 达到目标MFU所需的步骤时间:")
            for target_mfu in target_mfus:
                required_time = self.actual_flops / (target_mfu * peak_flops)
                print(f"    - {target_mfu*100:.0f}% MFU: {required_time:.3f}秒/步")
                
        except Exception as e:
            print(f"❌ 显示MFU信息错误: {e}")
            print(f"   actual_flops: {self.actual_flops}")
            print(f"   batch_size: {self.batch_size}")
            print(f"   actual_seq_length: {self.actual_seq_length}")
            import traceback
            traceback.print_exc()
    
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
            print(f"❌ 计算实际序列长度错误: {e}")
            print(f"   attention_mask类型: {type(attention_mask)}")
            print(f"   attention_mask形状: {attention_mask.shape if attention_mask is not None else 'None'}")
            print(f"   actual_seq_length: {self.actual_seq_length}")
            print(f"   seq_length: {self.seq_length}")
            import traceback
            traceback.print_exc()
            return self.actual_seq_length if self.actual_seq_length is not None else self.seq_length
    
    def start_training(self):
        """开始训练 - 优化版本，减少初始记录"""
        self.start_time = time.time()
        self.step_start_time = time.time()
        
        # 移除training/started记录，减少WandB调用
        print("✅ 训练监控已启动")
    
    def log_step(self, step: int, epoch: int, loss: float, grad_norm: float, learning_rate: float, attention_mask=None, real_time_flops=None, skip_wandb=False):
        """记录训练步骤 - 修复WandB记录频率，确保perf和training组指标正常显示
        
        Args:
            skip_wandb: 如果为True，跳过wandb记录（用于eval步骤时避免重复记录）
        """
        current_time = time.time()
        # 修复step_start_time可能为None的问题
        if self.step_start_time is not None:
            step_time = current_time - self.step_start_time
        else:
            step_time = 0.0
        
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
        
        # 🔥 修复：log_step方法只负责本地日志记录，WandB记录由trainer统一处理
        # 这样可以避免重复记录和step冲突问题
        if self.use_wandb and self._is_main_process() and not skip_wandb:
            # 只记录基础信息到本地日志，WandB记录由trainer的_build_training_metrics处理
            pass
        
        self.step_start_time = current_time
        
        # 使用动态本地日志保存频率
        if step % self.freq['local_save_freq'] == 0:
            self.save_logs()
    
    def log_epoch(self, epoch: int, avg_loss: float, elapsed_time: float, current_step: int = None):
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
            log_data = {
                "training/epoch_avg_loss": float(avg_loss),
                "training/epoch_time": float(elapsed_time),
                "training/epoch_number": int(epoch)
            }
            
            # 修复：总是使用指定的step，避免WandB自动step
            if current_step is not None:
                wandb.log(log_data, step=int(current_step), commit=True)
            else:
                # 如果没有提供step，跳过记录，避免WandB自动step
                print("⚠️  log_epoch: 未提供current_step，跳过WandB记录")
        
        self.save_logs()
    
    def log_evaluation(self, step: int, eval_loss: float, eval_accuracy: float, additional_metrics: dict = None):
        """记录评估结果 - 在eval组中显示指标"""
        if self.use_wandb and self._is_main_process():
            try:
                import wandb
                if wandb.run is None:
                    print("⚠️ WandB未初始化，跳过eval指标记录")
                    return
                
                log_data = {
                    "eval/overall_loss": float(eval_loss),
                    "eval/overall_accuracy": float(eval_accuracy),
                }
                
                # 添加额外的指标
                if additional_metrics:
                    for key, value in additional_metrics.items():
                        # 确保额外指标也在eval组中
                        if not key.startswith('eval/'):
                            key = f"eval/{key}"
                        log_data[key] = float(value) if isinstance(value, (int, float)) else value
                
                wandb.log(log_data, step=int(step), commit=True)
                print(f"📊 评估指标已记录到WandB (step={step}): {list(log_data.keys())}")
                
            except Exception as e:
                print(f"❌ 记录eval指标失败: {e}")
                print(f"   step: {step}")
                print(f"   eval_loss: {eval_loss}")
                print(f"   eval_accuracy: {eval_accuracy}")
                print(f"   additional_metrics: {additional_metrics}")
                print(f"   use_wandb: {self.use_wandb}")
                print(f"   is_main_process: {self._is_main_process()}")
                import traceback
                traceback.print_exc()
    
    def log_metrics(self, metrics: dict, step: int = None, commit: bool = True):
        """通用的指标记录方法 - 彻底修复WandB step=0问题"""
        # 检查是否是主进程且wandb可用
        if not self.use_wandb or not self._is_main_process():
            return
            
        if not WANDB_AVAILABLE:
            return

        # 检查wandb是否已初始化
        try:
            import wandb
            if wandb.run is None:
                print("⚠️ WandB未初始化，跳过指标记录")
                return
        except Exception as e:
            print(f"❌ 导入WandB失败: {e}")
            import traceback
            traceback.print_exc()
            return

        try:
            # 确保所有值都是可序列化的
            log_data = {}
            eval_metrics_count = 0
            training_metrics_count = 0
            perf_metrics_count = 0
            eval_metrics_list = []
            training_metrics_list = []
            perf_metrics_list = []
            
            for key, value in metrics.items():
                # 跳过step字段，避免重复
                if key == "step":
                    continue
                    
                # 处理不同类型的值
                if isinstance(value, (int, float)):
                    log_data[key] = float(value)
                elif hasattr(value, 'item'):  # torch tensor
                    log_data[key] = float(value.item())
                else:
                    log_data[key] = value
                
                # 统计各类指标数量和名称
                if 'eval/' in key:
                    eval_metrics_count += 1
                    eval_metrics_list.append(key)
                elif 'training/' in key:
                    training_metrics_count += 1
                    training_metrics_list.append(key)
                elif 'perf/' in key:
                    perf_metrics_count += 1
                    perf_metrics_list.append(key)
            
            # 🔥 彻底修复：使用最可靠的step控制方法
            if step is not None and step >= 0:
                actual_step = int(step)
                
                # 🔥 方法1：直接使用step参数，这是最可靠的方法
                print(f"🔧 记录数据到step {actual_step}")
                wandb.log(log_data, step=actual_step, commit=commit)
                step_info = f"step={actual_step}"
                
                # 验证记录结果
                current_wandb_step = getattr(wandb.run, 'step', 0)
                print(f"🔍 记录后WandB step: {current_wandb_step}")
                
                if current_wandb_step == actual_step:
                    print(f"✅ Step记录成功: {actual_step}")
                else:
                    print(f"⚠️ Step可能不匹配: 期望{actual_step}, WandB显示{current_wandb_step}")
                    # 但这不一定是错误，因为WandB可能会在后台更新step
                    
            else:
                # 如果step为None或负数，使用自动step
                wandb.log(log_data, commit=commit)
                step_info = "auto-step"
                current_wandb_step = getattr(wandb.run, 'step', 0)
                print(f"🔍 自动step记录，WandB step: {current_wandb_step}")
            
            # 改进同步策略
            if commit and wandb.run is not None:
                try:
                    # 等待数据同步
                    import time
                    time.sleep(0.1)  # 增加等待时间
                    
                    print(f"🔄 WandB数据已提交 ({step_info})")
                except Exception as sync_error:
                    print(f"⚠️ WandB同步失败: {sync_error}")
            
            # 输出记录信息（调试用）
            if self._is_main_process() and (training_metrics_count > 0 or eval_metrics_count > 0 or perf_metrics_count > 0):
                print(f"📊 WandB记录完成 ({step_info}): "
                      f"training={training_metrics_count}, eval={eval_metrics_count}, perf={perf_metrics_count}")
                if training_metrics_count > 0:
                    print(f"   📈 Training指标: {training_metrics_list}")
                if eval_metrics_count > 0:
                    print(f"   📊 Eval指标: {eval_metrics_list}")
                if perf_metrics_count > 0:
                    print(f"   ⚡ Perf指标: {perf_metrics_list}")
                
                # 显示WandB状态信息
                try:
                    final_wandb_step = getattr(wandb.run, 'step', 0)
                    print(f"   🔍 最终WandB step: {final_wandb_step}")
                    print(f"   🔗 WandB URL: {wandb.run.url}")
                    print(f"   📊 WandB项目: {wandb.run.project}")
                    
                    # 检查WandB run状态
                    if hasattr(wandb.run, 'state'):
                        print(f"   🏃 WandB状态: {wandb.run.state}")
                    
                    # 检查summary数据
                    try:
                        if hasattr(wandb.run, 'summary') and wandb.run.summary:
                            summary_keys = list(wandb.run.summary.keys())
                            print(f"   📋 WandB summary: {len(summary_keys)}个指标")
                            
                            # 检查training指标是否存在
                            if training_metrics_list:
                                found_training = [k for k in training_metrics_list if k in summary_keys]
                                if found_training:
                                    print(f"   ✅ Training指标已确认: {found_training}")
                                else:
                                    print(f"   ⚠️ Training指标未在summary中找到")
                                    
                            # 检查eval指标是否存在  
                            if eval_metrics_list:
                                found_eval = [k for k in eval_metrics_list if k in summary_keys]
                                if found_eval:
                                    print(f"   ✅ Eval指标已确认: {found_eval}")
                                else:
                                    print(f"   ⚠️ Eval指标未在summary中找到")
                        else:
                            print(f"   ⚠️ WandB summary为空")
                    except Exception as summary_error:
                        print(f"   ⚠️ 检查summary失败: {summary_error}")
                    
                    print(f"   ✅ WandB记录流程完成")
                        
                except Exception as wandb_info_error:
                    print(f"   ⚠️ 获取WandB信息失败: {wandb_info_error}")
            
        except Exception as e:
            print(f"❌ 记录指标到WandB失败: {e}")
            print(f"   尝试记录的指标: {list(metrics.keys())}")
            print(f"   step: {step}")
            print(f"   commit: {commit}")
            print(f"   use_wandb: {self.use_wandb}")
            print(f"   is_main_process: {self._is_main_process()}")
            print(f"   WANDB_AVAILABLE: {WANDB_AVAILABLE}")
            
            import traceback
            traceback.print_exc()
    
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
            print(f"❌ 保存日志失败: {e}")
            print(f"   log_file: {self.log_file}")
            print(f"   output_dir: {self.output_dir}")
            print(f"   logs keys: {list(logs.keys()) if 'logs' in locals() else 'N/A'}")
            import traceback
            traceback.print_exc()
    
    def finish_training(self):
        """结束训练 - 优化版本，减少WandB调用"""
        if self.use_wandb and self._is_main_process():
            # 简化结束记录，只记录总时间
            total_time = time.time() - self.start_time if self.start_time else 0
            wandb.log({"training/total_time": total_time}, commit=True)
            wandb.finish()
            print(f"📊 训练完成，总耗时: {total_time:.2f}秒")
    
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

class DummyMonitor:
    """虚拟监控器，用于非主进程，避免不必要的wandb操作"""
    
    def __init__(self, output_dir: str, config: Dict = None, log_file: str = "training_log.json"):
        self.output_dir = output_dir
        self.config = config or {}
        self.use_wandb = False
        
        # 添加必要的属性
        self.actual_flops = None
        self.step_start_time = None
        self.batch_size = config.get('train_batch_size', 32) if config else 32
        
    def start_training(self):
        """空实现，非主进程不启动训练监控"""
        pass
        
    def log_step(self, step: int, epoch: int, loss: float, grad_norm: float, learning_rate: float, attention_mask=None, real_time_flops=None, skip_wandb=False):
        """空实现，非主进程不记录步骤"""
        pass
    
    def log_epoch(self, epoch: int, avg_loss: float, elapsed_time: float, current_step: int = None):
        """空实现，非主进程不记录epoch"""
        pass
    
    def log_evaluation(self, step: int, eval_loss: float, eval_accuracy: float, additional_metrics: dict = None):
        """空实现，非主进程不记录评估"""
        pass
    
    def log_metrics(self, metrics: dict, step: int = None, commit: bool = True):
        """空实现，非主进程不记录指标"""
        pass
    
    def log_dataset_metrics(self, is_eval: bool = True, step: int = None):
        """空实现，非主进程不记录数据集指标"""
        pass
    
    def set_actual_flops(self, flops: float, seq_length: int = None):
        """空实现，非主进程不设置FLOPs"""
        pass
    
    def _calculate_actual_seq_length(self, attention_mask):
        """空实现，返回默认值"""
        return 512  # 返回一个合理的默认值
    
    def save_logs(self):
        """空实现，非主进程不保存日志"""
        pass
    
    def finish_training(self):
        """空实现，非主进程不结束wandb"""
        pass
    
    def set_model_ref(self, model):
        """空实现，非主进程不设置模型引用"""
        pass
    
    def profile_model_flops(self, batch_example: Dict):
        """空实现，非主进程不计算FLOPs"""
        pass
    
    def get_latest_metrics(self):
        """空实现，返回None"""
        return None
    
    def get_avg_metrics(self, last_n_steps: int = 100):
        """空实现，返回空字典"""
        return {} 

 