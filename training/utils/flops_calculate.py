from functools import lru_cache
from recovlm.utils.ds_utils import format_dict_or_list
import platform
import subprocess
import os
import re
import collections
from collections import defaultdict

def s(x):
    if isinstance(x, list): return sum(x)
    else: return x

@lru_cache(maxsize=1)
def get_gpu_model():
    """
    获取当前系统中NVIDIA显卡的型号信息
    
    返回:
    str: 显卡型号名称，如果无法检测则返回 "Unknown"
    """
    try:
        # 优先尝试使用nvidia-smi（最可靠的方法）
        if platform.system() in ["Linux", "Darwin"]:  # Linux/macOS
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        
        elif platform.system() == "Windows":  # Windows
            # 尝试使用nvidia-smi（如果在PATH中）
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                shell=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
            
            # 备选方案：使用Windows Management Instrumentation (WMI)
            try:
                import wmi
                c = wmi.WMI()
                gpus = c.Win32_VideoController()
                for gpu in gpus:
                    if "NVIDIA" in gpu.Name:
                        return gpu.Name
            except ImportError:
                pass
    
        # 备选方案：检查CUDA库（需要PyTorch或TensorFlow）
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except ImportError:
            pass
            
        try:
            import tensorflow as tf
            if tf.test.is_gpu_available():
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    details = tf.config.experimental.get_device_details(gpus[0])
                    return details.get('device_name', 'NVIDIA GPU')
        except ImportError:
            pass
    
        # 最后手段：检查系统环境变量或驱动文件
        if platform.system() == "Linux":
            # 检查驱动文件
            if os.path.exists("/proc/driver/nvidia/version"):
                with open("/proc/driver/nvidia/version", "r") as f:
                    first_line = f.readline().strip()
                    match = re.search(r"NVIDIA driver \S+ for (\S+)", first_line)
                    if match:
                        return match.group(1)
    
    except Exception as e:
        print(f"检测显卡型号时出错: {e}")
    
    return "Unknown"


@lru_cache(maxsize=1)
def is_h800():
    gpu_model = get_gpu_model()
    return gpu_model.split('\n')[0].strip()=='NVIDIA H800'

@lru_cache(maxsize=1)
def gpu_flops():
    if is_h800():
        return 989e12
    else:
        return 312e12

def calculate_decoder_flops_v1(num_head, head_dim, hidden_size, intermediate_size, kv_heads=None, is_causal=False, seq_len=1, batch_size=1, linear_factor=2, attn_output_layers=2):
    """
    计算Transformer解码器层的FLOPs
    
    参数:
    num_head (int): 注意力头的数量
    head_dim (int): 每个注意力头的维度
    hidden_size (int): 隐藏层大小
    intermediate_size (int): FFN中间层大小
    kv_heads (int, optional): KV注意力头的数量，用于Group Attention，默认None表示不使用
    is_causal (bool): 是否使用因果掩码
    seq_len (int): 输入序列长度
    batch_size (int): 批处理大小，将序列分割为多个样本
    
    返回:
    dict: 包含各步骤FLOPs和总FLOPs的字典
    """
    # 默认KV头数量等于查询头数量
    if kv_heads is None:
        kv_heads = num_head

    # 计算每个样本的序列长度
    seq_len_per_sample = None if isinstance(seq_len, list) else seq_len // batch_size
    
    s_seq_len = s(seq_len)

    # 注意力计算FLOPs
    # QKV投影 (不受batch_size影响)
    q_flops = linear_factor * s_seq_len * hidden_size * (num_head * head_dim)
    k_flops = linear_factor * s_seq_len * hidden_size * (kv_heads * head_dim)
    v_flops = linear_factor * s_seq_len * hidden_size * (kv_heads * head_dim)

    # 注意力分数计算 [seq_len_per_sample, num_head, seq_len_per_sample, head_dim]
    if isinstance(seq_len, list):
        attn_scores_flops = 0
        for i, seq_len_per_sample in enumerate(seq_len):
            attn_scores_flops += linear_factor * num_head * seq_len_per_sample * seq_len_per_sample * head_dim
    else:
        attn_scores_flops = linear_factor * num_head * seq_len_per_sample * seq_len_per_sample * head_dim * batch_size


    # 因果掩码（如果启用）会减少一半的注意力计算
    if is_causal:
        # 因果掩码下三角区域计算量: n(n+1)/2 ≈ n²/2
        attn_scores_flops *= 0.5
    
    attn_v_flops = attn_scores_flops

    # 注意力输出投影 (不受batch_size影响)
    attn_out_flops = linear_factor * s_seq_len * (num_head * head_dim) * hidden_size * attn_output_layers
    
    # 注意力总FLOPs
    attention_flops = q_flops + k_flops + v_flops + attn_scores_flops + attn_v_flops + attn_out_flops
    
    # FFN层FLOPs (不受batch_size影响)
    ffn_1_flops = linear_factor * s_seq_len * hidden_size * intermediate_size
    ffn_2_flops = linear_factor * s_seq_len * intermediate_size * hidden_size
    ffn_flops = ffn_1_flops + ffn_2_flops

    # 总FLOPs
    total_flops = attention_flops + ffn_flops

    return {
        'total_flops': total_flops,
        'attention': {
            'q_proj': q_flops,
            'k_proj': k_flops,
            'v_proj': v_flops,
            'attn_scores': attn_scores_flops,
            'attn_v': attn_v_flops,
            'attn_out': attn_out_flops,
            'total': attention_flops
        },
        'ffn': {
            'fc1': ffn_1_flops,
            'fc2': ffn_2_flops,
            'total': ffn_flops
        },
        'batch_info': {
            'batch_size': batch_size,
            'seq_len_per_sample': seq_len_per_sample
        }
    }

import easydict




def calculate_decoder_layers_flops(num_head, head_dim, hidden_size, intermediate_size,
                                 kv_heads=None, is_causal=False, seq_len=1, num_layers=1,
                                 linear_factor=2, batch_size=1, attn_output_layers=2):
    """
    计算多层Transformer解码器的FLOPs
    
    参数:
    num_head (int): 注意力头的数量
    head_dim (int): 每个注意力头的维度
    hidden_size (int): 隐藏层大小
    intermediate_size (int): FFN中间层大小
    kv_heads (int, optional): KV注意力头的数量，用于Group Attention
    is_causal (bool): 是否使用因果掩码
    seq_len (int): 输入序列长度
    num_layers (int): 解码器层数
    backend (function): 使用的计算方法，默认为v1
    linear_factor (int): 线性计算因子，用于调整计算量
    
    返回:
    dict: 包含各层详细FLOPs和总FLOPs的字典
    """
    layers_flops = []
    total_flops = 0
    # 计算每一层的FLOPs
    for layer_idx in range(num_layers):
        layer_flops = calculate_decoder_flops_v1(
            num_head=num_head,
            head_dim=head_dim,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            kv_heads=kv_heads,
            is_causal=is_causal,
            seq_len=seq_len,
            linear_factor=linear_factor,
            batch_size=batch_size,
            attn_output_layers=attn_output_layers
        )
        layers_flops.append({
            'layer_index': layer_idx,
            **layer_flops
        })
        total_flops += layer_flops['total_flops']

    # 构建返回字典
    return {
        'total_flops': total_flops,
        'per_layer_flops': layers_flops[0],
        'avg_flops_per_layer': total_flops / num_layers if num_layers > 0 else 0,
        'num_layers': num_layers,
        'backend': str(calculate_decoder_flops_v1),
    }


def calculate_vlm_flops(vit_params, llm_params, linear_factor=2, _gpu_flops=None):
    """
    计算VLM(Vision-Language Model)的总计算量
    
    参数:
    vit_params (easydict): 包含ViT参数的对象，需要包含以下字段:
        num_head: ViT的注意力头数量
        head_dim: ViT每个注意力头的维度
        hidden_size: ViT的隐藏层大小
        intermediate_size: ViT的FFN中间层大小
        num_layers: ViT的层数
        kv_heads: ViT的KV注意力头数量(可选)
        seq_len: ViT的序列长度(通常为patch数量+1)
        batch_size: ViT的批处理大小
    
    llm_params (easydict): 包含LLM参数的对象，需要包含以下字段:
        num_head: LLM的注意力头数量
        head_dim: LLM每个注意力头的维度
        hidden_size: LLM的隐藏层大小
        intermediate_size: LLM的FFN中间层大小
        num_layers: LLM的层数
        kv_heads: LLM的KV注意力头数量(可选)
        is_causal: LLM是否使用因果注意力(默认为True)
        seq_len: LLM的序列长度
        batch_size: LLM的批处理大小
    
    通用参数:
        backend: 使用的计算方法(默认为v1)
        linear_factor: 线性计算因子
    
    返回:
    dict: 包含ViT、LLM详细计算量和总计算量的字典
    """
    # 计算ViT的计算量
    vit_flops = calculate_decoder_layers_flops(
        num_head=vit_params.num_head,
        head_dim=vit_params.head_dim,
        hidden_size=vit_params.hidden_size,
        intermediate_size=vit_params.intermediate_size,
        num_layers=vit_params.num_layers,
        kv_heads=vit_params.get('kv_heads', None),
        is_causal=False,  # ViT通常不使用因果注意力
        seq_len=vit_params.seq_len,
        batch_size=vit_params.get('batch_size', 1),
        linear_factor=linear_factor,
        attn_output_layers=2
    )

    vit2llm_flops = linear_factor * s(vit_params.seq_len) * (vit_params.hidden_size * llm_params.hidden_size + llm_params.hidden_size * llm_params.hidden_size)
    vit_flops['total_flops'] += vit2llm_flops
    vit_flops['vit2llm_flops'] = vit2llm_flops
    
    # 计算LLM的计算量
    llm_flops = calculate_decoder_layers_flops(
        num_head=llm_params.num_head,
        head_dim=llm_params.head_dim,
        hidden_size=llm_params.hidden_size,
        intermediate_size=llm_params.intermediate_size,
        num_layers=llm_params.num_layers,
        kv_heads=llm_params.get('kv_heads', None),
        is_causal=llm_params.get('is_causal', True),
        seq_len=llm_params.seq_len,
        batch_size=llm_params.get('batch_size', 1),
        linear_factor=linear_factor,
        attn_output_layers=3
    )
    
    lm_head_flops = linear_factor * s(llm_params.seq_len) * (llm_params.hidden_size * llm_params.vocab_size)
    llm_flops['total_flops'] += lm_head_flops
    llm_flops['lm_head_flops'] = lm_head_flops
    
    # 计算总FLOPs
    total_flops = vit_flops['total_flops'] + llm_flops['total_flops']
    _gpu_flops = gpu_flops() if _gpu_flops is None else _gpu_flops
    return {
        'total_flops': total_flops,
        'vit': vit_flops,
        'llm': llm_flops,
        'vit_percentage': vit_flops['total_flops'] / total_flops * 100 if total_flops > 0 else 0,
        'llm_percentage': llm_flops['total_flops'] / total_flops * 100 if total_flops > 0 else 0,
        'vit_total_flops*3(T)': vit_flops['total_flops'] * 3 / 1e12,
        'llm_total_flops*3(T)': llm_flops['total_flops'] * 3 / 1e12,
        'total_flops*3(T)': total_flops * 3 / 1e12,
        'total_flops/gpu_flops': total_flops * 3 / _gpu_flops,
        'gpu_flops': _gpu_flops
    }


def calculate_vit_flops(vit_params):
    linear_factor = 2
    vit_flops = calculate_decoder_layers_flops(
        num_head=vit_params.num_head,
        head_dim=vit_params.head_dim,
        hidden_size=vit_params.hidden_size,
        intermediate_size=vit_params.intermediate_size,
        num_layers=vit_params.num_layers,
        kv_heads=vit_params.get('kv_heads', None),
        is_causal=False,  # ViT通常不使用因果注意力
        seq_len=vit_params.seq_len,
        batch_size=vit_params.get('batch_size', 1),
        linear_factor=linear_factor,
        attn_output_layers=2
    )
    #vit2llm_flops = linear_factor * s(vit_params.seq_len) * (vit_params.hidden_size * llm_params.hidden_size + llm_params.hidden_size * llm_params.hidden_size)
    #vit_flops['total_flops'] += vit2llm_flops
    #vit_flops['vit2llm_flops'] = vit2llm_flops
    return vit_flops





def calculate_llm_flops(llm_params):
    linear_factor = 2
    # 计算LLM的计算量
    llm_flops = calculate_decoder_layers_flops(
        num_head=llm_params.num_head,
        head_dim=llm_params.head_dim,
        hidden_size=llm_params.hidden_size,
        intermediate_size=llm_params.intermediate_size,
        num_layers=llm_params.num_layers,
        kv_heads=llm_params.get('kv_heads', None),
        is_causal=llm_params.get('is_causal', True),
        seq_len=llm_params.seq_len,
        batch_size=llm_params.get('batch_size', 1),
        linear_factor=linear_factor,
        attn_output_layers=3
    )
    
    lm_head_flops = linear_factor * s(llm_params.seq_len) * (llm_params.hidden_size * llm_params.vocab_size)
    llm_flops['total_flops'] += lm_head_flops
    llm_flops['lm_head_flops'] = lm_head_flops
    return llm_flops


def calculate_llm_flops_from_config(config_path, seq_len, batch_size):
    llm_params = extract_model_params(config_path)[0]
    llm_params.seq_len = seq_len
    llm_params.batch_size = batch_size
    return calculate_llm_flops(llm_params)


def calculate_vit_flops_from_config(config_path, seq_len, batch_size):
    vit_params = extract_model_params(config_path)[1]
    vit_params.seq_len = seq_len
    vit_params.batch_size = batch_size
    return calculate_vit_flops(vit_params)


@lru_cache(maxsize=32)
def extract_model_params(config_path):
    """
    从模型配置JSON文件中提取Transformer和Vision模块的参数
    支持Qwen3和InternVL架构
    
    参数:
    config_path (str): JSON配置文件路径
    
    返回:
    tuple: 包含两个字典的元组
        - transformer_params: Transformer模块参数
        - vision_params: Vision模块参数
    """
    import json

    # 读取JSON配置文件
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 判断模型架构类型
    if 'architectures' in config and 'InternVLChatModel' in config['architectures']:
        print(f"flops_counter: InternVLChatModel architectures")
        # InternVL架构处理逻辑
        llm_config = config['llm_config']

        # 提取LLM参数
        transformer_params = {
            'num_head': llm_config['num_attention_heads'],
            'head_dim': llm_config['hidden_size'] // llm_config['num_attention_heads'],
            'hidden_size': llm_config['hidden_size'],
            'intermediate_size': llm_config['intermediate_size'],
            'kv_heads': llm_config.get('num_key_value_heads', llm_config['num_attention_heads']),
            'num_layers': llm_config['num_hidden_layers'],
            'vocab_size': llm_config['vocab_size']
        }
        
        vision_config = config['vision_config']
        # 提取Vision参数 
        vision_params = {
            'num_head': vision_config['num_attention_heads'],
            'head_dim': vision_config['hidden_size'] / vision_config['num_attention_heads'],
            'hidden_size': vision_config['hidden_size'],
            'intermediate_size': vision_config['intermediate_size'],
            'num_layers': vision_config['num_hidden_layers'],
        }
        vision_params = {k: v for k, v in vision_params.items() if v is not None}
    elif 'architectures' in config and 'Qwen2_5_VLForConditionalGeneration' in config['architectures']:
        print(f"flops_counter: Qwen2_5_VLForConditionalGeneration architectures")

        # Qwen2.5 VL架构处理逻辑
        transformer_params = {
            'num_head': config['num_attention_heads'],
            'head_dim': config['hidden_size'] / config['num_attention_heads'],
            'hidden_size': config['hidden_size'],
            'intermediate_size': config['intermediate_size'],
            'kv_heads': config['num_key_value_heads'],
            'num_layers': config['num_hidden_layers'],
            'vocab_size': config['vocab_size']
        }
        vision_config = config['vision_config']
        vision_params = {
            'num_head': vision_config['num_heads'],
            'head_dim': vision_config['hidden_size'] / vision_config['num_heads'],
            'hidden_size': vision_config['hidden_size'],
            'intermediate_size': vision_config['intermediate_size'],
            'num_layers': vision_config['depth'],
            
        }
    elif 'architectures' in config and 'KeyeForConditionalGeneration' in config['architectures']:
        print(f"flops_counter: KeyeForConditionalGeneration architectures")
        transformer_params = {
            'num_head': config['num_attention_heads'],
            'head_dim': config['hidden_size'] / config['num_attention_heads'],
            'hidden_size': config['hidden_size'],
            'intermediate_size': config['intermediate_size'],
            'kv_heads': config['num_key_value_heads'],
            'num_layers': config['num_hidden_layers'],
            'vocab_size': config['vocab_size']
        }
        vision_config = config['vision_config']
        vision_params = {
            'num_head': vision_config['num_attention_heads'],
            'head_dim': vision_config['hidden_size'] / vision_config['num_attention_heads'],
            'hidden_size': vision_config['hidden_size'],
            'intermediate_size': vision_config['intermediate_size'],
            'num_layers': vision_config['num_hidden_layers'],
            
        }
    else:
        print(f"flops_counter: Qwen3siglip architectures")

        # Qwen3siglip架构处理逻辑 (保持原有逻辑)
        transformer_params = {
            'num_head': config['num_attention_heads'],
            'head_dim': config['head_dim'],
            'hidden_size': config['hidden_size'],
            'intermediate_size': config['intermediate_size'],
            'kv_heads': config['num_key_value_heads'],
            'num_layers': config['num_hidden_layers'],
            'vocab_size': config['vocab_size']
        }
        with open('/llm_reco_ssd/zhouyang12/models/siglip2-so400m-patch16-naflex/config.json', 'r') as f: vision_config = json.load(f)['vision_config']
        vision_config = config['vision_config']
        vision_params = {
            'num_head':vision_config['num_heads'],
            'head_dim': vision_config['hidden_size'] / vision_config['num_heads'],
            'hidden_size': vision_config['hidden_size'],
            'intermediate_size': vision_config['intermediate_size'],
            'num_layers': vision_config['depth'],
        }
        vision_params = {k: v for k, v in vision_params.items() if v is not None}
    
    return easydict.EasyDict(transformer_params), easydict.EasyDict(vision_params)
    


def calc_mfu(config_path, total_seq_len, image_token_merged_len, llm_batch_size, image_batch_size=None, secs_per_step=None, _gpu_flops=None):
    if image_batch_size is None: image_batch_size = llm_batch_size
    transformer_params, vision_params = extract_model_params(
        config_path
    )
    llm_params = easydict.EasyDict({
        **transformer_params,
        'is_causal': False,
        'seq_len': total_seq_len, 
        'batch_size': llm_batch_size
    })

    vit_params = easydict.EasyDict({
        **vision_params,
        'is_causal': False,
        'seq_len': [x*4 for x in image_token_merged_len] if isinstance(image_token_merged_len, list) else image_token_merged_len * 4, 
        'batch_size': image_batch_size
    })

    flops = calculate_vlm_flops(vit_params, llm_params, _gpu_flops=_gpu_flops)

    flops['input_args'] = easydict.EasyDict(
        config_path=config_path,
        total_seq_len=total_seq_len,
        image_token_merged_len=len(image_token_merged_len) if isinstance(image_token_merged_len, list) else image_token_merged_len,
        llm_batch_size=llm_batch_size,
        image_batch_size=image_batch_size,
        secs_per_step=secs_per_step
    )
    if secs_per_step is not None:
        flops['mfu'] = flops['total_flops/gpu_flops'] / secs_per_step
    return flops


class MFUStats:
  def __init__(self, args):
      self.tokens_for_mfu = collections.defaultdict(int)
      self.mfu_per_step_per_gpu = None
      self.args = args
      self.total_mfu = defaultdict(int)

  def set(self, num_image_tokens, num_tokens, num_samples, num_images):
      num_image_tokens = int(num_image_tokens)
      num_tokens = int(num_tokens)
      num_samples = int(num_samples)
      num_images = int(num_images)
      self.tokens_for_mfu["num_image_tokens"] += num_image_tokens
      self.tokens_for_mfu["num_tokens"] += num_tokens
      self.tokens_for_mfu["num_samples"] += num_samples
      self.tokens_for_mfu["num_images"] += num_images

  def mfu(self, secs, global_step):
      import easydict
      args = self.args
      tokens_for_mfu = self.tokens_for_mfu
      mfu_args = easydict.EasyDict(
        # 暂时认为各条样本长度均匀
        total_seq_len=round(tokens_for_mfu["num_tokens"] / args.logging_per_step), 
        image_token_merged_len=[round(tokens_for_mfu["num_image_tokens"]  / tokens_for_mfu["num_images"])] * round(tokens_for_mfu["num_images"] / args.logging_per_step)  if tokens_for_mfu["num_images"] != 0 else 1, 
        llm_batch_size=round(tokens_for_mfu["num_samples"] / args.logging_per_step), 
        secs_per_step=secs / args.logging_per_step
      )
      mfu_per_step_per_gpu = calc_mfu(os.path.join(args.model_dir, "config.json"), **mfu_args)
      self.mfu_per_step_per_gpu = mfu_per_step_per_gpu
      total_mfu = self.total_mfu
      total_mfu['llm_total_flops*3(T)'] += mfu_per_step_per_gpu['llm_total_flops*3(T)'] * args.logging_per_step
      total_mfu['vit_total_flops*3(T)'] += mfu_per_step_per_gpu['vit_total_flops*3(T)'] * args.logging_per_step
      total_mfu['mfu'] += mfu_per_step_per_gpu['mfu'] * args.logging_per_step
      mfu_log_dict = {
        "perf/mfu_per_step_per_gpu": mfu_per_step_per_gpu['mfu'],
        "perf/vit_flops_per_step_per_gpu": mfu_per_step_per_gpu['vit_total_flops*3(T)'],
        "perf/llm_flops_per_step_per_gpu": mfu_per_step_per_gpu['llm_total_flops*3(T)'],
        "perf/mfu_per_step_per_gpu_v2": total_mfu['mfu'] / global_step,
        "perf/vit_flops_per_step_per_gpu_v2": total_mfu['vit_total_flops*3(T)'] / global_step,
        "perf/llm_flops_per_step_per_gpu_v2": total_mfu['llm_total_flops*3(T)'] / global_step,
        "perf/num_images_per_step": tokens_for_mfu["num_images"] / args.logging_per_step,
      }
      self.tokens_for_mfu = collections.defaultdict(int)
      return mfu_log_dict

