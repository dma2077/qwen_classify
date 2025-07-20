#!/usr/bin/env python3
"""
数据加载器调试脚本 - 用于排查训练卡住问题
"""

import os
import sys
import yaml
import time
import torch

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def debug_dataloader():
    """调试数据加载器"""
    print("🔍 开始调试数据加载器...")
    
    # 设置环境变量
    os.environ['NCCL_NTHREADS'] = '64'
    
    # 加载配置
    config_file = "configs/fast_eval_config.yaml"  # 使用快速配置
    print(f"📋 加载配置文件: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 准备配置
    from training.utils.config_utils import prepare_config
    config = prepare_config(config)
    
    print("✅ 配置加载完成")
    
    # 创建数据加载器
    print("📦 创建数据加载器...")
    from data.dataloader import create_dataloaders
    
    start_time = time.time()
    train_loader, val_loader = create_dataloaders(config)
    dataloader_time = time.time() - start_time
    
    print(f"✅ 数据加载器创建完成 (耗时: {dataloader_time:.2f}s)")
    print(f"📊 训练数据集大小: {len(train_loader.dataset)}")
    print(f"📊 验证数据集大小: {len(val_loader.dataset)}")
    print(f"📊 训练批次数: {len(train_loader)}")
    
    # 测试获取第一个batch
    print("🔥 测试获取第一个训练batch...")
    start_time = time.time()
    
    try:
        # 设置超时保护
        import signal
        
        def timeout_handler(signum, frame):
            print("⚠️ 获取batch超时！")
            raise TimeoutError("获取batch超时")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30秒超时
        
        first_batch = next(iter(train_loader))
        signal.alarm(0)  # 取消超时
        
        batch_time = time.time() - start_time
        print(f"✅ 成功获取第一个batch (耗时: {batch_time:.2f}s)")
        
        # 检查batch内容
        print("📋 Batch内容:")
        for key, value in first_batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  • {key}: {value.shape} ({value.dtype})")
            else:
                print(f"  • {key}: {type(value)} (长度: {len(value) if hasattr(value, '__len__') else 'N/A'})")
                
    except TimeoutError:
        print("❌ 获取第一个batch超时")
        return False
    except Exception as e:
        print(f"❌ 获取第一个batch失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试获取多个batch
    print("🔥 测试获取前3个batch...")
    start_time = time.time()
    
    try:
        for i, batch in enumerate(train_loader):
            if i >= 3:
                break
            print(f"  ✅ 成功获取batch {i+1}")
            
        multi_batch_time = time.time() - start_time
        print(f"✅ 成功获取前3个batch (耗时: {multi_batch_time:.2f}s)")
        
    except Exception as e:
        print(f"❌ 获取多个batch失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("🎉 数据加载器测试完成，没有发现问题！")
    return True

if __name__ == "__main__":
    debug_dataloader() 