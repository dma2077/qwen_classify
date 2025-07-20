#!/usr/bin/env python3
"""
训练速度测试脚本
"""

import os
import sys
import time
import yaml
import torch

# 设置环境变量
os.environ['NCCL_NTHREADS'] = '64'

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_training_speed():
    """测试训练速度"""
    print("🚀 测试训练速度...")
    
    # 加载配置
    config_file = "configs/fast_eval_config.yaml"
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 减少训练步数以快速测试
    config['train']['num_epochs'] = 1
    config['eval']['eval_steps'] = 100  # 增加评估间隔
    config['train']['save_steps'] = 1000  # 增加保存间隔
    config['train']['logging_steps'] = 50  # 增加日志间隔
    
    print(f"🔧 测试配置: eval_steps={config['eval']['eval_steps']}, save_steps={config['train']['save_steps']}")
    
    # 准备配置
    from training.utils.config_utils import prepare_config
    config = prepare_config(config)
    
    # 创建数据加载器
    from data.dataloader import create_dataloaders
    
    start_time = time.time()
    train_loader, val_loader = create_dataloaders(config)
    loader_time = time.time() - start_time
    
    print(f"✅ 数据加载器创建: {loader_time:.2f}s")
    print(f"📊 训练数据集大小: {len(train_loader.dataset)}")
    print(f"📊 训练批次数: {len(train_loader)}")
    
    # 测试单个batch的处理时间
    print("🔥 测试单个batch处理时间...")
    
    # 创建模型
    from models.qwen2_5_vl_classify import Qwen25VLClassify
    model = Qwen25VLClassify(config['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 测试前向传播时间
    model.train()
    batch_times = []
    
    for i, batch in enumerate(train_loader):
        if i >= 5:  # 只测试前5个batch
            break
        
        batch_start = time.time()
        
        # 模拟trainer的数据准备
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
        
        if "image_grid_thw" in batch:
            forward_kwargs["image_grid_thw"] = batch["image_grid_thw"].to(device, non_blocking=True)
        
        # 前向传播
        with torch.no_grad():  # 只测试前向传播时间
            outputs = model(**forward_kwargs)
            loss = outputs.loss
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        print(f"  Batch {i+1}: {batch_time:.3f}s (loss: {loss.item():.4f})")
    
    avg_batch_time = sum(batch_times) / len(batch_times)
    print(f"📊 平均batch前向传播时间: {avg_batch_time:.3f}s")
    
    # 预估训练时间
    estimated_step_time = avg_batch_time * 1.5  # 考虑反向传播等额外开销
    print(f"📊 预估完整训练步骤时间: {estimated_step_time:.1f}s")
    
    if estimated_step_time > 20:
        print("❌ 训练速度仍然很慢，需要进一步优化")
        return False
    elif estimated_step_time > 10:
        print("⚠️ 训练速度一般，建议继续优化")
        return True
    else:
        print("✅ 训练速度良好")
        return True

def test_data_loading_speed():
    """测试数据加载速度"""
    print("\n🔥 测试数据加载速度...")
    
    # 加载配置
    config_file = "configs/fast_eval_config.yaml"
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 准备配置
    from training.utils.config_utils import prepare_config
    config = prepare_config(config)
    
    # 创建数据加载器
    from data.dataloader import create_dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # 测试数据加载速度
    print("测试数据迭代速度...")
    
    data_times = []
    total_start = time.time()
    
    for i, batch in enumerate(train_loader):
        if i >= 10:  # 只测试前10个batch
            break
        
        data_start = time.time()
        # 简单访问数据
        _ = batch["input_ids"].shape
        _ = batch["pixel_values"].shape
        _ = batch["labels"].shape
        data_time = time.time() - data_start
        data_times.append(data_time)
        
        if i < 5:
            print(f"  Data batch {i+1}: {data_time:.3f}s")
    
    total_time = time.time() - total_start
    avg_data_time = sum(data_times) / len(data_times)
    
    print(f"📊 平均数据加载时间: {avg_data_time:.3f}s")
    print(f"📊 总时间 (10 batch): {total_time:.1f}s")
    
    if avg_data_time > 1.0:
        print("⚠️ 数据加载可能存在瓶颈")
        return False
    else:
        print("✅ 数据加载速度正常")
        return True

if __name__ == "__main__":
    print("🚀 开始训练速度测试")
    print("=" * 50)
    
    # 测试1: 数据加载速度
    data_ok = test_data_loading_speed()
    
    # 测试2: 训练速度
    training_ok = test_training_speed()
    
    print("\n" + "=" * 50)
    print("📊 速度测试结果总结:")
    print(f"  • 数据加载速度: {'✅ 正常' if data_ok else '⚠️ 需要优化'}")
    print(f"  • 训练速度: {'✅ 良好' if training_ok else '❌ 需要优化'}")
    
    if training_ok and data_ok:
        print("🎉 训练速度测试通过！")
    else:
        print("❌ 训练速度需要进一步优化") 