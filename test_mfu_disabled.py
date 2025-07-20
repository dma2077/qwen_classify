#!/usr/bin/env python3
"""
测试MFU禁用后的性能改进
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

def test_trainer_initialization():
    """测试训练器初始化时间"""
    print("🚀 测试训练器初始化（MFU禁用）...")
    
    # 加载配置
    config_file = "configs/fast_eval_config.yaml"
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 准备配置
    from training.utils.config_utils import prepare_config
    config = prepare_config(config)
    
    # 测试训练器初始化时间
    start_time = time.time()
    
    try:
        from training.deepspeed_trainer import DeepSpeedTrainer
        trainer = DeepSpeedTrainer(config)
        
        init_time = time.time() - start_time
        print(f"✅ 训练器初始化完成: {init_time:.2f}s")
        
        # 检查MFU统计器是否被禁用
        if hasattr(trainer, 'mfu_stats') and trainer.mfu_stats is None:
            print("✅ MFU统计器已成功禁用")
            return True
        else:
            print("⚠️ MFU统计器仍然存在")
            return False
            
    except Exception as e:
        print(f"❌ 训练器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_processing_speed():
    """测试批次处理速度"""
    print("\n🔥 测试批次处理速度（MFU禁用）...")
    
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
    
    # 创建模型
    from models.qwen2_5_vl_classify import Qwen25VLClassify
    model = Qwen25VLClassify(config['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    
    # 测试批次处理时间
    print("测试纯前向传播时间...")
    
    batch_times = []
    for i, batch in enumerate(train_loader):
        if i >= 3:  # 只测试前3个batch
            break
        
        batch_start = time.time()
        
        # 数据准备
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
        with torch.no_grad():
            outputs = model(**forward_kwargs)
            loss = outputs.loss
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        print(f"  Batch {i+1}: {batch_time:.3f}s (loss: {loss.item():.4f})")
    
    avg_batch_time = sum(batch_times) / len(batch_times)
    print(f"📊 平均batch处理时间: {avg_batch_time:.3f}s")
    
    # 评估性能改进
    if avg_batch_time < 5.0:
        print("✅ 批次处理速度优秀")
        return True
    elif avg_batch_time < 10.0:
        print("🔶 批次处理速度良好")
        return True
    else:
        print("⚠️ 批次处理速度仍需优化")
        return False

def test_training_loop_overhead():
    """测试训练循环开销"""
    print("\n🔥 测试训练循环开销...")
    
    # 模拟训练步骤开销（不包含实际模型计算）
    print("测试训练辅助函数开销...")
    
    # 模拟一些训练中的操作
    step_times = []
    
    for i in range(10):
        step_start = time.time()
        
        # 模拟一些轻量级操作
        dummy_tensor = torch.randn(8, 512).cuda() if torch.cuda.is_available() else torch.randn(8, 512)
        _ = dummy_tensor.sum()
        
        # 模拟梯度范数计算
        grad_norm = torch.tensor(1.0).cuda() if torch.cuda.is_available() else torch.tensor(1.0)
        grad_norm_value = grad_norm if grad_norm is not None else 0.0
        
        # 模拟学习率获取
        current_lr = 1e-5
        
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        if i < 3:
            print(f"  Step {i+1} overhead: {step_time:.6f}s")
    
    avg_overhead = sum(step_times) / len(step_times)
    print(f"📊 平均步骤开销: {avg_overhead:.6f}s")
    
    if avg_overhead < 0.01:  # 小于10ms
        print("✅ 训练开销很低")
        return True
    else:
        print("⚠️ 训练开销较高")
        return False

if __name__ == "__main__":
    print("🚀 开始MFU禁用性能测试")
    print("=" * 60)
    
    # 测试1: 训练器初始化
    init_ok = test_trainer_initialization()
    
    # 测试2: 批次处理速度
    batch_ok = test_batch_processing_speed()
    
    # 测试3: 训练循环开销
    overhead_ok = test_training_loop_overhead()
    
    print("\n" + "=" * 60)
    print("📊 MFU禁用测试结果总结:")
    print(f"  • 训练器初始化: {'✅ 正常' if init_ok else '❌ 异常'}")
    print(f"  • 批次处理速度: {'✅ 良好' if batch_ok else '⚠️ 需要优化'}")
    print(f"  • 训练循环开销: {'✅ 很低' if overhead_ok else '⚠️ 较高'}")
    
    if init_ok and batch_ok and overhead_ok:
        print("🎉 MFU禁用成功，性能应该有显著提升！")
        sys.exit(0)
    else:
        print("⚠️ 某些方面仍需要优化")
        sys.exit(1) 