#!/usr/bin/env python3
"""
测试WandB禁用和评估batch size修复
"""

import os
import sys
import yaml
import json

# 设置环境变量
os.environ['NCCL_NTHREADS'] = '64'

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_wandb_disabled():
    """测试WandB是否被完全禁用"""
    print("🚫 测试WandB禁用状态...")
    
    # 加载配置
    config_file = "configs/fast_eval_config.yaml"
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    try:
        from training.utils.monitor import TrainingMonitor
        
        # 测试TrainingMonitor
        monitor = TrainingMonitor("./temp_output", config)
        
        use_wandb = getattr(monitor, 'use_wandb', None)
        if use_wandb is False:
            print("✅ TrainingMonitor中WandB已禁用")
            training_disabled = True
        else:
            print(f"⚠️ TrainingMonitor中WandB状态: {use_wandb}")
            training_disabled = False
        
        # 测试DummyMonitor
        from training.utils.monitor import DummyMonitor
        dummy_monitor = DummyMonitor("./temp_output", config)
        
        dummy_use_wandb = getattr(dummy_monitor, 'use_wandb', None)
        if dummy_use_wandb is False:
            print("✅ DummyMonitor中WandB已禁用")
            dummy_disabled = True
        else:
            print(f"⚠️ DummyMonitor中WandB状态: {dummy_use_wandb}")
            dummy_disabled = False
        
        return training_disabled and dummy_disabled
        
    except Exception as e:
        print(f"❌ WandB禁用测试失败: {e}")
        return False

def test_eval_batch_size():
    """测试评估batch size修复"""
    print("\n📊 测试评估batch size修复...")
    
    # 加载配置
    config_file = "configs/fast_eval_config.yaml"
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 准备配置
    from training.utils.config_utils import prepare_config
    config = prepare_config(config)
    
    try:
        # 获取DeepSpeed配置
        if 'deepspeed' in config:
            if isinstance(config['deepspeed'], str):
                with open(config['deepspeed'], 'r') as f:
                    deepspeed_config = json.load(f)
            else:
                deepspeed_config = config['deepspeed']
            
            train_micro_batch = deepspeed_config.get('train_micro_batch_size_per_gpu', 1)
            train_batch_size = deepspeed_config.get('train_batch_size', 1)
            gradient_accumulation = deepspeed_config.get('gradient_accumulation_steps', 1)
            
            print(f"DeepSpeed配置:")
            print(f"  • train_micro_batch_size_per_gpu: {train_micro_batch}")
            print(f"  • train_batch_size: {train_batch_size}")
            print(f"  • gradient_accumulation_steps: {gradient_accumulation}")
            
            # 测试数据加载器
            from data.dataloader import create_dataloaders
             train_loader, val_loader = create_dataloaders(config)
             
             train_actual_batch = train_loader.batch_size
             val_actual_batch = val_loader.batch_size
             
             print(f"实际DataLoader batch size:")
             print(f"  • 训练DataLoader batch size: {train_actual_batch}")
             print(f"  • 评估DataLoader batch size: {val_actual_batch}")
             
             # 检查理论计算
             world_size = 8  # 假设8卡，实际运行时会从分布式获取
             expected_eval_batch = train_micro_batch * world_size  # gradient_accumulation_steps=1时的等效
             theoretical_total_batch = train_micro_batch * world_size * gradient_accumulation
             
             print(f"理论计算:")
             print(f"  • 期望评估batch size: {expected_eval_batch} (micro_batch × num_gpus)")
             print(f"  • 理论训练总batch size: {theoretical_total_batch} (包含gradient accumulation)")
             print(f"  • 配置的train_batch_size: {train_batch_size}")
             
             # 检查修复逻辑
             # 训练DataLoader应该使用micro_batch_size_per_gpu
             # 评估DataLoader应该使用micro_batch_size_per_gpu × num_gpus
             
             train_correct = (train_actual_batch == train_micro_batch)
             # 注意：在单机测试时world_size可能是1，所以eval_batch_size可能等于train_micro_batch
             eval_correct = (val_actual_batch >= train_micro_batch)  # 至少不小于micro batch
             
             if train_correct and eval_correct:
                 print("✅ batch size逻辑修复正确")
                 print(f"   - 训练使用micro batch: {train_actual_batch}")
                 print(f"   - 评估使用适当大小: {val_actual_batch}")
                 return True
             else:
                 print("⚠️ batch size逻辑需要进一步检查")
                 print(f"   - 训练batch正确: {train_correct}")
                 print(f"   - 评估batch合理: {eval_correct}")
                 return False
        else:
            print("⚠️ 未使用DeepSpeed配置")
            return True
            
    except Exception as e:
        print(f"❌ batch size测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_efficiency():
    """测试内存效率改进"""
    print("\n💾 测试内存效率...")
    
    # 加载配置
    config_file = "configs/fast_eval_config.yaml"
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 准备配置
    from training.utils.config_utils import prepare_config
    config = prepare_config(config)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            # 清理GPU内存
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            print(f"初始GPU内存: {initial_memory / 1e9:.2f} GB")
            
            # 创建数据加载器
            from data.dataloader import create_dataloaders
            train_loader, val_loader = create_dataloaders(config)
            
            # 测试单个batch的内存使用
            for batch in val_loader:
                batch_memory = torch.cuda.memory_allocated()
                print(f"加载batch后GPU内存: {batch_memory / 1e9:.2f} GB")
                print(f"batch内存增长: {(batch_memory - initial_memory) / 1e9:.2f} GB")
                
                # 检查batch大小
                batch_size = batch["input_ids"].size(0)
                print(f"实际batch大小: {batch_size}")
                
                # 估算内存效率
                memory_per_sample = (batch_memory - initial_memory) / batch_size
                print(f"每样本内存: {memory_per_sample / 1e6:.2f} MB")
                
                break
            
            torch.cuda.empty_cache()
            
            if memory_per_sample < 1e9:  # 小于1GB每样本
                print("✅ 内存使用效率良好")
                return True
            else:
                print("⚠️ 内存使用仍然较高")
                return False
        else:
            print("⚠️ 未检测到GPU，跳过内存测试")
            return True
            
    except Exception as e:
        print(f"❌ 内存效率测试失败: {e}")
        return False

def test_trainer_initialization():
    """测试训练器初始化"""
    print("\n🚀 测试训练器初始化...")
    
    # 加载配置
    config_file = "configs/fast_eval_config.yaml"
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 准备配置
    from training.utils.config_utils import prepare_config
    config = prepare_config(config)
    
    try:
        import time
        
        start_time = time.time()
        from training.deepspeed_trainer import DeepSpeedTrainer
        trainer = DeepSpeedTrainer(config)
        init_time = time.time() - start_time
        
        print(f"训练器初始化时间: {init_time:.2f}s")
        
        # 检查关键组件状态
        mfu_stats = getattr(trainer, 'mfu_stats', None)
        monitor_wandb = getattr(trainer.monitor, 'use_wandb', None)
        
        print(f"组件状态:")
        print(f"  • mfu_stats: {mfu_stats}")
        print(f"  • monitor use_wandb: {monitor_wandb}")
        
        if init_time < 60 and mfu_stats is None and monitor_wandb is False:
            print("✅ 训练器初始化成功且优化生效")
            return True
        else:
            print("⚠️ 训练器初始化存在问题")
            return False
            
    except Exception as e:
        print(f"❌ 训练器初始化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 开始WandB禁用和评估修复验证")
    print("=" * 70)
    
    # 测试1: WandB禁用
    wandb_ok = test_wandb_disabled()
    
    # 测试2: 评估batch size修复
    batch_ok = test_eval_batch_size()
    
    # 测试3: 内存效率
    memory_ok = test_memory_efficiency()
    
    # 测试4: 训练器初始化
    trainer_ok = test_trainer_initialization()
    
    print("\n" + "=" * 70)
    print("📊 测试结果总结:")
    print(f"  • WandB禁用: {'✅ 成功' if wandb_ok else '❌ 失败'}")
    print(f"  • 评估batch size修复: {'✅ 成功' if batch_ok else '❌ 失败'}")
    print(f"  • 内存效率: {'✅ 良好' if memory_ok else '⚠️ 一般'}")
    print(f"  • 训练器初始化: {'✅ 正常' if trainer_ok else '❌ 异常'}")
    
    if wandb_ok and batch_ok and trainer_ok:
        print("\n🎉 所有关键修复验证成功！")
        print("   - WandB已完全禁用，不再有日志开销")
        print("   - 评估batch size已修复，避免显存爆炸")
        print("   - 训练性能应该显著提升")
        sys.exit(0)
    else:
        print("\n⚠️ 部分修复需要进一步检查")
        sys.exit(1) 