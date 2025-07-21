#!/usr/bin/env python3
"""
测试DataLoader是否会卡死
"""

import os
import sys
import yaml
import deepspeed

# 设置环境变量
os.environ['NCCL_NTHREADS'] = '64'
os.environ['MASTER_PORT'] = '29501'
os.environ['MASTER_ADDR'] = 'localhost'

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_dataloader():
    """只测试DataLoader，不涉及模型"""
    
    print("🔍 开始DataLoader测试...")
    
    # 初始化分布式
    deepspeed.init_distributed()
    print("✅ 分布式初始化完成")
    
    # 加载配置 - 你需要提供你的配置文件路径
    config_path = "configs/foodx251_cosine_5e_6_ls.yaml"  # 替换为你的配置文件
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        print("请修改config_path为你的实际配置文件路径")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置DeepSpeed配置
    deepspeed_config_path = "configs/ds_s2_as_8.json"  # 替换为你的DeepSpeed配置
    if not os.path.exists(deepspeed_config_path):
        print(f"❌ DeepSpeed配置文件不存在: {deepspeed_config_path}")
        print("请修改deepspeed_config_path为你的实际配置文件路径")
        return
    
    config['deepspeed'] = deepspeed_config_path
    
    # 准备配置
    from training.utils.config_utils import prepare_config
    config = prepare_config(config)
    
    print("✅ 配置加载完成")
    
    # 创建DataLoader
    from data.dataloader import create_dataloaders
    print("🔍 开始创建DataLoader...")
    
    try:
        train_loader, val_loader = create_dataloaders(config)
        print("✅ DataLoader创建成功")
        print(f"📊 训练集大小: {len(train_loader.dataset)}")
        print(f"📊 验证集大小: {len(val_loader.dataset)}")
        print(f"📊 训练batch数量: {len(train_loader)}")
        print(f"📊 验证batch数量: {len(val_loader)}")
    except Exception as e:
        print(f"❌ DataLoader创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试获取第一个batch
    print("🔍 尝试获取第一个batch...")
    
    try:
        # 测试训练DataLoader
        print("🔍 测试训练DataLoader...")
        train_iter = iter(train_loader)
        first_batch = next(train_iter)
        print("✅ 成功获取训练batch")
        print(f"📊 Batch keys: {list(first_batch.keys())}")
        print(f"📊 Input shape: {first_batch['input_ids'].shape}")
        print(f"📊 Pixel values shape: {first_batch['pixel_values'].shape}")
        
        # 测试获取第二个batch
        print("🔍 尝试获取第二个batch...")
        second_batch = next(train_iter)
        print("✅ 成功获取第二个batch")
        
        # 测试获取第三个batch
        print("🔍 尝试获取第三个batch...")
        third_batch = next(train_iter)
        print("✅ 成功获取第三个batch")
        
    except Exception as e:
        print(f"❌ 获取batch失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("🎉 DataLoader测试完成，没有发现卡死问题")

if __name__ == "__main__":
    test_dataloader() 