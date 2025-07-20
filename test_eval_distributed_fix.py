#!/usr/bin/env python3
"""
测试分布式评估修复的脚本
"""

import os
import sys
import yaml
import torch

# 设置环境变量
os.environ['NCCL_NTHREADS'] = '64'
os.environ['NCCL_TIMEOUT'] = '3600'  # 1小时超时
os.environ['NCCL_SOCKET_TIMEOUT'] = '3600'  # socket超时

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_evaluation_fix():
    """测试评估修复"""
    print("🚀 测试分布式评估修复...")
    
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
    
    print(f"✅ 数据加载器创建成功")
    print(f"📊 验证数据集大小: {len(val_loader.dataset)}")
    print(f"📊 验证批次数: {len(val_loader)}")
    
    # 创建分布式上下文
    from training.utils.distributed import DistributedContext
    dist_ctx = DistributedContext()
    
    print(f"🔧 分布式状态: 是否分布式={dist_ctx.world_size > 1}, rank={dist_ctx.rank}")
    
    # 创建模型
    from models.qwen2_5_vl_classify import Qwen25VLClassify
    
    print("🏗️ 创建模型...")
    model = Qwen25VLClassify(config['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"🔧 模型设备: {device}")
    print(f"🔧 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试修复的评估函数
    try:
        print("🔥 开始测试修复的评估函数...")
        
        from training.utils.evaluation import evaluate_model
        
        # 进行评估
        eval_loss, eval_accuracy = evaluate_model(model, val_loader, device)
        
        # 检查结果
        if eval_loss >= 999.0:
            print("❌ 评估返回了错误标识，表示评估失败")
            return False
        else:
            print(f"✅ 评估成功完成!")
            print(f"📊 评估损失: {eval_loss:.4f}")
            print(f"📊 评估准确率: {eval_accuracy:.4f}")
            return True
        
    except Exception as e:
        print(f"❌ 评估测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_small_data():
    """使用少量数据测试评估"""
    print("\n🔥 使用少量数据测试评估...")
    
    # 加载配置
    config_file = "configs/ultra_fast_eval_config.yaml"
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 修改为使用更少的数据
    config['data']['max_samples'] = 50  # 只使用50个样本
    config['train']['batch_size'] = 4   # 小批次
    config['eval']['batch_size'] = 4    # 小批次
    
    print(f"🔧 使用小数据集测试: max_samples={config['data']['max_samples']}")
    
    # 准备配置
    from training.utils.config_utils import prepare_config
    config = prepare_config(config)
    
    # 创建数据加载器
    from data.dataloader import create_dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    print(f"✅ 小数据集加载器创建成功")
    print(f"📊 验证数据集大小: {len(val_loader.dataset)}")
    print(f"📊 验证批次数: {len(val_loader)}")
    
    # 创建简单模型进行快速测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        from training.utils.evaluation import evaluate_single_dataset_fast
        
        # 创建模型
        from models.qwen2_5_vl_classify import Qwen25VLClassify
        model = Qwen25VLClassify(config['model'])
        model = model.to(device)
        
        # 进行评估
        eval_loss, eval_accuracy = evaluate_single_dataset_fast(model, val_loader, device)
        
        if eval_loss >= 999.0:
            print("❌ 小数据集评估失败")
            return False
        else:
            print(f"✅ 小数据集评估成功!")
            print(f"📊 评估损失: {eval_loss:.4f}")
            print(f"📊 评估准确率: {eval_accuracy:.4f}")
            return True
        
    except Exception as e:
        print(f"❌ 小数据集评估测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 开始分布式评估修复测试")
    print("=" * 60)
    
    # 测试1: 正常评估测试
    success1 = test_evaluation_fix()
    
    # 测试2: 小数据集测试
    success2 = test_with_small_data()
    
    print("\n" + "=" * 60)
    print("📊 测试结果总结:")
    print(f"  • 正常评估测试: {'✅ 通过' if success1 else '❌ 失败'}")
    print(f"  • 小数据集测试: {'✅ 通过' if success2 else '❌ 失败'}")
    
    if success1 and success2:
        print("🎉 所有测试通过！分布式评估修复成功")
        sys.exit(0)
    else:
        print("❌ 部分测试失败，需要进一步调试")
        sys.exit(1) 