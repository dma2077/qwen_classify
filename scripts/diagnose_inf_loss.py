#!/usr/bin/env python3
"""
诊断脚本：检查多数据集训练中Loss为inf的问题
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def diagnose_dataset_labels(config_path):
    """诊断数据集标签是否正确"""
    
    print("🔍 开始诊断数据集标签映射...")
    print("=" * 80)
    
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_configs = config.get('datasets', {}).get('dataset_configs', {})
    if not dataset_configs:
        print("❌ 没有找到多数据集配置")
        return False
    
    print("📋 数据集配置:")
    for dataset_name, dataset_config in dataset_configs.items():
        num_classes = dataset_config.get('num_classes', 'N/A')
        print(f"  • {dataset_name}: {num_classes} classes")
    
    # 检查数据文件
    data_config = config.get('data', {})
    
    if 'val_jsonl_list' in data_config:
        jsonl_files = data_config['val_jsonl_list']
    elif 'val_jsonl' in data_config:
        jsonl_files = [data_config['val_jsonl']]
    else:
        print("❌ 没有找到验证数据文件配置")
        return False
    
    print("\n🔍 检查数据文件中的标签范围...")
    
    import json
    dataset_label_stats = {}
    
    for jsonl_file in jsonl_files:
        if not os.path.exists(jsonl_file):
            print(f"⚠️ 文件不存在: {jsonl_file}")
            continue
            
        print(f"\n📂 分析文件: {jsonl_file}")
        
        with open(jsonl_file, 'r') as f:
            line_count = 0
            for line in f:
                try:
                    line_count += 1
                    item = json.loads(line.strip())
                    
                    label = int(item.get('label', -1))
                    dataset_name = item.get('dataset_name', 'unknown')
                    
                    if dataset_name not in dataset_label_stats:
                        dataset_label_stats[dataset_name] = {
                            'min_label': float('inf'),
                            'max_label': -1,
                            'labels': set(),
                            'count': 0
                        }
                    
                    stats = dataset_label_stats[dataset_name]
                    stats['min_label'] = min(stats['min_label'], label)
                    stats['max_label'] = max(stats['max_label'], label)
                    stats['labels'].add(label)
                    stats['count'] += 1
                    
                    if line_count <= 5:  # 显示前5行示例
                        print(f"  示例 {line_count}: dataset={dataset_name}, label={label}")
                        
                except Exception as e:
                    print(f"  ❌ 解析第{line_count}行出错: {e}")
                    
                if line_count >= 1000:  # 只检查前1000行
                    print(f"  ℹ️ 已检查前{line_count}行...")
                    break
    
    print("\n📊 标签统计结果:")
    print("=" * 80)
    
    has_issues = False
    
    for dataset_name, stats in dataset_label_stats.items():
        expected_classes = dataset_configs.get(dataset_name, {}).get('num_classes')
        min_label = stats['min_label']
        max_label = stats['max_label']
        unique_labels = len(stats['labels'])
        
        print(f"\n🔍 {dataset_name}:")
        print(f"  • 配置的类别数: {expected_classes}")
        print(f"  • 实际标签范围: {min_label} - {max_label}")
        print(f"  • 唯一标签数量: {unique_labels}")
        print(f"  • 样本数量: {stats['count']}")
        
        # 检查问题
        issues = []
        
        if expected_classes is not None:
            if max_label >= expected_classes:
                issues.append(f"最大标签{max_label}超出类别范围[0, {expected_classes-1}]")
                has_issues = True
                
            if min_label < 0:
                issues.append(f"发现负标签{min_label}")
                has_issues = True
                
            if unique_labels != expected_classes:
                issues.append(f"唯一标签数{unique_labels}与配置的类别数{expected_classes}不匹配")
        
        if issues:
            print(f"  ❌ 发现问题:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(f"  ✅ 标签映射正常")
    
    return not has_issues

def diagnose_logits_masking(config_path):
    """诊断logits masking逻辑"""
    
    print("\n🔍 诊断logits masking逻辑...")
    print("=" * 80)
    
    try:
        from models.qwen2_5_vl_classify import Qwen2_5_VLForImageClassification
        import yaml
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        dataset_configs = config.get('datasets', {}).get('dataset_configs', {})
        enable_logits_masking = config.get('datasets', {}).get('enable_logits_masking', True)
        
        print(f"📋 Logits masking配置:")
        print(f"  • 启用状态: {enable_logits_masking}")
        print(f"  • 数据集数量: {len(dataset_configs)}")
        
        # 创建测试模型（不加载权重）
        print("\n🧪 创建测试用的模型实例...")
        
        # 创建最小配置用于测试
        test_config = {
            'model': {
                'pretrained_name': config['model']['pretrained_name'],
                'num_labels': config['model']['num_labels']
            },
            'loss': {'type': 'cross_entropy'},
            'datasets': config.get('datasets', {})
        }
        
        # 模拟logits masking逻辑（不需要实际加载模型）
        print("🧪 模拟logits masking过程...")
        
        num_labels = config['model']['num_labels']
        batch_size = 4
        
        # 创建测试logits
        test_logits = torch.randn(batch_size, num_labels)
        print(f"  • 原始logits形状: {test_logits.shape}")
        print(f"  • 原始logits范围: [{test_logits.min():.3f}, {test_logits.max():.3f}]")
        
        # 模拟不同数据集的masking
        for dataset_name, dataset_config in dataset_configs.items():
            num_classes = dataset_config.get('num_classes')
            if num_classes is None:
                continue
                
            print(f"\n  🔍 测试 {dataset_name} (classes: {num_classes}):")
            
            # 应用masking
            test_masked = test_logits.clone()
            if enable_logits_masking and num_classes < num_labels:
                test_masked[:, num_classes:] = float('-inf')
                
                # 检查结果
                valid_range = test_masked[:, :num_classes]
                masked_range = test_masked[:, num_classes:]
                
                print(f"    • 有效logits范围[0:{num_classes}]: [{valid_range.min():.3f}, {valid_range.max():.3f}]")
                print(f"    • 被mask的logits[{num_classes}:]: {masked_range[0][:5].tolist()}")
                
                # 检查是否所有有效logits都是-inf
                if torch.all(torch.isinf(valid_range)):
                    print(f"    ❌ 警告: 所有有效位置都是-inf!")
                else:
                    print(f"    ✅ 有效位置包含正常值")
                    
                # 测试softmax结果
                try:
                    softmax_result = torch.softmax(test_masked, dim=-1)
                    if torch.any(torch.isnan(softmax_result)) or torch.any(torch.isinf(softmax_result)):
                        print(f"    ❌ Softmax产生了NaN或Inf")
                    else:
                        print(f"    ✅ Softmax结果正常")
                except Exception as e:
                    print(f"    ❌ Softmax计算失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Logits masking诊断失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def diagnose_loss_computation():
    """诊断损失计算中的数值问题"""
    
    print("\n🔍 诊断损失计算数值稳定性...")
    print("=" * 80)
    
    # 测试各种极端情况
    test_cases = [
        ("正常情况", torch.randn(4, 101), torch.randint(0, 101, (4,))),
        ("logits有-inf", torch.cat([torch.randn(4, 50), torch.full((4, 51), float('-inf'))], dim=1), torch.randint(0, 50, (4,))),
        ("logits全为-inf", torch.full((4, 101), float('-inf')), torch.randint(0, 101, (4,))),
        ("logits很大值", torch.randn(4, 101) * 100, torch.randint(0, 101, (4,))),
        ("标签越界", torch.randn(4, 101), torch.tensor([0, 50, 99, 150])),  # 150超出范围
    ]
    
    for case_name, logits, labels in test_cases:
        print(f"\n🧪 测试: {case_name}")
        print(f"  • logits形状: {logits.shape}")
        print(f"  • logits范围: [{logits.min():.3f}, {logits.max():.3f}]")
        print(f"  • labels: {labels.tolist()}")
        
        try:
            # 标准CrossEntropy
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction='mean')
            print(f"  • CrossEntropy Loss: {loss.item():.6f}")
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"    ❌ 损失为NaN或Inf!")
            else:
                print(f"    ✅ 损失计算正常")
                
        except Exception as e:
            print(f"    ❌ 损失计算失败: {e}")

def create_fixed_config(original_config_path, output_path):
    """创建修复后的配置文件"""
    
    print(f"\n🔧 创建修复后的配置文件...")
    print("=" * 80)
    
    import yaml
    
    with open(original_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 修复建议
    fixes_applied = []
    
    # 1. 降低学习率
    if config.get('training', {}).get('lr', 1e-5) > 1e-6:
        config['training']['lr'] = 1e-6
        fixes_applied.append("降低学习率到1e-6")
    
    # 2. 添加梯度裁剪
    if 'max_grad_norm' not in config.get('training', {}):
        config['training']['max_grad_norm'] = 1.0
        fixes_applied.append("添加梯度裁剪 (max_grad_norm=1.0)")
    
    # 3. 暂时禁用logits masking
    if config.get('datasets', {}).get('enable_logits_masking', True):
        config['datasets']['enable_logits_masking'] = False
        fixes_applied.append("暂时禁用logits masking")
    
    # 4. 使用更稳定的损失函数
    if config.get('loss', {}).get('type') != 'cross_entropy':
        config['loss'] = {'type': 'cross_entropy'}
        fixes_applied.append("使用标准CrossEntropy损失")
    
    # 5. 添加数值稳定性配置
    if 'numerical_stability' not in config:
        config['numerical_stability'] = {
            'check_inf_loss': True,
            'clip_logits': True,
            'logits_clip_value': 10.0
        }
        fixes_applied.append("添加数值稳定性检查")
    
    # 保存修复后的配置
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 修复后的配置已保存到: {output_path}")
    print(f"🔧 应用的修复:")
    for fix in fixes_applied:
        print(f"  • {fix}")
    
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scripts/diagnose_inf_loss.py <config_file.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)
    
    print("🚀 开始诊断多数据集训练的Loss=inf问题")
    print("这个脚本将检查:")
    print("1. 数据集标签映射是否正确")
    print("2. Logits masking逻辑是否有问题") 
    print("3. 损失计算的数值稳定性")
    print("4. 生成修复后的配置文件")
    print("")
    
    # 运行诊断
    label_ok = diagnose_dataset_labels(config_path)
    masking_ok = diagnose_logits_masking(config_path)
    diagnose_loss_computation()
    
    # 创建修复配置
    fixed_config_path = config_path.replace('.yaml', '_fixed.yaml')
    create_fixed_config(config_path, fixed_config_path)
    
    print("\n" + "=" * 80)
    print("📋 诊断结果汇总:")
    print(f"  • 标签映射检查: {'✅ 正常' if label_ok else '❌ 有问题'}")
    print(f"  • Logits masking检查: {'✅ 正常' if masking_ok else '❌ 有问题'}")
    print(f"  • 修复配置文件: {fixed_config_path}")
    
    if not label_ok or not masking_ok:
        print("\n⚠️ 发现问题！建议:")
        print("1. 使用修复后的配置文件重新训练")
        print("2. 检查数据文件中的标签是否正确")
        print("3. 考虑暂时禁用logits masking")
        print("4. 降低学习率并添加梯度裁剪")
    else:
        print("\n✅ 基础检查通过，但仍建议使用修复后的配置以提高稳定性") 