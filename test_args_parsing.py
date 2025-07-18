#!/usr/bin/env python3
"""
测试参数解析
"""

import os
import sys
import argparse
import deepspeed

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_args_parsing():
    """测试参数解析"""
    print("🔍 测试参数解析")
    print("="*50)
    
    # 模拟命令行参数
    test_args = [
        "training/complete_train.py",
        "--config", "configs/food101_cosine_hold.yaml",
        "--deepspeed_config", "configs/ds_s2.json",
        "--seed", "42"
    ]
    
    print(f"📋 模拟命令行参数:")
    print(f"  {' '.join(test_args)}")
    print()
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Qwen2.5-VL图像分类完整训练")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--local_rank", type=int, default=-1, help="本地进程排名")
    parser.add_argument("--resume_from", type=str, help="恢复训练的检查点路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # 添加DeepSpeed参数
    parser = deepspeed.add_config_arguments(parser)
    
    print("📋 解析参数...")
    try:
        args = parser.parse_args(test_args[1:])  # 跳过脚本名
        print(f"✅ 参数解析成功")
        print(f"  • config: {args.config}")
        print(f"  • deepspeed_config: {getattr(args, 'deepspeed_config', 'NOT_FOUND')}")
        print(f"  • seed: {args.seed}")
        print(f"  • local_rank: {args.local_rank}")
        
        # 检查DeepSpeed配置
        if hasattr(args, 'deepspeed_config') and args.deepspeed_config:
            if os.path.exists(args.deepspeed_config):
                print(f"  ✅ DeepSpeed配置文件存在: {args.deepspeed_config}")
            else:
                print(f"  ❌ DeepSpeed配置文件不存在: {args.deepspeed_config}")
        else:
            print(f"  ❌ DeepSpeed配置文件未指定")
            
    except Exception as e:
        print(f"❌ 参数解析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_args_parsing() 