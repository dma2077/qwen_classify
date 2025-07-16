#!/usr/bin/env python3
"""
清理检查点脚本
清理指定目录中的多余检查点，只保留最佳模型检查点
"""

import os
import glob
import shutil
import argparse
import json
from pathlib import Path


def find_best_model_step(output_dir):
    """查找最佳模型的步数"""
    best_step = None
    best_value = None
    
    # 检查training_log.json文件
    log_file = os.path.join(output_dir, "training_log.json")
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
            
            # 从日志中查找最佳模型信息
            if 'step_logs' in logs:
                for step_info in logs['step_logs']:
                    if 'best_model_step' in step_info:
                        return step_info['best_model_step']
        except Exception as e:
            print(f"⚠️  读取训练日志失败: {e}")
    
    # 如果无法从日志获取，查找最新的best-model-step目录
    best_model_pattern = os.path.join(output_dir, "best-model-step-*")
    best_model_dirs = glob.glob(best_model_pattern)
    
    if best_model_dirs:
        def extract_step(path):
            try:
                return int(os.path.basename(path).split('-')[-1])
            except:
                return 0
        
        best_model_dirs.sort(key=extract_step)
        return extract_step(best_model_dirs[-1])
    
    return None


def cleanup_directory(output_dir, dry_run=False):
    """清理指定目录中的检查点"""
    if not os.path.exists(output_dir):
        print(f"❌ 目录不存在: {output_dir}")
        return False
    
    print(f"🔍 检查目录: {output_dir}")
    
    # 查找所有检查点目录
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    best_model_pattern = os.path.join(output_dir, "best-model-step-*")
    
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    best_model_dirs = glob.glob(best_model_pattern)
    
    print(f"📊 发现 {len(checkpoint_dirs)} 个常规检查点")
    print(f"📊 发现 {len(best_model_dirs)} 个最佳模型检查点")
    
    if len(checkpoint_dirs) == 0 and len(best_model_dirs) <= 1:
        print("✅ 无需清理，目录已经很干净")
        return True
    
    # 找到最佳模型的步数
    best_step = find_best_model_step(output_dir)
    if best_step:
        print(f"🏆 最佳模型步数: {best_step}")
    
    # 清理策略
    dirs_to_remove = []
    
    # 1. 删除所有常规检查点（checkpoint-*）
    dirs_to_remove.extend(checkpoint_dirs)
    
    # 2. 删除除最新之外的所有最佳模型检查点
    if len(best_model_dirs) > 1:
        def extract_step(path):
            try:
                return int(os.path.basename(path).split('-')[-1])
            except:
                return 0
        
        best_model_dirs.sort(key=extract_step)
        dirs_to_remove.extend(best_model_dirs[:-1])  # 保留最后一个
    
    if not dirs_to_remove:
        print("✅ 无需清理")
        return True
    
    # 显示将要删除的目录
    print(f"\n🗑️  将要删除 {len(dirs_to_remove)} 个检查点:")
    total_size = 0
    for dir_path in dirs_to_remove:
        size = get_directory_size(dir_path)
        total_size += size
        print(f"  - {os.path.basename(dir_path)} ({format_size(size)})")
    
    print(f"\n💾 总计释放空间: {format_size(total_size)}")
    
    if dry_run:
        print("\n🔍 这是预览模式，实际不会删除文件")
        return True
    
    # 确认删除
    if len(dirs_to_remove) > 0:
        response = input(f"\n❓ 确定要删除这 {len(dirs_to_remove)} 个检查点吗? (y/N): ")
        if response.lower() != 'y':
            print("❌ 取消删除")
            return False
    
    # 执行删除
    success_count = 0
    for dir_path in dirs_to_remove:
        try:
            print(f"🗑️  删除: {os.path.basename(dir_path)}")
            shutil.rmtree(dir_path)
            success_count += 1
        except Exception as e:
            print(f"❌ 删除失败 {dir_path}: {e}")
    
    print(f"\n✅ 成功删除 {success_count}/{len(dirs_to_remove)} 个检查点")
    
    # 显示保留的最佳模型
    remaining_best = glob.glob(best_model_pattern)
    if remaining_best:
        print(f"🏆 保留最佳模型: {os.path.basename(remaining_best[0])}")
    
    return success_count == len(dirs_to_remove)


def get_directory_size(path):
    """获取目录大小"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except:
        pass
    return total_size


def format_size(size_bytes):
    """格式化文件大小"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    size_index = 0
    
    while size_bytes >= 1024 and size_index < len(size_names) - 1:
        size_bytes /= 1024.0
        size_index += 1
    
    return f"{size_bytes:.1f}{size_names[size_index]}"


def main():
    parser = argparse.ArgumentParser(description="清理训练检查点，只保留最佳模型")
    parser.add_argument('directories', nargs='+', 
                       help='要清理的输出目录路径')
    parser.add_argument('--dry-run', action='store_true',
                       help='预览模式，不实际删除文件')
    parser.add_argument('--auto-confirm', action='store_true',
                       help='自动确认删除，不询问用户')
    
    args = parser.parse_args()
    
    print("🧹 检查点清理工具")
    print("="*50)
    
    if args.dry_run:
        print("🔍 运行在预览模式")
    
    success_count = 0
    total_count = len(args.directories)
    
    for output_dir in args.directories:
        print(f"\n{'='*50}")
        if cleanup_directory(output_dir, dry_run=args.dry_run):
            success_count += 1
    
    print(f"\n{'='*50}")
    print(f"📊 清理完成: {success_count}/{total_count} 个目录处理成功")
    
    if args.dry_run:
        print("💡 要实际执行清理，请移除 --dry-run 参数")


if __name__ == "__main__":
    main() 