#!/usr/bin/env python3
"""
删除Python项目中的__pycache__目录
"""

import shutil
from pathlib import Path

def main():
    """查找并删除所有__pycache__目录"""
    
    # 从当前目录开始搜索
    root_path = Path('.')
    
    print("🔍 搜索__pycache__目录...")
    
    # 查找所有__pycache__目录
    pycache_dirs = list(root_path.rglob('__pycache__'))
    
    if not pycache_dirs:
        print("✅ 没有找到__pycache__目录")
        return
    
    print(f"📁 找到 {len(pycache_dirs)} 个__pycache__目录")
    
    # 删除所有__pycache__目录
    deleted_count = 0
    
    for pycache_dir in pycache_dirs:
        try:
            print(f"🗑️  删除: {pycache_dir}")
            shutil.rmtree(pycache_dir)
            deleted_count += 1
        except Exception as e:
            print(f"❌ 删除失败: {pycache_dir} - {e}")
    
    print(f"✅ 成功删除 {deleted_count} 个__pycache__目录")

if __name__ == "__main__":
    main() 