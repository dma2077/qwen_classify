#!/usr/bin/env python3
"""
检查GLIBC兼容性并解决FlashAttention安装问题
"""

import os
import sys
import subprocess
import platform

def check_glibc_version():
    """检查GLIBC版本"""
    try:
        # 方法1: 使用ldd检查
        result = subprocess.run(['ldd', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'ldd' in line and 'GLIBC' in line:
                    print(f"📋 GLIBC版本: {line.strip()}")
                    return line.strip()
        
        # 方法2: 检查libc.so.6
        result = subprocess.run(['strings', '/lib/x86_64-linux-gnu/libc.so.6'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if line.startswith('GLIBC_'):
                    print(f"📋 发现GLIBC版本: {line}")
        
        return "未知"
    except Exception as e:
        print(f"❌ 检查GLIBC版本失败: {e}")
        return "检查失败"

def check_flash_attention():
    """检查FlashAttention安装状态"""
    try:
        import flash_attn
        print("✅ FlashAttention已安装")
        return True
    except ImportError as e:
        print(f"❌ FlashAttention未安装: {e}")
        return False
    except Exception as e:
        print(f"❌ FlashAttention导入失败: {e}")
        return False

def suggest_solutions():
    """建议解决方案"""
    print("\n🔧 解决方案建议:")
    print("1. 使用conda安装（推荐）:")
    print("   conda install -c conda-forge flash-attn")
    print()
    print("2. 安装较旧版本:")
    print("   pip install flash-attn==2.3.6 --no-build-isolation")
    print()
    print("3. 使用预编译版本:")
    print("   pip install flash-attn --no-build-isolation --index-url https://download.pytorch.org/whl/cu121")
    print()
    print("4. 如果GLIBC版本过低，可以:")
    print("   - 升级系统GLIBC")
    print("   - 使用Docker容器")
    print("   - 使用eager attention（性能稍差但兼容性好）")

def main():
    """主函数"""
    print("🔍 检查系统兼容性...")
    
    # 检查系统信息
    print(f"🖥️  系统: {platform.system()} {platform.release()}")
    print(f"🐍 Python版本: {sys.version}")
    
    # 检查GLIBC版本
    glibc_version = check_glibc_version()
    
    # 检查FlashAttention
    flash_attn_ok = check_flash_attention()
    
    if not flash_attn_ok:
        suggest_solutions()
    else:
        print("🎉 系统兼容性检查通过!")

if __name__ == "__main__":
    main() 