#!/bin/bash

# FlashAttention安装脚本 - 解决GLIBC版本问题

echo "🔧 安装FlashAttention..."

# 方案1: 使用conda安装（推荐，解决GLIBC问题）
echo "📦 尝试使用conda安装flash-attn..."
conda install -c conda-forge flash-attn -y

# 如果conda安装失败，尝试其他方案
if [ $? -ne 0 ]; then
    echo "⚠️ conda安装失败，尝试pip安装..."
    
    # 方案2: 使用pip安装预编译版本
    pip install flash-attn --no-build-isolation --index-url https://download.pytorch.org/whl/cu121
    
    # 如果还是失败，尝试方案3
    if [ $? -ne 0 ]; then
        echo "⚠️ pip安装失败，尝试安装较旧版本..."
        pip install flash-attn==2.3.6 --no-build-isolation
    fi
fi

# 验证安装
echo "🔍 验证FlashAttention安装..."
python -c "import flash_attn; print('✅ FlashAttention安装成功!')" || {
    echo "❌ FlashAttention安装失败"
    echo "💡 建议手动安装:"
    echo "   1. conda install -c conda-forge flash-attn"
    echo "   2. 或者: pip install flash-attn==2.3.6 --no-build-isolation"
    exit 1
}

echo "🎉 FlashAttention安装完成!" 