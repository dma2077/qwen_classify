#!/bin/bash

# 检查可用端口的脚本

echo "🔍 检查可用端口..."

# 检查常用端口
ports=(29500 29501 29502 29503 29504 29505 29506 29507 29508 29509)

for port in "${ports[@]}"; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "❌ 端口 $port 已被占用"
    else
        echo "✅ 端口 $port 可用"
    fi
done

echo ""
echo "📊 当前占用29500端口的进程:"
lsof -i :29500 2>/dev/null || echo "  没有进程占用29500端口"

echo ""
echo "💡 建议:"
echo "  1. 使用可用的端口，如29501, 29502等"
echo "  2. 或者设置 MASTER_PORT=0 让系统自动分配"
echo "  3. 在训练脚本中添加: export MASTER_PORT=29501" 