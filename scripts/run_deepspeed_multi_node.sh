#!/bin/bash

# Qwen2.5-VL食物分类多机多GPU训练脚本

# 配置参数
CONFIG_FILE="configs/config_1e_5_ls.yaml"
DEEPSPEED_CONFIG="configs/ds_s2.json"
HOSTFILE="configs/hostfile"

# 设置代理（如果需要）
export http_proxy=http://oversea-squid1.jp.txyun:11080 
export https_proxy=http://oversea-squid1.jp.txyun:11080

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 检查hostfile是否存在
if [ ! -f "$HOSTFILE" ]; then
    echo "❌ 错误: hostfile不存在: $HOSTFILE"
    echo "请先创建hostfile并配置机器信息"
    exit 1
fi

# 显示hostfile内容
echo "📋 当前hostfile配置:"
cat $HOSTFILE
echo ""

# 启动多机分布式训练
echo "🔥 启动多机分布式训练..."
deepspeed --hostfile $HOSTFILE \
    training/train.py \
    --config $CONFIG_FILE \
    --deepspeed_config $DEEPSPEED_CONFIG

echo "✅ 多机训练脚本执行完成！" 