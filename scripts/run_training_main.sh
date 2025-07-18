#!/bin/bash

# Qwen2.5-VL图像分类训练启动脚本
# 支持多种配置和模式

set -e

# 默认配置
DEFAULT_CONFIG="configs/config_3e_6_ls.yaml"
DEFAULT_DEEPSPEED_CONFIG="configs/ds_s2.json"
DEFAULT_NUM_GPUS=8
DEFAULT_MASTER_PORT=29500

# 解析命令行参数
CONFIG_FILE=${1:-$DEFAULT_CONFIG}
DEEPSPEED_CONFIG=${2:-$DEFAULT_DEEPSPEED_CONFIG}
NUM_GPUS=${3:-$DEFAULT_NUM_GPUS}
MASTER_PORT=${4:-$DEFAULT_MASTER_PORT}

# 显示帮助信息
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "用法: $0 [配置文件] [DeepSpeed配置] [GPU数量] [主端口]"
    echo ""
    echo "参数:"
    echo "  配置文件         - 训练配置文件路径 (默认: $DEFAULT_CONFIG)"
    echo "  DeepSpeed配置    - DeepSpeed配置文件路径 (默认: $DEFAULT_DEEPSPEED_CONFIG)"
    echo "  GPU数量          - 使用的GPU数量 (默认: $DEFAULT_NUM_GPUS)"
    echo "  主端口           - 分布式训练主端口 (默认: $DEFAULT_MASTER_PORT)"
    echo ""
    echo "示例:"
    echo "  $0                                    # 使用默认配置"
    echo "  $0 configs/config_3e_6_ls.yaml       # 指定配置文件"
    echo "  $0 configs/config_3e_6_ls.yaml configs/ds_s2.json 4 29501  # 完整参数"
    echo ""
    echo "可用的配置文件:"
    ls -1 configs/config_*.yaml 2>/dev/null || echo "  无配置文件"
    echo ""
    echo "可用的DeepSpeed配置:"
    ls -1 configs/ds_*.json 2>/dev/null || echo "  无DeepSpeed配置"
    exit 0
fi

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    echo "💡 可用的配置文件:"
    ls -1 configs/config_*.yaml 2>/dev/null || echo "  无配置文件"
    exit 1
fi

if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "❌ DeepSpeed配置文件不存在: $DEEPSPEED_CONFIG"
    echo "💡 可用的DeepSpeed配置:"
    ls -1 configs/ds_*.json 2>/dev/null || echo "  无DeepSpeed配置"
    exit 1
fi

# 检查GPU可用性
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  nvidia-smi未找到，可能没有GPU或CUDA未安装"
else
    AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "🖥️  可用GPU数量: $AVAILABLE_GPUS"
    if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
        echo "⚠️  请求的GPU数量 ($NUM_GPUS) 超过可用数量 ($AVAILABLE_GPUS)"
        echo "💡 自动调整为可用GPU数量"
        NUM_GPUS=$AVAILABLE_GPUS
    fi
fi

echo "🚀 启动Qwen2.5-VL图像分类训练"
echo "=" * 50
echo "📋 配置文件: $CONFIG_FILE"
echo "⚙️  DeepSpeed配置: $DEEPSPEED_CONFIG"
echo "🖥️  GPU数量: $NUM_GPUS"
echo "🔌 主端口: $MASTER_PORT"
echo "=" * 50

# 启动训练
deepspeed \
    --num_gpus $NUM_GPUS \
    --master_port $MASTER_PORT \
    training/train.py \
    --config $CONFIG_FILE \
    --deepspeed_config $DEEPSPEED_CONFIG

echo "🎉 训练完成!" 