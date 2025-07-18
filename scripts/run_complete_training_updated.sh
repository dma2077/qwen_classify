#!/bin/bash

# 完整的Qwen2.5-VL图像分类训练启动脚本
# 使用complete_train.py，包含FlashAttention、DeepSpeed、WandB监控

set -e

# 配置参数
CONFIG_FILE="configs/complete_training_config.yaml"
DEEPSPEED_CONFIG="configs/ds_config_zero2.json"
NUM_GPUS=8
MASTER_PORT=29500
SEED=42

# 显示帮助信息
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "用法: $0 [配置文件] [DeepSpeed配置] [GPU数量] [主端口] [随机种子]"
    echo ""
    echo "参数:"
    echo "  配置文件         - 训练配置文件路径 (默认: $CONFIG_FILE)"
    echo "  DeepSpeed配置    - DeepSpeed配置文件路径 (默认: $DEEPSPEED_CONFIG)"
    echo "  GPU数量          - 使用的GPU数量 (默认: $NUM_GPUS)"
    echo "  主端口           - 分布式训练主端口 (默认: $MASTER_PORT)"
    echo "  随机种子         - 随机种子 (默认: $SEED)"
    echo ""
    echo "示例:"
    echo "  $0                                    # 使用默认配置"
    echo "  $0 configs/complete_training_config.yaml  # 指定配置文件"
    echo "  $0 configs/complete_training_config.yaml configs/ds_config_zero2.json 4 29501 123  # 完整参数"
    exit 0
fi

# 解析命令行参数
CONFIG_FILE=${1:-$CONFIG_FILE}
DEEPSPEED_CONFIG=${2:-$DEEPSPEED_CONFIG}
NUM_GPUS=${3:-$NUM_GPUS}
MASTER_PORT=${4:-$MASTER_PORT}
SEED=${5:-$SEED}

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    echo "💡 可用的配置文件:"
    ls -1 configs/*.yaml 2>/dev/null || echo "  无配置文件"
    exit 1
fi

if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "❌ DeepSpeed配置文件不存在: $DEEPSPEED_CONFIG"
    echo "💡 可用的DeepSpeed配置:"
    ls -1 configs/ds_*.json 2>/dev/null || echo "  无DeepSpeed配置"
    exit 1
fi

# 检查complete_train.py是否存在
if [ ! -f "training/complete_train.py" ]; then
    echo "❌ 训练脚本不存在: training/complete_train.py"
    echo "💡 请确保complete_train.py文件存在"
    exit 1
fi

# 检查FlashAttention（可选）
echo "🔍 检查FlashAttention安装..."
python scripts/check_glibc_compatibility.py

# 如果FlashAttention不可用，继续使用eager attention
python -c "import flash_attn; print('✅ FlashAttention可用')" || {
    echo "⚠️ FlashAttention不可用，将使用eager attention（性能稍差但兼容性好）"
    echo "💡 如需安装FlashAttention，请运行:"
    echo "   conda install -c conda-forge flash-attn"
    echo "   或: pip install flash-attn==2.3.6 --no-build-isolation"
}

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

echo "🚀 启动完整Qwen2.5-VL图像分类训练"
echo "=" * 60
echo "📋 配置文件: $CONFIG_FILE"
echo "⚙️  DeepSpeed配置: $DEEPSPEED_CONFIG"
echo "🖥️  GPU数量: $NUM_GPUS"
echo "🔌 主端口: $MASTER_PORT"
echo "🎲 随机种子: $SEED"
echo "📝 训练脚本: training/complete_train.py"
echo "=" * 60

# 启动训练
deepspeed \
    --num_gpus $NUM_GPUS \
    --master_port $MASTER_PORT \
    training/complete_train.py \
    --config $CONFIG_FILE \
    --deepspeed_config $DEEPSPEED_CONFIG \
    --seed $SEED

echo "🎉 完整训练完成!" 