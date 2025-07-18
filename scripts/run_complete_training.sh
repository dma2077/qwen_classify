#!/bin/bash

# 完整的Qwen2.5-VL图像分类训练启动脚本
# 包含FlashAttention、DeepSpeed、WandB监控

set -e

# 配置参数
CONFIG_FILE="configs/complete_training_config.yaml"
DEEPSPEED_CONFIG="configs/ds_config_zero2.json"
NUM_GPUS=8
MASTER_PORT=29500

# 检查flash-attn是否安装
echo "🔍 检查FlashAttention安装..."
python scripts/check_glibc_compatibility.py

# 如果FlashAttention不可用，继续使用eager attention
python -c "import flash_attn; print('✅ FlashAttention可用')" || {
    echo "⚠️ FlashAttention不可用，将使用eager attention（性能稍差但兼容性好）"
    echo "💡 如需安装FlashAttention，请运行:"
    echo "   conda install -c conda-forge flash-attn"
    echo "   或: pip install flash-attn==2.3.6 --no-build-isolation"
}

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "❌ DeepSpeed配置文件不存在: $DEEPSPEED_CONFIG"
    exit 1
fi

echo "🚀 开始训练..."
echo "📋 配置文件: $CONFIG_FILE"
echo "⚙️  DeepSpeed配置: $DEEPSPEED_CONFIG"
echo "🖥️  GPU数量: $NUM_GPUS"

# 启动训练
deepspeed \
    --num_gpus $NUM_GPUS \
    --master_port $MASTER_PORT \
    training/complete_train.py \
    --config $CONFIG_FILE \
    --deepspeed_config $DEEPSPEED_CONFIG \
    --seed 42

echo "🎉 训练完成!" 