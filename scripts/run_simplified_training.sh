#!/bin/bash

# 简化的训练启动脚本
# 去除FlashAttention相关的环境变量设置，让DeepSpeed完全控制数值精度

set -e

# 配置参数
CONFIG_FILE="configs/config_3e_6_ls.yaml"
DEEPSPEED_CONFIG="configs/ds_s2.json"
NUM_GPUS=8
MASTER_PORT=29500

echo "🚀 启动简化训练..."
echo "📋 配置文件: $CONFIG_FILE"
echo "📋 DeepSpeed配置: $DEEPSPEED_CONFIG"
echo "📋 GPU数量: $NUM_GPUS"

# 启动训练
deepspeed --num_gpus=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    training/train.py \
    --config $CONFIG_FILE \
    --deepspeed_config $DEEPSPEED_CONFIG

echo "✅ 训练完成" 