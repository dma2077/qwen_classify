#!/bin/bash

# Qwen2.5-VL食物分类多GPU训练脚本 (默认8GPU)

# 配置参数
CONFIG_FILE="configs/config_3e_6.yaml"
DEEPSPEED_CONFIG="configs/ds_s2.json"
NUM_GPUS=8

# 设置代理（如果需要）
export http_proxy=http://oversea-squid1.jp.txyun:11080 
export https_proxy=http://oversea-squid1.jp.txyun:11080

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 启动多GPU分布式训练
echo "🔥 启动多GPU分布式训练..."
deepspeed --num_gpus=$NUM_GPUS \
    training/train.py \
    --config $CONFIG_FILE \
    --deepspeed_config $DEEPSPEED_CONFIG

echo "✅ 训练脚本执行完成！" 