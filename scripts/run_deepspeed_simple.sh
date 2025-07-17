#!/bin/bash

# Qwen2.5-VL食物分类多GPU训练脚本 (简化版本)

# 配置参数
CONFIG_FILE="configs/config_1e_5_ls.yaml"
DEEPSPEED_CONFIG="configs/ds_s2.json"
NUM_GPUS=8

# 设置代理（如果需要）
export http_proxy=http://oversea-squid1.jp.txyun:11080 
export https_proxy=http://oversea-squid1.jp.txyun:11080

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 🔥 最小化环境变量设置
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

# 启动多GPU分布式训练
echo "🔥 启动多GPU分布式训练..."
deepspeed --num_gpus=$NUM_GPUS \
    training/train.py \
    --config $CONFIG_FILE \
    --deepspeed_config $DEEPSPEED_CONFIG

echo "✅ 训练脚本执行完成！" 