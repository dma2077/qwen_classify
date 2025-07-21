#!/bin/bash

# Qwen2.5-VL食物分类多GPU训练脚本 (默认8GPU)

# 配置参数
CONFIG_FILE="configs/foodx251_cosine_hold_5e_6_ls.yaml"
DEEPSPEED_CONFIG="configs/ds_s2_as_8.json"
NUM_GPUS=8

# 设置代理（如果需要）
export http_proxy=http://oversea-squid1.jp.txyun:11080 
export https_proxy=http://oversea-squid1.jp.txyun:11080
wandb login f3b76ea66a38b2a211dc706fa95b02c761994b73

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 启动多GPU分布式训练
echo "🔥 启动多GPU分布式训练..."
deepspeed --num_gpus=$NUM_GPUS \
    training/train.py \
    --config $CONFIG_FILE \
    --deepspeed_config $DEEPSPEED_CONFIG

echo "✅ 训练脚本执行完成！" 
