#!/bin/bash

# Qwen2.5-VL食物分类多GPU训练脚本 (默认8GPU)

# 配置参数
CONFIG_FILE="configs/food172_cosine_hold_5e_6_ls.yaml"
DEEPSPEED_CONFIG="configs/ds_s2.json"
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

# HOSTFILE="/etc/mpi/hostfile"
# # 显示hostfile内容
# echo "📋 当前hostfile配置:"
# cat $HOSTFILE
# echo ""
# # 启动多机分布式训练
# echo "🔥 启动多机分布式训练..."
# echo "🔥 启动多机分布式训练..."
# $PYTHON -m wandb login f3b76ea66a38b2a211dc706fa95b02c761994b73
# $PYTHON -m deepspeed --hostfile $HOSTFILE \
#     training/train.py \
#     --config  $CONFIG_FILE \
#     --deepspeed_config $DEEPSPEED_CONFIG

echo "✅ 训练脚本执行完成！" 

