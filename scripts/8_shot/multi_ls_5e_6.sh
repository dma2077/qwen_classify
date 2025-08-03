#!/bin/bash

# Qwen2.5-VL食物分类多GPU训练脚本 (默认8GPU)

# 配置参数
CONFIG_FILE="configs/8_shot/multi_cosine_5e_6_ls.yaml"
DEEPSPEED_CONFIG="configs/ds_s2_as_8.json"
NUM_GPUS=8

# 设置代理（如果需要）
export http_proxy=http://oversea-squid1.jp.txyun:11080 
export https_proxy=http://oversea-squid1.jp.txyun:11080
wandb login f3b76ea66a38b2a211dc706fa95b02c761994b73

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

MASTER_PORT=29502
# 启动多GPU分布式训练
echo "🔥 启动多GPU分布式训练..."
nohup deepspeed --master_port=$MASTER_PORT --num_gpus=$NUM_GPUS \
    training/train.py \
    --config $CONFIG_FILE \
    --deepspeed_config $DEEPSPEED_CONFIG > logs/multi_dataset_epoch5_8shot.log 2>&1 &

echo "✅ 训练脚本执行完成！" 
