#!/bin/bash

# 配置参数
CONFIG_FILE="configs/food101_cosine_hold_5e_6_ls.yaml"
DEEPSPEED_CONFIG="configs/ds_s2.json"
NUM_GPUS=8

# 设置代理（如果需要）
export http_proxy=http://oversea-squid1.jp.txyun:11080 
export https_proxy=http://oversea-squid1.jp.txyun:11080
wandb login f3b76ea66a38b2a211dc706fa95b02c761994b73

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 精简完整训练启动脚本
deepspeed --num_gpus $NUM_GPUS --master_port 29500 \
    training/complete_train.py \
    --config $CONFIG_FILE \
    --deepspeed_config $DEEPSPEED_CONFIG \
    --seed 42 