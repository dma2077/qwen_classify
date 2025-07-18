#!/bin/bash

# 简化的训练启动脚本
# DeepSpeed配置通过命令行参数传入，YAML中不包含DeepSpeed配置

# 配置参数
CONFIG_FILE="configs/food101_cosine_hold.yaml"
DEEPSPEED_CONFIG="configs/ds_s2.json"
NUM_GPUS=8

# 设置代理（如果需要）
export http_proxy=http://oversea-squid1.jp.txyun:11080 
export https_proxy=http://oversea-squid1.jp.txyun:11080
wandb login f3b76ea66a38b2a211dc706fa95b02c761994b73

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "🚀 启动简化训练..."
echo "  • 配置文件: $CONFIG_FILE"
echo "  • DeepSpeed配置: $DEEPSPEED_CONFIG"
echo "  • GPU数量: $NUM_GPUS"

# 启动训练
deepspeed --num_gpus $NUM_GPUS --master_port 29500 \
    training/complete_train.py \
    --config $CONFIG_FILE \
    --deepspeed_config $DEEPSPEED_CONFIG \
    --seed 42

echo "✅ 训练脚本执行完成！" 