#!/bin/bash

# Qwen2.5-VL食物分类多GPU训练脚本 (默认8GPU) - 修复端口冲突

# 配置参数
CONFIG_FILE="configs/food2k_cosine_5e_6_ls.yaml"
DEEPSPEED_CONFIG="configs/ds_s2.json"
NUM_GPUS=8

# 设置代理（如果需要）
export http_proxy=http://oversea-squid1.jp.txyun:11080 
export https_proxy=http://oversea-squid1.jp.txyun:11080
wandb login f3b76ea66a38b2a211dc706fa95b02c761994b73

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 🔥 修复端口冲突：设置不同的端口
export MASTER_PORT=29501  # 使用29501端口，避免29500冲突
export MASTER_ADDR=localhost

# 启动多GPU分布式训练
echo "🔥 启动多GPU分布式训练..."
echo "📊 使用端口: $MASTER_PORT"
deepspeed --num_gpus=$NUM_GPUS \
    training/train.py \
    --config $CONFIG_FILE \
    --deepspeed_config $DEEPSPEED_CONFIG

echo "✅ 训练脚本执行完成！" 