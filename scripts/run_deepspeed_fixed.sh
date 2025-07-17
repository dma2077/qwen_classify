#!/bin/bash

# Qwen2.5-VL食物分类多GPU训练脚本 (修复版本)

# 配置参数
CONFIG_FILE="configs/config_1e_5_ls.yaml"
DEEPSPEED_CONFIG="configs/ds_s2.json"
NUM_GPUS=8

# 设置代理（如果需要）
export http_proxy=http://oversea-squid1.jp.txyun:11080 
export https_proxy=http://oversea-squid1.jp.txyun:11080

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 🔥 修复DeepSpeed卡住问题的环境变量
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_TIMEOUT=1800
export NCCL_BLOCKING_WAIT=1

# 设置分布式训练环境变量
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0

# 清理可能的残留进程
echo "🧹 清理可能的残留进程..."
pkill -f "deepspeed" || true
pkill -f "python.*train.py" || true

# 等待端口释放
sleep 2

# 检查GPU状态
echo "🔍 检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits

# 启动多GPU分布式训练
echo "🔥 启动多GPU分布式训练..."
deepspeed --num_gpus=$NUM_GPUS \
    --no_python \
    --no_local_rank \
    training/train.py \
    --config $CONFIG_FILE \
    --deepspeed_config $DEEPSPEED_CONFIG

echo "✅ 训练脚本执行完成！" 