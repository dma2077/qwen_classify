#!/bin/bash

# Qwen2.5-VL食物分类多GPU训练脚本 (修复版本)

# 配置参数
CONFIG_FILE="configs/foodx251_cosine_5e_6_ls.yaml"
DEEPSPEED_CONFIG="configs/ds_s2_as_8.json"
NUM_GPUS=8

# 🔥 修复：针对200核机器的分布式环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=8
export MASTER_ADDR=localhost
export MASTER_PORT=29501

# 针对高核心数机器的NCCL优化
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800  # 30分钟超时，适应高核心数机器
export NCCL_IB_DISABLE=1  # 禁用InfiniBand，使用以太网
export NCCL_SOCKET_IFNAME=^docker,lo  # 排除虚拟网络接口
export NCCL_P2P_DISABLE=1  # 禁用P2P，提高稳定性
export OMP_NUM_THREADS=8   # 限制OpenMP线程数，避免过度并行

# 设置代理（如果需要）
export http_proxy=http://oversea-squid1.jp.txyun:11080 
export https_proxy=http://oversea-squid1.jp.txyun:11080
wandb login f3b76ea66a38b2a211dc706fa95b02c761994b73

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 🔥 修复：清理旧进程
echo "🧹 清理旧进程..."
pkill -f "complete_train.py" || true
sleep 2

# 创建日志目录
mkdir -p logs

# 启动多GPU分布式训练
echo "🔥 启动多GPU分布式训练 (修复版本)..."
echo "  配置文件: $CONFIG_FILE"
echo "  DeepSpeed配置: $DEEPSPEED_CONFIG"
echo "  GPU数量: $NUM_GPUS"

deepspeed --num_gpus=$NUM_GPUS \
    training/complete_train.py \
    --config $CONFIG_FILE \
    --deepspeed_config $DEEPSPEED_CONFIG

echo "🏁 训练脚本结束" 