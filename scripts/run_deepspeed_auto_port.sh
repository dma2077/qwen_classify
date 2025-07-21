#!/bin/bash

# Qwen2.5-VL食物分类多GPU训练脚本 - 自动端口检测版本

# 配置参数
CONFIG_FILE="configs/food2k_cosine_5e_6_ls.yaml"
DEEPSPEED_CONFIG="configs/ds_s2.json"
NUM_GPUS=8

# 🔥 修复NCCL警告：首先设置NCCL_NTHREADS
export NCCL_NTHREADS=64
echo "🔧 设置 NCCL_NTHREADS=$NCCL_NTHREADS"

# 设置代理（如果需要）
export http_proxy=http://oversea-squid1.jp.txyun:11080 
export https_proxy=http://oversea-squid1.jp.txyun:11080
wandb login f3b76ea66a38b2a211dc706fa95b02c761994b73

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 🔥 自动检测可用端口
find_available_port() {
    local start_port=29500
    local max_attempts=20
    
    for i in $(seq 0 $max_attempts); do
        local port=$((start_port + i))
        if ! lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            echo $port
            return 0
        fi
    done
    
    # 如果找不到可用端口，使用随机端口
    echo "0"
}

# 获取可用端口
AVAILABLE_PORT=$(find_available_port)

if [ "$AVAILABLE_PORT" = "0" ]; then
    echo "⚠️  无法找到可用端口，使用随机端口"
    export MASTER_PORT=0
else
    echo "🎯 使用端口: $AVAILABLE_PORT"
    export MASTER_PORT=$AVAILABLE_PORT
fi

export MASTER_ADDR=localhost

# 显示端口配置
echo "📊 端口配置:"
echo "   MASTER_PORT: $MASTER_PORT"
echo "   MASTER_ADDR: $MASTER_ADDR"

# 启动多GPU分布式训练
echo "🔥 启动多GPU分布式训练..."
deepspeed --num_gpus=$NUM_GPUS \
    training/train.py \
    --config $CONFIG_FILE \
    --deepspeed_config $DEEPSPEED_CONFIG

echo "✅ 训练脚本执行完成！" 