#!/bin/bash

# Qwen2.5-VL食物分类多GPU训练脚本 (默认8GPU) - 修复端口冲突版本

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
# 方法1: 使用环境变量设置端口
export MASTER_PORT=29501  # 使用29501端口
export MASTER_ADDR=localhost

# 方法2: 如果29501也被占用，可以尝试其他端口
# export MASTER_PORT=29502
# export MASTER_ADDR=localhost

# 方法3: 使用随机端口（推荐）
# export MASTER_PORT=0  # 让系统自动分配可用端口
# export MASTER_ADDR=localhost

# 检查端口是否被占用
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️  端口 $port 已被占用"
        return 1
    else
        echo "✅ 端口 $port 可用"
        return 0
    fi
}

# 自动选择可用端口
select_available_port() {
    local start_port=29500
    local max_attempts=10
    
    for i in $(seq 0 $max_attempts); do
        local port=$((start_port + i))
        if check_port $port; then
            export MASTER_PORT=$port
            export MASTER_ADDR=localhost
            echo "🎯 使用端口: $port"
            return 0
        fi
    done
    
    echo "❌ 无法找到可用端口，使用随机端口"
    export MASTER_PORT=0
    export MASTER_ADDR=localhost
}

# 自动选择可用端口
select_available_port

# 显示当前端口设置
echo "📊 当前端口配置:"
echo "   MASTER_PORT: $MASTER_PORT"
echo "   MASTER_ADDR: $MASTER_ADDR"

# 启动多GPU分布式训练
echo "🔥 启动多GPU分布式训练..."
deepspeed --num_gpus=$NUM_GPUS \
    training/train.py \
    --config $CONFIG_FILE \
    --deepspeed_config $DEEPSPEED_CONFIG

echo "✅ 训练脚本执行完成！" 