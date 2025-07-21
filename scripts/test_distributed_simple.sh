#!/bin/bash

# 简化的分布式测试脚本

# 配置参数
NUM_GPUS=8

# 设置基本环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 清理旧进程
echo "🧹 清理旧进程..."
pkill -f "complete_train.py" || true
pkill -f "check_deepspeed_launch.py" || true
sleep 2

echo "🔍 测试1: 检查DeepSpeed启动..."
deepspeed --num_gpus=$NUM_GPUS check_deepspeed_launch.py

echo ""
echo "🔍 测试2: 如果启动正常，运行简化训练..."
read -p "是否继续运行训练测试？(y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    deepspeed --num_gpus=$NUM_GPUS \
        training/complete_train.py \
        --config configs/foodx251_cosine_5e_6_ls.yaml \
        --deepspeed_config configs/ds_s2_as_8.json
fi 