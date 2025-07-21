#!/bin/bash

# 无评估训练脚本 - 只进行训练和保存所有checkpoint

set -e

echo "🚀 开始无评估训练 - 保存所有checkpoint模式"
echo "================================"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# 检查端口是否可用
if lsof -Pi :29500 -sTCP:LISTEN -t >/dev/null ; then
    echo "❌ 端口 29500 已被占用，请更换端口或终止占用进程"
    exit 1
fi

# 训练参数
CONFIG_FILE="configs/food101_no_eval_save_all.yaml"
NUM_GPUS=4

echo "📋 训练配置:"
echo "  • 配置文件: $CONFIG_FILE"
echo "  • GPU数量: $NUM_GPUS"
echo "  • 跳过评估: 是"
echo "  • 保存所有checkpoint: 是"
echo "  • 最佳模型追踪: 禁用"
echo ""

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 运行训练
echo "🎯 启动训练..."
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    training/deepspeed_trainer.py \
    --config "$CONFIG_FILE"

echo "🎉 训练完成！"
echo ""
echo "📂 检查输出目录中的所有checkpoint:"
echo "   /mmu_mllm_hdd_2/madehua/model/qwen_classify/food101_no_eval_save_all/"
echo ""
echo "💡 提示:"
echo "  • 此模式下所有的checkpoint都会被保存"
echo "  • 没有进行任何评估，训练速度更快"
echo "  • 模型质量需要在训练完成后单独评估" 