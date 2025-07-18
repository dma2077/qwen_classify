#!/bin/bash

# Qwen2.5-VL图像分类训练快速启动脚本
# 最简单的使用方式

set -e

echo "🚀 Qwen2.5-VL图像分类训练快速启动"
echo "=" * 40

# 检查是否有GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "🖥️  检测到 $GPU_COUNT 个GPU"
    
    # 根据GPU数量自动调整配置
    if [ "$GPU_COUNT" -ge 8 ]; then
        NUM_GPUS=8
        echo "📋 使用8个GPU进行训练"
    elif [ "$GPU_COUNT" -ge 4 ]; then
        NUM_GPUS=4
        echo "📋 使用4个GPU进行训练"
    elif [ "$GPU_COUNT" -ge 2 ]; then
        NUM_GPUS=2
        echo "📋 使用2个GPU进行训练"
    else
        NUM_GPUS=1
        echo "📋 使用1个GPU进行训练"
    fi
else
    echo "⚠️  未检测到GPU，使用CPU模式（不推荐）"
    NUM_GPUS=1
fi

# 检查配置文件
if [ -f "configs/config_3e_6_ls.yaml" ]; then
    CONFIG_FILE="configs/config_3e_6_ls.yaml"
    echo "📋 使用配置文件: $CONFIG_FILE"
elif [ -f "configs/complete_training_config.yaml" ]; then
    CONFIG_FILE="configs/complete_training_config.yaml"
    echo "📋 使用配置文件: $CONFIG_FILE"
else
    echo "❌ 未找到配置文件"
    echo "💡 请确保以下文件之一存在:"
    echo "   - configs/config_3e_6_ls.yaml"
    echo "   - configs/complete_training_config.yaml"
    exit 1
fi

# 检查DeepSpeed配置
if [ -f "configs/ds_s2.json" ]; then
    DEEPSPEED_CONFIG="configs/ds_s2.json"
    echo "⚙️  使用DeepSpeed配置: $DEEPSPEED_CONFIG"
elif [ -f "configs/ds_config_zero2.json" ]; then
    DEEPSPEED_CONFIG="configs/ds_config_zero2.json"
    echo "⚙️  使用DeepSpeed配置: $DEEPSPEED_CONFIG"
else
    echo "❌ 未找到DeepSpeed配置文件"
    exit 1
fi

# 检查训练脚本
if [ -f "training/complete_train.py" ]; then
    TRAIN_SCRIPT="training/complete_train.py"
    echo "📝 使用训练脚本: $TRAIN_SCRIPT"
elif [ -f "training/train.py" ]; then
    TRAIN_SCRIPT="training/train.py"
    echo "📝 使用训练脚本: $TRAIN_SCRIPT"
else
    echo "❌ 未找到训练脚本"
    exit 1
fi

echo "=" * 40
echo "🎯 开始训练..."

# 启动训练
if [ "$TRAIN_SCRIPT" = "training/complete_train.py" ]; then
    # 使用complete_train.py
    deepspeed \
        --num_gpus $NUM_GPUS \
        --master_port 29500 \
        $TRAIN_SCRIPT \
        --config $CONFIG_FILE \
        --deepspeed_config $DEEPSPEED_CONFIG \
        --seed 42
else
    # 使用train.py
    deepspeed \
        --num_gpus $NUM_GPUS \
        --master_port 29500 \
        $TRAIN_SCRIPT \
        --config $CONFIG_FILE \
        --deepspeed_config $DEEPSPEED_CONFIG
fi

echo "🎉 训练完成!" 