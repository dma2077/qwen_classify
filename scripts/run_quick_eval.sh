#!/bin/bash

# 快速评估训练脚本
# 使用20%评估数据，标准批次大小

echo "🚀 启动快速评估训练..."
echo "📊 配置: 20%评估数据, 标准批次(16), 评估间隔500步"

# 检查DeepSpeed配置
DEEPSPEED_CONFIG="configs/ds_minimal.json"
if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "❌ DeepSpeed配置文件不存在: $DEEPSPEED_CONFIG"
    exit 1
fi

# 启动训练
deepspeed --include localhost:0 \
    training/complete_train.py \
    --config configs/food101_cosine.yaml \
    --deepspeed_config $DEEPSPEED_CONFIG

echo "✅ 快速评估训练完成" 