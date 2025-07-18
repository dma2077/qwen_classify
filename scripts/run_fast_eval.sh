#!/bin/bash

# 快速评估训练脚本
# 使用10%评估数据，大批次大小，减少评估频率

echo "🚀 启动快速评估训练..."
echo "📊 配置: 10%评估数据, 大批次(32), 评估间隔1000步"

# 检查DeepSpeed配置
DEEPSPEED_CONFIG="configs/ds_large_batch.json"
if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "❌ DeepSpeed配置文件不存在: $DEEPSPEED_CONFIG"
    exit 1
fi

# 启动训练
deepspeed --include localhost:0 \
    training/complete_train.py \
    --config configs/food101_fast_eval.yaml \
    --deepspeed_config $DEEPSPEED_CONFIG

echo "✅ 快速评估训练完成" 