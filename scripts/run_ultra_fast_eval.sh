#!/bin/bash

# 超快速评估训练启动脚本
# 使用大批次大小和全量评估数据，大幅提升评估速度

echo "🚀 启动超快速评估训练..."
echo "📊 优化配置："
echo "  • 训练批次大小：64 x 4 GPU = 256"
echo "  • 评估批次大小：64 x 4 GPU = 256"
echo "  • 评估频率：每500步（大幅减少）"
echo "  • 评估数据：全量数据（100%）"
echo "  • 进度条更新：每100批次"
echo ""

# 检查flash-attn是否可用
python -c "import flash_attn; print('✅ FlashAttention可用')" 2>/dev/null || {
    echo "⚠️  FlashAttention不可用，将使用默认注意力机制"
    export FLASH_ATTENTION_FORCE_ENABLE=0
    export FLASH_ATTENTION_2=0
}

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# 启动训练
deepspeed --num_gpus=4 \
    training/complete_train.py \
    --config configs/ultra_fast_eval_config.yaml \
    --deepspeed_config configs/ds_large_eval_batch.json

echo "✅ 超快速评估训练完成" 