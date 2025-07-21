#!/bin/bash

# 测试WandB修复的训练脚本

# 🔥 修复NCCL警告：首先设置NCCL_NTHREADS
export NCCL_NTHREADS=64
echo "🔧 设置 NCCL_NTHREADS=$NCCL_NTHREADS"

echo "🧪 测试WandB指标记录修复..."
echo "📊 配置:"
echo "  • 每步都记录training指标"
echo "  • 每5步评估一次"
echo "  • 每步都记录perf指标"
echo "  • 使用简化的DeepSpeed配置"
echo ""

# 设置环境变量
export MASTER_PORT=29501
export MASTER_ADDR=localhost

# 设置代理（如果需要）
export http_proxy=http://oversea-squid1.jp.txyun:11080 
export https_proxy=http://oversea-squid1.jp.txyun:11080

# 登录WandB
wandb login f3b76ea66a38b2a211dc706fa95b02c761994b73

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 创建输出目录
mkdir -p test_output

echo "🚀 启动测试训练..."
echo "📋 配置文件: configs/test_wandb_fix.yaml"
echo "⚙️  DeepSpeed配置: configs/ds_test_wandb.json"
echo ""

# 启动训练（使用单GPU进行测试）
deepspeed --num_gpus=1 \
    training/train.py \
    --config configs/test_wandb_fix.yaml \
    --deepspeed_config configs/ds_test_wandb.json

echo ""
echo "✅ 测试训练完成！"
echo ""
echo "📊 请在WandB界面中检查以下内容:"
echo "  1. ✅ training/* 指标是否连续显示（每步都有）"
echo "  2. ✅ perf/* 指标是否正常显示"
echo "  3. ✅ eval/* 指标是否在评估步骤时显示"
echo "  4. ✅ 所有指标是否使用统一的step轴"
echo "  5. ✅ 没有step冲突或重复记录"
echo ""
echo "🔍 如果仍有问题，请检查:"
echo "  • WandB项目: test_wandb_fix"
echo "  • 运行名称: test_wandb_run"
echo "  • 日志文件: test_output/training_log.json" 