#!/bin/bash

# 全面测试所有指标在WandB中的显示
# 验证training、eval、perf等所有指标组是否正常显示

set -e

echo "🧪 开始全面测试所有指标显示..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# 清理之前的进程
echo "🧹 清理之前的进程..."
pkill -f "deepspeed" || true
pkill -f "python.*train.py" || true
sleep 2

# 运行测试脚本
echo "📊 运行指标显示测试脚本..."
python test_all_metrics_display.py

echo "✅ 测试脚本执行完成！"
echo ""
echo "📋 检查清单："
echo "1. 打开WandB界面，查看项目 'qwen-classify-test'"
echo "2. 找到运行 'all_metrics_display_test'"
echo "3. 确认以下指标组是否正常显示："
echo "   • training/* - 训练指标（loss, lr, epoch, grad_norm）"
echo "   • eval/* - 评估指标（overall_loss, overall_accuracy）"
echo "   • perf/* - 性能指标（mfu, step_time, gpu_memory等）"
echo "4. 确认所有指标都有统一的'step'作为x轴"
echo "5. 确认图表正常显示，没有数据缺失"
echo ""
echo "🔍 如果发现问题，请检查："
echo "   • WandB API连接是否正常"
echo "   • 指标定义是否正确"
echo "   • 数据记录频率是否合适"
echo "   • 是否有step冲突或重复记录" 