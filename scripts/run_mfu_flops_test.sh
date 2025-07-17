#!/bin/bash

# 测试MFU和FLOPs指标在WandB中的显示
# 验证性能指标是否能正确记录和显示

set -e

echo "🧪 开始测试MFU和FLOPs指标显示..."

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
echo "📊 运行MFU和FLOPs指标测试脚本..."
python test_mfu_flops_display.py

echo "✅ 测试脚本执行完成！"
echo ""
echo "📋 检查清单："
echo "1. 打开WandB界面，查看项目 'qwen-classify-test'"
echo "2. 找到运行 'mfu_flops_display_test'"
echo "3. 确认以下性能指标是否正常显示："
echo "   • perf/mfu - Model FLOPs Utilization"
echo "   • perf/mfu_percent - MFU百分比"
echo "   • perf/actual_flops - 实际FLOPs"
echo "   • perf/flops_per_second - 每秒FLOPs"
echo "   • perf/tokens_per_second - 每秒token数"
echo "   • perf/samples_per_second - 每秒样本数"
echo "   • perf/step_time - 步骤时间"
echo "   • perf/steps_per_second - 每秒步数"
echo "   • perf/actual_seq_length - 实际序列长度"
echo "4. 确认MFU值不是0，而是有实际的计算值"
echo "5. 确认FLOPs相关指标都有合理的数值"
echo ""
echo "🔍 如果发现问题，请检查："
echo "   • WandB API连接是否正常"
echo "   • 性能指标定义是否正确"
echo "   • MFU计算是否成功"
echo "   • FLOPs数据是否正确"
echo "   • 监控频率配置是否合适" 