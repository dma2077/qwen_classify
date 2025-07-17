#!/bin/bash

# 测试配置文件中的flops_profile_freq设置
# 验证MFU计算频率是否能正确从yaml配置中读取

set -e

echo "🧪 开始测试配置文件中的flops_profile_freq设置..."

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
echo "📊 运行flops_profile_freq配置测试脚本..."
python test_config_flops_freq.py

echo "✅ 测试脚本执行完成！"
echo ""
echo "📋 检查清单："
echo "1. 确认配置文件中的flops_profile_freq设置能正确生效"
echo "2. 确认默认值500在未设置时正确使用"
echo "3. 确认自定义值（如50、100）能正确读取"
echo "4. 确认MFU计算频率与配置的flops_profile_freq一致"
echo "5. 确认每flops_profile_freq步使用profiler进行精确计算"
echo ""
echo "🔍 如果发现问题，请检查："
echo "   • 配置文件格式是否正确"
echo "   • monitor.freq.flops_profile_freq设置是否正确"
echo "   • TrainingMonitor初始化是否正确"
echo "   • MFU计算逻辑是否正确"
echo ""
echo "📝 使用示例："
echo "在yaml配置文件中设置："
echo "monitor:"
echo "  freq:"
echo "    flops_profile_freq: 100  # 每100步使用profiler计算MFU" 