#!/bin/bash

# 完整指标显示测试脚本

echo "🚀 开始完整指标显示测试"
echo "=" * 60

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="qwen_classify_test"

# 运行简单测试（快速验证）
echo "📊 步骤1: 运行简单指标测试..."
python test_all_metrics_display.py

echo ""
echo "=" * 60
echo "📊 步骤2: 运行正式训练测试（使用修复后的配置）..."
echo "使用配置文件: configs/config_all_metrics_test.yaml"

# 运行正式训练测试
python training/train.py \
    --config configs/config_all_metrics_test.yaml \
    --deepspeed_config configs/ds_s2.json

echo ""
echo "=" * 60
echo "✅ 测试完成！"
echo "📊 请在WandB界面中检查以下指标组是否正常显示："
echo "   • training/* - 训练指标（loss, lr, epoch, grad_norm）"
echo "   • perf/* - 性能指标（step_time, tokens_per_second, gpu_memory等）"
echo "   • eval/* - 评估指标（overall_loss, overall_accuracy）"
echo "   • system/* - 系统指标（自动生成）"
echo ""
echo "🔗 如果某些指标组仍然不显示，请检查："
echo "   1. 监控频率配置是否合适（all_freq建议设置为1-20）"
echo "   2. 训练步数是否足够（至少需要几十步）"
echo "   3. WandB界面是否需要刷新"
echo ""
echo "💡 调试建议："
echo "   • 查看训练日志中的'监控频率配置'输出"
echo "   • 确认'已记录X个指标'的消息"
echo "   • 检查WandB run的URL是否正确" 