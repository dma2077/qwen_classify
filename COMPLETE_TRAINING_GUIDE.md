# 🚀 完整的Qwen2.5-VL图像分类训练指南

## 📋 概述

本指南提供完整的Qwen2.5-VL图像分类训练解决方案，包含：
- ✅ FlashAttention支持（自动降级到eager attention）
- ✅ DeepSpeed分布式训练
- ✅ WandB监控和日志
- ✅ 性能优化和监控
- ✅ 多数据集支持

## 🔧 环境准备

### 1. 安装依赖

```bash
# 基础依赖
pip install torch torchvision torchaudio
pip install transformers datasets accelerate
pip install deepspeed wandb

# FlashAttention（可选，如果安装失败会自动降级）
conda install -c conda-forge flash-attn
# 或者
pip install flash-attn==2.3.6 --no-build-isolation
```

### 2. 检查系统兼容性

```bash
python scripts/check_glibc_compatibility.py
```

## 🚀 开始训练

### 方法一：使用完整训练脚本（推荐）

```bash
# 单GPU训练
python training/complete_train.py \
    --config configs/complete_training_config.yaml \
    --deepspeed_config configs/ds_config_zero2.json

# 多GPU训练
deepspeed --num_gpus 8 \
    training/complete_train.py \
    --config configs/complete_training_config.yaml \
    --deepspeed_config configs/ds_config_zero2.json
```

### 方法二：使用启动脚本

```bash
# 修改脚本中的GPU数量
chmod +x scripts/run_complete_training.sh
./scripts/run_complete_training.sh
```

## 📊 监控和日志

### 1. WandB监控

训练过程中会自动记录以下指标：

**训练指标**：
- `training/loss` - 训练损失
- `training/lr` - 学习率
- `training/grad_norm` - 梯度范数
- `training/epoch` - 当前epoch

**评估指标**：
- `eval/overall_loss` - 整体验证损失
- `eval/overall_accuracy` - 整体验证准确率
- `eval/food101_accuracy` - Food101数据集准确率
- `best_overall_accuracy` - 最佳准确率
- `best_model_step` - 最佳模型步数

**性能指标**：
- `perf/mfu` - Model FLOPs Utilization
- `perf/step_time` - 每步训练时间
- `perf/samples_per_second` - 每秒处理样本数
- `perf/gpu_memory_gb` - GPU内存使用

### 2. 本地日志

训练日志保存在：
- `outputs/complete_training/logs/` - 训练日志
- `outputs/complete_training/checkpoints/` - 模型检查点

## ⚙️ 配置说明

### 主要配置项

```yaml
# 模型配置
model:
  pretrained_name: "Qwen/Qwen2.5-VL-7B-Instruct"
  num_labels: 101

# 训练配置
training:
  num_epochs: 10
  learning_rate: 1e-5
  batch_size: 8
  gradient_accumulation_steps: 4
  
  # 性能优化
  gradient_checkpointing: false  # 优先计算速度
  memory_efficient_attention: true  # 启用FlashAttention
  dataloader_num_workers: 16  # 数据加载worker数量

# 监控配置
monitor:
  all_freq:
    training_log_freq: 10    # 训练指标记录频率
    eval_log_freq: 50        # 评估频率
    perf_log_freq: 10        # 性能指标记录频率
```

### DeepSpeed配置

```json
{
  "train_batch_size": 256,
  "train_micro_batch_size_per_gpu": 8,
  "gradient_accumulation_steps": 4,
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "overlap_comm": true,
    "reduce_scatter": true,
    "contiguous_gradients": true
  },
  "bf16": {"enabled": true},
  "fp16": {"enabled": false}
}
```

## 🔍 故障排除

### 1. FlashAttention GLIBC问题

如果遇到 `GLIBC_2.32 not found` 错误：

```bash
# 方案1: 使用conda安装
conda install -c conda-forge flash-attn

# 方案2: 安装较旧版本
pip install flash-attn==2.3.6 --no-build-isolation

# 方案3: 使用eager attention（自动降级）
# 代码会自动检测并降级，无需手动操作
```

### 2. 内存不足

```yaml
# 在配置文件中调整
training:
  batch_size: 4  # 减小批次大小
  gradient_accumulation_steps: 8  # 增加梯度累积

# 在DeepSpeed配置中启用offload
"zero_optimization": {
  "stage": 2,
  "offload_optimizer": {"device": "cpu"},
  "offload_param": {"device": "cpu"}
}
```

### 3. 训练速度慢

```yaml
# 启用梯度检查点（会降低速度但节省内存）
training:
  gradient_checkpointing: true

# 增加数据加载worker数量
training:
  dataloader_num_workers: 32
```

## 📈 性能优化建议

### 1. 硬件配置

- **GPU**: A100/H100 推荐，V100 可用
- **CPU**: 多核CPU，推荐16+核心
- **内存**: 64GB+ 推荐
- **存储**: SSD推荐，提高数据加载速度

### 2. 软件配置

- **CUDA**: 11.8+ 推荐
- **PyTorch**: 2.0+ 推荐
- **Transformers**: 4.35+ 推荐
- **FlashAttention**: 2.0+ 推荐

### 3. 训练参数调优

```yaml
# 学习率调优
training:
  learning_rate: 1e-5  # 基础学习率
  warmup_steps: 100    # 预热步数

# 批次大小调优
training:
  batch_size: 8        # 根据GPU内存调整
  gradient_accumulation_steps: 4  # 保持总批次大小

# 评估频率调优
monitor:
  all_freq:
    eval_log_freq: 50  # 根据数据集大小调整
```

## 🎯 预期性能

### 训练速度
- **FlashAttention**: 1.5-2倍速度提升
- **DeepSpeed ZeRO-2**: 1.2-1.5倍速度提升
- **bf16混合精度**: 1.3-1.8倍速度提升

### 内存使用
- **FlashAttention**: 减少10-20%内存使用
- **DeepSpeed ZeRO-2**: 减少50-70%内存使用
- **梯度检查点**: 减少30-50%内存使用（但会降低速度）

### 准确率
- **Food101**: 预期85-90%准确率
- **训练时间**: 8GPU A100约2-4小时

## 📝 注意事项

1. **FlashAttention**: 如果安装失败会自动降级到eager attention
2. **内存监控**: 密切关注GPU内存使用，避免OOM
3. **检查点保存**: 定期保存检查点，防止训练中断
4. **WandB**: 确保WandB配置正确，避免日志丢失
5. **数据路径**: 确保数据文件路径正确

## 🔄 恢复训练

```bash
# 从检查点恢复训练
python training/complete_train.py \
    --config configs/complete_training_config.yaml \
    --deepspeed_config configs/ds_config_zero2.json \
    --resume_from outputs/complete_training/checkpoints/checkpoint-1000
```

## 📞 技术支持

如果遇到问题，请检查：
1. 系统兼容性检查结果
2. 训练日志中的错误信息
3. WandB中的性能指标
4. GPU内存使用情况 