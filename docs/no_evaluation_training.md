# 无评估训练模式 - 快速训练与完整checkpoint保存

本文档介绍如何使用新增的无评估训练模式，该模式允许您跳过所有评估步骤，专注于训练，并保存所有的checkpoint而不是只保存最佳的。

## 🎯 适用场景

这种模式特别适合以下场景：

1. **快速原型开发**: 需要快速训练模型进行初步验证
2. **大规模训练**: 评估耗时较长，希望先完成训练再评估
3. **分布式训练**: 避免评估时的同步开销
4. **资源受限**: 减少内存和计算开销
5. **批量实验**: 需要保存每个检查点用于后续分析

## 🔧 配置参数

在配置文件的 `training` 部分添加以下参数：

### 核心参数

```yaml
training:
  # 跳过所有评估（训练中和结束时）
  skip_evaluation: true
  
  # 保存所有checkpoint，而不是只保存最佳的
  save_all_checkpoints: true
```

### 🔥 最高优先级覆盖机制

**重要：`skip_evaluation` 具有最高优先级！**

当 `skip_evaluation: true` 时，会**强制覆盖**所有相关配置，无论您在配置文件中如何设置：

```yaml
# 即使您在配置文件中设置了这些参数，也会被强制覆盖：
training:
  skip_evaluation: true           # 🔥 最高优先级参数
  
  # ❌ 以下参数会被强制覆盖，无论您如何设置：
  save_all_checkpoints: false    # -> 强制改为 true
  
  best_model_tracking:
    enabled: true                 # -> 强制改为 false
    save_best_only: true         # -> 强制改为 false
  
  evaluation:
    partial_eval_during_training: true   # -> 强制改为 false
    full_eval_at_end: true              # -> 强制改为 false
    eval_best_model_only: true          # -> 强制改为 false
```

**覆盖后的实际效果：**
- ✅ `best_model_enabled`: false (强制禁用)
- ✅ `save_best_only`: false (强制禁用)
- ✅ `save_all_checkpoints`: true (强制启用)
- ✅ `partial_eval_during_training`: false (强制禁用)
- ✅ `full_eval_at_end`: false (强制禁用)
- ✅ `eval_best_model_only`: false (强制禁用)
- ✅ 所有评估步骤都会被跳过
- ✅ 所有checkpoint都会被保存

## 📋 完整配置示例

参考配置文件：`configs/food101_no_eval_save_all.yaml`

```yaml
model:
  pretrained_name: "/path/to/Qwen2.5-VL-7B-Instruct"
  num_labels: 101

loss:
  type: "label_smoothing"
  smoothing: 0.1

datasets:
  dataset_configs:
    food101:
      num_classes: 101
      description: "Food-101 dataset"

data:
  train_jsonl: "/path/to/food101_train.jsonl"
  val_jsonl: "/path/to/food101_test.jsonl"

training:
  epochs: 5
  lr: 5e-6
  weight_decay: 0.01
  warmup_steps: 200
  output_dir: "/path/to/output"
  logging_steps: 50
  save_steps: 200
  eval_steps: 200  # 虽然设置了，但会被忽略
  
  # 🔥 关键配置
  skip_evaluation: true         # 跳过所有评估
  save_all_checkpoints: true    # 保存所有checkpoint

wandb:
  enabled: true
  project: "qwen_classify_no_eval"
  log_dataset_metrics: false  # 跳过评估时不需要
```

## 🚀 使用方法

### 1. 使用预定义脚本

```bash
# 直接运行无评估训练脚本
bash scripts/train_no_eval_save_all.sh
```

### 2. 手动运行

```bash
# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 运行训练
torchrun \
    --nproc_per_node=4 \
    --master_addr=localhost \
    --master_port=29500 \
    training/deepspeed_trainer.py \
    --config configs/food101_no_eval_save_all.yaml
```

## 📂 输出结构

训练完成后，输出目录将包含所有的checkpoint：

```
output_dir/
├── checkpoint-200/          # 第200步的checkpoint
│   ├── deepspeed/
│   ├── hf_model/
│   └── training_info.json
├── checkpoint-400/          # 第400步的checkpoint
├── checkpoint-600/
├── ...
└── checkpoint-{final}/      # 最终checkpoint
```

每个checkpoint包含：
- **DeepSpeed格式**: `deepspeed/` 目录
- **HuggingFace格式**: `hf_model/` 目录  
- **训练信息**: `training_info.json`

## 📊 训练日志

即使跳过评估，训练日志仍会记录：

- ✅ 训练损失和学习率
- ✅ 梯度范数和性能指标
- ✅ GPU使用率和内存统计
- ❌ 评估损失和准确率（被跳过）
- ❌ 最佳模型指标（被禁用）

## 🔄 训练后评估

训练完成后，您可以手动评估任何checkpoint：

```python
# 使用专门的评估脚本
python examples/evaluate_checkpoint.py \
    --checkpoint_path /path/to/checkpoint-800 \
    --config_path configs/food101_no_eval_save_all.yaml \
    --eval_data /path/to/test_data.jsonl
```

## ⚡ 性能优势

相比标准训练模式，无评估模式提供：

- **训练速度提升**: 20-40%（取决于评估频率）
- **内存使用减少**: 避免评估时的额外内存开销
- **分布式效率**: 减少进程间同步需求
- **存储完整性**: 保留所有训练状态用于分析

## 🔧 高级配置

### 灵活的checkpoint保存

如果只想保存所有checkpoint但仍要进行评估：

```yaml
training:
  skip_evaluation: false       # 保持评估
  save_all_checkpoints: true   # 但保存所有checkpoint
```

### 自定义保存频率

```yaml
training:
  save_steps: 100              # 每100步保存一次
  skip_evaluation: true
  save_all_checkpoints: true
```

## 🚨 注意事项

1. **磁盘空间**: 保存所有checkpoint需要更多存储空间
2. **模型质量**: 无法实时监控模型性能
3. **最佳时机**: 无法确定最佳停止点
4. **后处理**: 需要训练后手动选择最佳checkpoint

## 💡 最佳实践

1. **磁盘监控**: 确保有足够的存储空间
2. **定期检查**: 监控训练损失曲线
3. **分阶段训练**: 可以先短时间训练验证配置
4. **后评估策略**: 制定训练后的评估计划

## 🆚 模式对比

| 特性 | 标准模式 | 无评估模式 |
|------|----------|------------|
| 训练速度 | 中等 | 快 |
| 评估反馈 | 实时 | 无 |
| Checkpoint数量 | 最佳+少量 | 全部 |
| 资源使用 | 高 | 中等 |
| 适用场景 | 交互式开发 | 批量训练 |

选择适合您需求的模式，充分利用计算资源！ 