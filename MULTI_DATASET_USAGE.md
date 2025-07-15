# 多数据集训练与评估功能使用指南

本指南详细介绍如何使用新增的多数据集功能，包括数据准备、配置设置、训练和评估等步骤。

## 🎯 功能概览

### 主要新功能
1. **多数据集文件支持** - 从多个jsonl文件读取不同数据集
2. **数据集shuffle** - 将所有数据集混合并shuffle
3. **差异化评估比例** - 不同数据集使用不同的评估比例
4. **部分评估** - 训练过程中仅评估部分数据，提高训练效率
5. **完整评估** - 训练结束后对最佳模型进行完整评估
6. **最佳模型追踪** - 自动追踪并保存最佳模型
7. **分数据集指标** - 分别显示每个数据集和整体的loss、accuracy

## 📊 配置文件设置

### 完整配置示例 (`configs/multi_datasets_config.yaml`)

```yaml
model:
  pretrained_name: "/path/to/Qwen2.5-VL-7B-Instruct"
  num_labels: 2000  # 所有数据集中的最大类别数

# 多数据集配置
datasets:
  dataset_configs:
    food101:
      num_classes: 101
      description: "Food-101 dataset"
      eval_ratio: 0.2  # 训练过程中评估20%的数据
    food2k:
      num_classes: 2000
      description: "Food2K dataset"
      eval_ratio: 0.1  # 训练过程中评估10%的数据
    imagenet:
      num_classes: 1000
      description: "ImageNet dataset"
      eval_ratio: 0.2  # 训练过程中评估20%的数据
  enable_logits_masking: true
  shuffle_datasets: true

data:
  # 多个训练数据文件
  train_jsonl_list:
    - "/data/food101/train.jsonl"
    - "/data/food2k/train.jsonl"
    - "/data/imagenet/train.jsonl"
  # 多个验证数据文件
  val_jsonl_list:
    - "/data/food101/test.jsonl"
    - "/data/food2k/test.jsonl"
    - "/data/imagenet/test.jsonl"

training:
  epochs: 5
  lr: 1e-5
  output_dir: "/output/multi_datasets_1e_5"
  
  # 最佳模型追踪配置
  best_model_tracking:
    enabled: true
    metric: "overall_accuracy"  # 可选: overall_accuracy, overall_loss, food101_accuracy等
    mode: "max"  # max 或 min
    save_best_only: true
  
  # 评估配置
  evaluation:
    partial_eval_during_training: true  # 训练时部分评估
    full_eval_at_end: true             # 结束时完整评估
    eval_best_model_only: true         # 只对最佳模型完整评估

wandb:
  enabled: true
  project: "qwen_multi_datasets"
  log_dataset_metrics: true  # 记录分数据集指标
  log_overall_metrics: true  # 记录整体指标
```

## 📁 数据格式

### 数据文件格式

每个jsonl文件包含数据集名称信息：

```json
{"image_path": "/data/food101/image1.jpg", "label": 0, "dataset_name": "food101"}
{"image_path": "/data/food101/image2.jpg", "label": 1, "dataset_name": "food101"}
{"image_path": "/data/food2k/image1.jpg", "label": 0, "dataset_name": "food2k"}
{"image_path": "/data/food2k/image2.jpg", "label": 500, "dataset_name": "food2k"}
```

### 数据准备脚本示例

```python
import json
import os

def prepare_multi_dataset_file(dataset_configs):
    """
    准备多数据集文件
    
    Args:
        dataset_configs: {
            'food101': {'path': '/data/food101', 'num_classes': 101},
            'food2k': {'path': '/data/food2k', 'num_classes': 2000},
        }
    """
    
    for dataset_name, config in dataset_configs.items():
        data_path = config['path']
        
        # 训练数据
        train_data = []
        for label in range(config['num_classes']):
            label_dir = os.path.join(data_path, 'train', str(label))
            if os.path.exists(label_dir):
                for img_file in os.listdir(label_dir):
                    if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                        train_data.append({
                            "image_path": os.path.join(label_dir, img_file),
                            "label": label,
                            "dataset_name": dataset_name
                        })
        
        # 保存训练数据
        with open(f"{dataset_name}_train.jsonl", "w") as f:
            for item in train_data:
                f.write(json.dumps(item) + "\n")
        
        print(f"✅ {dataset_name} 训练数据: {len(train_data)} 样本")

# 使用示例
dataset_configs = {
    'food101': {'path': '/data/food101', 'num_classes': 101},
    'food2k': {'path': '/data/food2k', 'num_classes': 2000},
}

prepare_multi_dataset_file(dataset_configs)
```

## 🚀 训练命令

### 基本训练命令

```bash
# 多数据集训练
python training/train.py \
    --config configs/multi_datasets_config.yaml \
    --deepspeed_config configs/ds_s2.json
```

### 分布式训练

```bash
# 4卡训练
deepspeed --num_gpus=4 training/train.py \
    --config configs/multi_datasets_config.yaml \
    --deepspeed_config configs/ds_s2.json
```

## 📊 训练输出示例

### 数据加载阶段

```
📊 多数据集加载统计 (训练模式):
  • food101: 75,750 samples (37.9%)
  • food2k: 100,000 samples (50.0%)
  • imagenet: 24,300 samples (12.1%)
📊 总计: 200,050 samples
🔀 数据已shuffle

📊 多数据集加载统计 (评估模式):
  • food101: 25,250 → 5,050 (20.0%) samples (50.5%)
  • food2k: 25,000 → 2,500 (10.0%) samples (25.0%)
  • imagenet: 10,000 → 2,000 (20.0%) samples (20.0%)
📊 部分评估后总计: 9,550 samples
```

### 训练过程日志

```
📊 TRAIN - food101: Loss=0.5234, Acc=0.8567 (85.67%), Samples=1024
📊 TRAIN - food2k: Loss=0.6123, Acc=0.7890 (78.90%), Samples=1280
📊 TRAIN - imagenet: Loss=0.7456, Acc=0.7123 (71.23%), Samples=512
📊 TRAIN - OVERALL: Acc=0.7893 (78.93%), Samples=2816

🏆 发现更好模型! overall_accuracy: 0.8234 (步骤 1000)
```

### 评估结果

```
================================ 多数据集评估结果 ================================
📈 Overall Loss:     0.6234
🎯 Overall Accuracy: 0.7890 (78.90%)
📊 Total Samples:    9,550
✅ Total Correct:    7,535

📂 food101:
  • Loss:     0.5234
  • Accuracy: 0.8567 (85.67%)
  • Samples:  5,050 (Correct: 4,326)

📂 food2k:
  • Loss:     0.6123
  • Accuracy: 0.7890 (78.90%)
  • Samples:  2,500 (Correct: 1,973)

📂 imagenet:
  • Loss:     0.7456
  • Accuracy: 0.7123 (71.23%)
  • Samples:  2,000 (Correct: 1,425)
================================================================================
```

### 完整评估结果

```
🔍 开始对最佳模型进行完整评估
================================================================================

📊 多数据集加载统计 (评估模式):
  • food101: 25,250 samples (50.5%)
  • food2k: 25,000 samples (50.0%)
  • imagenet: 10,000 samples (20.0%)
📊 总计: 60,250 samples

🎯 最佳模型完整评估结果:
   • 整体准确率: 0.8234 (82.34%)
   • 总样本数: 60,250
   • 正确样本数: 49,609
```

## 📈 WandB监控

### 记录的指标类型

1. **训练指标**
   ```
   train_food101_loss, train_food101_accuracy
   train_food2k_loss, train_food2k_accuracy
   train_imagenet_loss, train_imagenet_accuracy
   train_overall_accuracy, train_overall_samples
   ```

2. **评估指标**
   ```
   eval_food101_loss, eval_food101_accuracy
   eval_food2k_loss, eval_food2k_accuracy
   eval_imagenet_loss, eval_imagenet_accuracy
   eval_overall_accuracy, eval_overall_samples
   ```

3. **最佳模型指标**
   ```
   best_model_step, best_overall_accuracy
   ```

4. **完整评估指标**
   ```
   final_eval_food101_accuracy, final_eval_food2k_accuracy
   final_eval_overall_accuracy, final_eval_overall_samples
   ```

## 🔧 高级配置

### 自定义评估比例

```yaml
datasets:
  dataset_configs:
    large_dataset:
      num_classes: 10000
      eval_ratio: 0.05  # 大数据集只评估5%
    small_dataset:
      num_classes: 10
      eval_ratio: 1.0   # 小数据集评估全部
```

### 自定义最佳模型指标

```yaml
training:
  best_model_tracking:
    enabled: true
    metric: "food101_accuracy"  # 追踪特定数据集的accuracy
    mode: "max"
```

### 禁用部分功能

```yaml
datasets:
  enable_logits_masking: false  # 禁用logits masking
  shuffle_datasets: false       # 禁用数据shuffle

training:
  evaluation:
    partial_eval_during_training: false  # 训练时不进行评估
    full_eval_at_end: false             # 不进行完整评估

wandb:
  log_dataset_metrics: false  # 不记录分数据集指标
```

## 🚨 注意事项

### 1. 内存和计算资源

- **分类头大小**: `num_labels` 设置为所有数据集的最大类别数
- **数据加载**: 多数据集会增加内存使用
- **评估时间**: 完整评估需要更多时间

### 2. 数据平衡

```yaml
# 推荐: 在配置中平衡不同数据集的权重
datasets:
  dataset_configs:
    large_dataset:
      eval_ratio: 0.1   # 大数据集少评估
    small_dataset:
      eval_ratio: 0.5   # 小数据集多评估
```

### 3. 标签一致性

- 确保每个数据集的标签都从0开始
- 检查标签范围不超过对应的 `num_classes`

### 4. 文件路径

- 使用绝对路径避免路径问题
- 确保所有节点都能访问数据文件

## 🔍 故障排除

### 常见问题

1. **数据加载错误**
   ```
   ⚠️ 跳过不存在的文件: /path/to/missing/file.jsonl
   ```
   **解决**: 检查文件路径是否正确

2. **标签超范围**
   ```
   RuntimeError: Target 1001 is out of bounds for num_classes=1000
   ```
   **解决**: 增加 `model.num_labels` 或检查数据标签

3. **内存不足**
   ```
   CUDA out of memory
   ```
   **解决**: 减少 `eval_ratio` 或 `batch_size`

### 调试技巧

1. **检查数据分布**
   ```python
   # 在训练开始前添加数据统计
   from collections import Counter
   dataset_names = [item["dataset_name"] for item in dataset.data_list]
   print("数据集分布:", Counter(dataset_names))
   ```

2. **验证logits masking**
   ```python
   # 检查mask后的logits
   print("Mask前:", logits[0, :10])
   print("Mask后:", masked_logits[0, :10])
   ```

3. **监控最佳模型**
   ```python
   # 训练过程中检查最佳模型信息
   print(f"当前最佳: {trainer.best_metric_value:.4f} (步骤 {trainer.best_model_step})")
   ```

## 🎉 总结

新的多数据集功能提供了：

- ✅ **灵活的数据管理** - 支持多文件、多数据集
- ✅ **智能评估策略** - 部分评估 + 完整评估
- ✅ **最佳模型追踪** - 自动保存最优模型
- ✅ **详细的指标监控** - 分数据集 + 整体指标
- ✅ **完整的向后兼容** - 支持原有单数据集配置

使用这些功能可以显著提高多数据集训练的效率和可监控性！ 