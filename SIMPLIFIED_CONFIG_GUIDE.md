# 简化配置流程指南

## 概述

重构后的代码将DeepSpeed配置从YAML文件中移除，改为通过命令行参数传入。这样可以简化配置管理，避免配置冲突。

## 主要变更

### 1. YAML配置文件 (`configs/food101_cosine_hold.yaml`)

**移除的内容：**
```yaml
# 移除了这部分
deepspeed:
  config_file: "configs/ds_minimal.json"
  zero_stage: 2
```

**保留的内容：**
- 模型配置
- 数据配置
- 训练配置
- WandB配置

### 2. 命令行参数

`complete_train.py` 现在需要两个必需参数：

```bash
python training/complete_train.py \
    --config configs/food101_cosine_hold.yaml \
    --deepspeed_config configs/ds_s2.json
```

### 3. 配置流程

1. **加载YAML配置**：只包含模型、数据、训练等配置
2. **验证DeepSpeed配置文件**：检查文件是否存在
3. **添加DeepSpeed配置**：将配置文件路径添加到config中
4. **初始化DeepSpeed**：使用指定的配置文件初始化

## 使用方法

### 1. 测试配置流程

```bash
python test_simplified_config.py
```

### 2. 运行训练

```bash
# 使用简化脚本
bash scripts/run_simple.sh

# 或直接使用deepspeed命令
deepspeed --num_gpus 8 --master_port 29500 \
    training/complete_train.py \
    --config configs/food101_cosine_hold.yaml \
    --deepspeed_config configs/ds_s2.json \
    --seed 42
```

## 优势

1. **配置分离**：训练配置和DeepSpeed配置分开管理
2. **避免冲突**：不会出现YAML和命令行配置冲突的问题
3. **更清晰**：配置来源更明确
4. **更灵活**：可以轻松切换不同的DeepSpeed配置

## 文件结构

```
configs/
├── food101_cosine_hold.yaml    # 训练配置（不含DeepSpeed）
├── ds_s2.json                  # DeepSpeed配置
└── ds_minimal.json             # 其他DeepSpeed配置

scripts/
├── run_simple.sh               # 简化启动脚本
└── run_complete.sh             # 完整启动脚本

training/
├── complete_train.py           # 主训练脚本
├── deepspeed_trainer.py        # DeepSpeed训练器
└── utils/
    └── config_utils.py         # 配置工具
```

## 注意事项

1. `--deepspeed_config` 参数现在是必需的
2. YAML文件中不再包含DeepSpeed相关配置
3. DeepSpeed配置文件必须包含 `train_batch_size` 和 `train_micro_batch_size_per_gpu` 字段 