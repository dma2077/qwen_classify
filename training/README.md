# Training Module

## 目录结构

```
training/
├── __init__.py                 # 模块入口
├── deepspeed_trainer.py        # 核心训练器
├── train.py                    # 主训练脚本
├── utils/                      # 工具包
│   ├── __init__.py
│   ├── model_utils.py          # 模型相关工具
│   ├── distributed.py          # 分布式训练工具
│   ├── monitor.py              # 训练监控工具
│   ├── evaluation.py           # 模型评估工具
│   └── config_utils.py         # 配置处理工具
├── lr_scheduler.py             # 学习率调度器
└── model.py                    # 模型包装器
```

## 主要组件

### DeepSpeedTrainer
核心训练器，负责训练流程的管理和控制。

### 工具模块

#### model_utils.py
- `save_hf_model()`: 保存HuggingFace格式的模型

#### distributed.py
- `DistributedContext`: 分布式训练上下文管理器

#### monitor.py
- `TrainingMonitor`: 训练过程监控和日志记录

#### evaluation.py
- `evaluate_model()`: 模型评估函数

#### config_utils.py
- `prepare_config()`: 配置参数准备和映射函数

## 使用方法

```python
from training import DeepSpeedTrainer

# 创建训练器
trainer = DeepSpeedTrainer(config)

# 设置模型和数据
trainer.setup_model(model, train_loader, val_loader, optimizer, lr_scheduler)

# 开始训练
trainer.train()
```

## 设计原则

1. **单一职责**: 每个模块只负责一个特定的功能
2. **高内聚低耦合**: 相关功能组织在一起，模块间依赖最小化
3. **可复用性**: 工具函数可以独立使用
4. **简洁性**: 核心训练器只包含训练逻辑，辅助功能分离到工具模块
5. **职责分离**: 配置处理、模型评估、监控等功能各自独立 