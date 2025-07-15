# 学习率调度器使用指南

本指南详细介绍了系统中可用的各种学习率调度器及其配置方法。

## 📊 支持的调度器类型

### 1. **Cosine 余弦衰减调度器** ✨ 推荐
```yaml
lr_scheduler:
  type: "cosine"
  final_lr_ratio: 0.1    # 最终学习率为初始LR的10%
  num_cycles: 0.5        # 余弦周期数，默认0.5
```

**特点：**
- 🌊 平滑的余弦曲线衰减
- 🎯 在训练后期提供更细致的调整
- 📈 被广泛证明在深度学习中效果良好

**衰减倍数计算：**
- 默认配置（`final_lr_ratio: 0.0`）：衰减到接近0，衰减倍数为 **∞**
- `final_lr_ratio: 0.1`：衰减到原始LR的10%，衰减倍数为 **10x**
- `final_lr_ratio: 0.05`：衰减到原始LR的5%，衰减倍数为 **20x**

### 2. **Linear 线性衰减调度器**
```yaml
lr_scheduler:
  type: "linear"
  final_lr_ratio: 0.05   # 最终学习率为初始LR的5%
```

**特点：**
- 📉 简单的线性衰减
- ⚡ 训练前期快速衰减，适合快速收敛
- 🎯 可预测的学习率变化

**适用场景：**
- 训练时间较短的任务
- 需要快速收敛的场景

### 3. **Polynomial 多项式衰减调度器**
```yaml
lr_scheduler:
  type: "polynomial"
  final_lr_ratio: 0.1    # 最终学习率为初始LR的10%
  power: 2.0             # 多项式幂次，控制衰减曲线
```

**特点：**
- 📊 可控制的非线性衰减曲线
- 🔧 `power=1.0` 等同于线性衰减
- 📈 `power>1.0` 前期衰减慢，后期衰减快
- 📉 `power<1.0` 前期衰减快，后期衰减慢

### 4. **Exponential 指数衰减调度器**
```yaml
lr_scheduler:
  type: "exponential"
  decay_rate: 0.95       # 每步的衰减率
  # 或者指定最终比例
  final_lr_ratio: 0.01   # 系统会自动计算decay_rate
```

**特点：**
- 📊 指数衰减模式
- ⚡ 早期快速衰减
- 🎯 适合需要激进学习率衰减的场景

### 5. **Constant 常数调度器**
```yaml
lr_scheduler:
  type: "constant"
  # 无需额外参数
```

**特点：**
- 📊 Warmup后学习率保持不变
- 🛠️ 适合调试和对比实验
- ⚡ 简单直接，无需调参

### 6. **Cosine Restarts 带重启的余弦调度器** 🚀 高级
```yaml
lr_scheduler:
  type: "cosine_restarts"
  final_lr_ratio: 0.1         # 每个周期的最低学习率比例
  restart_period_epochs: 2     # 每2个epoch重启一次
```

**特点：**
- 🔄 周期性重启，防止陷入局部最小值
- 🎯 每个重启周期都有机会跳出局部最优
- 📈 适合长期训练和难以收敛的任务

## 📈 学习率衰减对比

### 当前默认Cosine调度器分析
**默认配置 (`num_cycles=0.5`, `final_lr_ratio=0.0`)：**
- ✅ 学习率从初始值平滑衰减到接近0
- ✅ 衰减倍数：**∞倍**（完全衰减）
- ✅ 适合大多数训练场景

### 推荐配置对比

| 调度器类型 | final_lr_ratio | 衰减倍数 | 适用场景 |
|-----------|---------------|---------|----------|
| cosine | 0.0 | ∞x | 标准训练 |
| cosine | 0.1 | 10x | 保守衰减 |
| cosine | 0.05 | 20x | 中等衰减 |
| linear | 0.05 | 20x | 快速收敛 |
| exponential | decay_rate=0.95 | ~148x* | 激进衰减 |
| constant | N/A | 1x | 调试对比 |

*基于5个epoch的计算示例

## 🔧 配置示例

### 保守衰减配置（推荐新手）
```yaml
training:
  lr: 1e-5
  lr_scheduler:
    type: "cosine"
    final_lr_ratio: 0.1  # 保留10%的学习率
    num_cycles: 0.5
```

### 标准衰减配置（推荐）
```yaml
training:
  lr: 1e-5
  lr_scheduler:
    type: "cosine"
    final_lr_ratio: 0.05  # 保留5%的学习率
    num_cycles: 0.5
```

### 激进衰减配置（高级用户）
```yaml
training:
  lr: 2e-5
  lr_scheduler:
    type: "linear"
    final_lr_ratio: 0.01  # 保留1%的学习率
```

### 长期训练配置（防止过拟合）
```yaml
training:
  epochs: 10
  lr: 2e-5
  lr_scheduler:
    type: "cosine_restarts"
    final_lr_ratio: 0.1
    restart_period_epochs: 2
```

## 📊 学习率曲线可视化

训练时会在控制台显示学习率配置信息：

```
📈 学习率调度器配置:
  • 调度器类型: cosine
  • Warmup步数: 200
  • 总训练步数: 5,000
  • 余弦周期数: 0.5
  • 最终学习率比例: 10.0%
  • 学习率衰减倍数: 10.0x
```

## 🎯 选择建议

### 🥇 **推荐优先级**

1. **Cosine (final_lr_ratio=0.1)** - 最平衡的选择
2. **Cosine (final_lr_ratio=0.05)** - 标准深度学习配置  
3. **Cosine Restarts** - 长期训练和困难任务
4. **Linear** - 快速实验和短期训练
5. **Polynomial** - 需要自定义衰减曲线
6. **Exponential** - 特殊需求场景

### 🔍 **根据任务特点选择**

- **图像分类**: Cosine (final_lr_ratio=0.05-0.1)
- **微调预训练模型**: Cosine (final_lr_ratio=0.1) 
- **从头训练**: Cosine Restarts 或 Linear
- **快速实验**: Linear 或 Constant
- **长期训练**: Cosine Restarts

## 🚀 高级技巧

### 动态调整
如果训练过程中发现学习率衰减过快或过慢，可以：
1. 调整 `final_lr_ratio` 参数
2. 更换调度器类型
3. 修改 `warmup_steps` 参数

### 监控指标
通过WandB观察：
- 学习率曲线变化
- 训练loss的响应
- 验证accuracy的变化趋势

### 实验对比
建议对比实验：
```bash
# 实验1：标准cosine
python training/train.py --config configs/multi_datasets_config.yaml

# 实验2：线性衰减
python training/train.py --config configs/multi_datasets_linear_scheduler.yaml

# 实验3：带重启的cosine
python training/train.py --config configs/multi_datasets_cosine_restarts.yaml
```

## ❓ 常见问题

**Q: 为什么cosine调度器效果更好？**
A: Cosine调度器提供了平滑的衰减曲线，在训练后期保持较小但非零的学习率，有助于精细调整。

**Q: 什么时候使用重启机制？**
A: 当训练容易陷入局部最优或需要长期训练时，重启机制可以提供跳出局部最优的机会。

**Q: 如何选择final_lr_ratio？**
A: 一般建议0.05-0.1，具体取决于任务复杂度和训练时长。复杂任务可以设置更大的值。

**Q: warmup_steps如何设置？**
A: 通常设置为总训练步数的5-10%，或者200-1000步之间。 