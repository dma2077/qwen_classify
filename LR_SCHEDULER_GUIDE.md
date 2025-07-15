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

### 2. **Cosine with Hold 余弦+平稳期调度器** 🚀 强烈推荐
```yaml
lr_scheduler:
  type: "cosine_with_hold"
  hold_ratio: 0.3        # 平稳期占非warmup步数的30%
  # 或者直接指定步数：
  # hold_steps: 1000     # 直接指定平稳期步数（优先级高于hold_ratio）
  final_lr_ratio: 0.05   # 最终学习率为初始LR的5%
  num_cycles: 0.5        # 余弦周期数
```

**学习率曲线：**
```
   LR
    │           ┌───────────┐
    │          /             \
    │───/─────/                \───
    │  /     ↑   hold_steps      \
    │ / warmup   (平稳期)         \  cosine_decay
    └┴────────────────────────────────→ step
     warmup    hold phase      decay phase
```

**特点：**
- 🎯 **三阶段设计**：Warmup → Hold → Cosine Decay
- 🚀 **更好的收敛**：高学习率平稳期让模型充分学习
- 📈 **稳定训练**：避免过早衰减导致的收敛问题
- 🛠️ **灵活配置**：可按比例或绝对步数设置平稳期

**适用场景：**
- 复杂任务和大模型训练
- 需要充分探索的训练任务
- 多数据集联合训练
- 从预训练模型微调

### 3. **Linear 线性衰减调度器**
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

### 4. **Polynomial 多项式衰减调度器**
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

### 5. **Exponential 指数衰减调度器**
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

### 6. **Constant 常数调度器**
```yaml
lr_scheduler:
  type: "constant"
  # 无需额外参数
```

**特点：**
- 📊 Warmup后学习率保持不变
- 🛠️ 适合调试和对比实验
- ⚡ 简单直接，无需调参

### 7. **Cosine Restarts 带重启的余弦调度器** 🚀 高级
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
| **cosine_with_hold** | **0.05** | **20x** | **复杂任务（推荐）** |
| cosine | 0.1 | 10x | 保守衰减 |
| cosine | 0.05 | 20x | 中等衰减 |
| linear | 0.05 | 20x | 快速收敛 |
| exponential | decay_rate=0.95 | ~148x* | 激进衰减 |
| constant | N/A | 1x | 调试对比 |

*基于5个epoch的计算示例

## 🔧 配置示例

### 🥇 **最推荐配置（余弦+平稳期）**
```yaml
training:
  lr: 1e-5
  lr_scheduler:
    type: "cosine_with_hold"
    hold_ratio: 0.3  # 30%时间保持高学习率
    final_lr_ratio: 0.05  # 最终衰减到5%
    num_cycles: 0.5
```

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

### Cosine with Hold 调度器输出示例：
```
📈 学习率调度器配置:
  • 调度器类型: cosine_with_hold
  • Warmup步数: 200
  • Hold平稳期步数: 1,200
  • Cosine衰减步数: 2,800
  • Hold比例: 30.0%
  • 余弦周期数: 0.5
  • 最终学习率比例: 5.0%
  • 学习率衰减倍数: 20.0x
```

## 🎯 选择建议

### 🥇 **推荐优先级**

1. **🚀 Cosine with Hold (hold_ratio=0.3, final_lr_ratio=0.05)** - **最佳选择**
2. **Cosine (final_lr_ratio=0.1)** - 最平衡的选择
3. **Cosine (final_lr_ratio=0.05)** - 标准深度学习配置  
4. **Cosine Restarts** - 长期训练和困难任务
5. **Linear** - 快速实验和短期训练
6. **Polynomial** - 需要自定义衰减曲线
7. **Exponential** - 特殊需求场景

### 🔍 **根据任务特点选择**

- **🎯 多数据集训练**: **Cosine with Hold (强烈推荐)**
- **图像分类**: Cosine with Hold 或 Cosine (final_lr_ratio=0.05-0.1)
- **微调预训练模型**: Cosine with Hold (hold_ratio=0.2-0.4)
- **从头训练**: Cosine with Hold 或 Cosine Restarts
- **快速实验**: Linear 或 Constant
- **长期训练**: Cosine with Hold 或 Cosine Restarts

### 💡 **Cosine with Hold 参数选择指南**

**hold_ratio 建议：**
- **0.2-0.3**: 标准配置，适合大多数任务
- **0.3-0.4**: 复杂任务，需要更多高学习率训练
- **0.1-0.2**: 简单任务，快速收敛

**final_lr_ratio 建议：**
- **0.05**: 标准配置，20倍衰减
- **0.1**: 保守配置，10倍衰减  
- **0.01**: 激进配置，100倍衰减

## 🚀 高级技巧

### 动态调整
如果训练过程中发现学习率衰减过快或过慢，可以：
1. 调整 `final_lr_ratio` 参数
2. 更换调度器类型
3. 修改 `warmup_steps` 参数
4. 调整 `hold_ratio` 或 `hold_steps`

### 监控指标
通过WandB观察：
- 学习率曲线变化
- 训练loss的响应
- 验证accuracy的变化趋势
- Hold期间的性能稳定性

### 实验对比
建议对比实验：
```bash
# 实验1：余弦+平稳期（推荐）
python training/train.py --config configs/multi_datasets_cosine_hold.yaml

# 实验2：标准cosine
python training/train.py --config configs/multi_datasets_config.yaml

# 实验3：线性衰减
python training/train.py --config configs/multi_datasets_linear_scheduler.yaml

# 实验4：带重启的cosine
python training/train.py --config configs/multi_datasets_cosine_restarts.yaml
```

## ❓ 常见问题

**Q: 为什么余弦+平稳期调度器效果更好？**
A: 平稳期让模型在高学习率下充分学习，避免过早衰减导致的收敛不充分问题。余弦衰减则在后期提供精细调整。

**Q: hold_ratio设置多少合适？**
A: 一般建议0.2-0.4，复杂任务可以设置更大的值。可以通过实验找到最佳比例。

**Q: 什么时候使用重启机制？**
A: 当训练容易陷入局部最优或需要长期训练时，重启机制可以提供跳出局部最优的机会。

**Q: 如何选择final_lr_ratio？**
A: 一般建议0.05-0.1，具体取决于任务复杂度和训练时长。复杂任务可以设置更大的值。

**Q: warmup_steps如何设置？**
A: 通常设置为总训练步数的5-10%，或者200-1000步之间。

**Q: Cosine with Hold 和普通Cosine的主要区别？**
A: 主要区别在于Hold期间学习率保持恒定，这样模型可以在高学习率下更充分地学习，通常能获得更好的最终性能。 