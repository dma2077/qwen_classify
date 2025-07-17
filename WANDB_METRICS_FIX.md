# WandB指标记录修复

## 🚨 问题描述

用户反馈WandB中不会显示`perf`组和`training`组的一些指标，导致性能监控和训练监控不完整。

## 🔍 根本原因分析

经过代码检查发现，原先的`log_step`函数中有两个主要问题：

### 1. 记录频率过低
- **基础训练指标**：每100步记录一次（过于稀疏）
- **性能指标**：每200步记录一次（几乎看不到）
- **结果**：WandB界面中大部分训练过程没有数据点

### 2. batch_size获取错误
- 代码使用了不正确的batch_size，影响MFU和性能指标计算
- 没有正确从DeepSpeed配置中获取有效的batch_size

## 🔧 修复方案

### 1. 优化WandB记录频率

```python
# 修复前（记录频率过低）
should_log_basic = (step % 100 == 0)     # 每100步
should_log_detailed = (step % 200 == 0)  # 每200步

# 修复后（合理的记录频率）
should_log_training = (step % 10 == 0)   # 每10步记录训练指标
should_log_perf = (step % 20 == 0)       # 每20步记录性能指标
should_log_gpu = (step % 50 == 0)        # 每50步记录GPU使用情况
```

### 2. 正确的指标分组

**训练组 (`training/`)**:
- `training/loss` - 训练损失
- `training/lr` - 学习率
- `training/epoch` - 当前epoch
- `training/grad_norm` - 梯度范数

**性能组 (`perf/`)**:
- `perf/step_time` - 每步耗时
- `perf/steps_per_second` - 每秒步数
- `perf/mfu` - 模型FLOPs利用率
- `perf/mfu_percent` - MFU百分比
- `perf/tokens_per_second` - 每秒处理的tokens数
- `perf/samples_per_second` - 每秒处理的样本数
- `perf/actual_flops` - 实际FLOPs
- `perf/flops_per_second` - 每秒FLOPs
- `perf/gpu_memory_allocated_gb` - GPU已分配内存
- `perf/gpu_memory_utilization_percent` - GPU内存利用率

### 3. 修复batch_size获取

```python
def _get_effective_batch_size(self, config: Dict) -> int:
    """正确获取有效的batch size"""
    # 1. 优先从DeepSpeed配置获取train_batch_size
    # 2. 或者通过micro_batch × gradient_accumulation × world_size计算
    # 3. 最后使用默认值
```

## 📊 修复效果

### 记录频率对比

| 指标组 | 修复前 | 修复后 | 改善倍数 |
|--------|--------|--------|----------|
| `training/` | 每100步 | 每10步 | ✅ **10倍** |
| `perf/` | 每200步 | 每20步 | ✅ **10倍** |
| `GPU指标` | 无 | 每50步 | ✅ **新增** |

### 新增性能指标

1. **MFU相关**：
   - `perf/mfu` - 模型FLOPs利用率（0-1）
   - `perf/mfu_percent` - MFU百分比显示

2. **吞吐量指标**：
   - `perf/tokens_per_second` - token处理速度
   - `perf/samples_per_second` - 样本处理速度
   - `perf/steps_per_second` - 训练步数速度

3. **GPU监控**：
   - `perf/gpu_memory_allocated_gb` - GPU内存使用
   - `perf/gpu_memory_utilization_percent` - GPU利用率

4. **FLOPs监控**：
   - `perf/actual_flops` - 实际测量的FLOPs
   - `perf/flops_per_second` - FLOPs处理速度

## 🧪 验证方法

### 1. 检查WandB界面

运行训练后，在WandB界面应该看到：

**训练组 (`training/`)**：
- 每10步更新的loss、学习率等指标
- 连续的训练曲线，无明显空白

**性能组 (`perf/`)**：
- 每20步更新的性能指标
- MFU、吞吐量等关键性能数据
- GPU内存使用情况

### 2. 检查日志输出

训练开始时应该看到：
```
📊 TrainingMonitor初始化: batch_size=32
📊 从DeepSpeed配置获取batch_size: 32
```

### 3. 验证指标计算

- MFU值应该在合理范围内（通常0.1-0.5）
- 吞吐量指标应该与实际训练速度匹配
- GPU内存使用应该接近实际观察值

## 🚀 优势

### 1. 更好的训练监控
- **实时可见**：每10步更新，及时发现问题
- **全面覆盖**：训练、性能、GPU使用全方位监控
- **准确计算**：正确的batch_size确保指标准确性

### 2. 性能优化指导
- **MFU监控**：了解模型计算效率
- **吞吐量跟踪**：优化训练速度
- **资源监控**：合理使用GPU资源

### 3. 问题诊断能力
- **梯度监控**：及时发现梯度爆炸/消失
- **内存跟踪**：避免OOM错误
- **性能趋势**：识别性能瓶颈

这个修复确保了WandB中的`training/`和`perf/`组指标能够正常显示和更新，为训练过程提供全面的监控和分析能力。 