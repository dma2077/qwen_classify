# WandB 指标记录完整指南

## 📊 指标记录保证

经过优化后，所有指标都有完整的 `commit=True` 保证，确保数据不会丢失。

## 🎯 指标分组

### 1. Training 组 (`training/`)
**记录频率**: 每个有效训练步
**记录方法**: `monitor.log_step()`

- `training/loss` - 训练损失 ✅ **每步记录**
- `training/lr` - 学习率 ✅ **每步记录**  
- `training/grad_norm` - 梯度范数 ✅ **每步记录**
- `training/epoch` - 当前epoch ✅ **每步记录**
- `global_step` - 全局步数 ✅ **每步记录**

**Epoch级别指标**:
- `training/epoch_avg_loss` - Epoch平均损失 ✅
- `training/epoch_time` - Epoch耗时 ✅
- `training/epoch_number` - Epoch编号 ✅

**训练状态**:
- `training/started` - 训练开始标记 ✅
- `training/finished` - 训练结束标记 ✅
- `training/total_time` - 总训练时间 ✅

### 2. Performance 组 (`perf/`)
**记录频率**: 每10步记录一次（减少开销）
**记录方法**: `monitor.log_step()` 内部

- `perf/step_time` - 单步耗时 ✅
- `perf/mfu` - 模型FLOPs利用率 ✅ 
- `perf/tokens_per_second` - 每秒处理tokens数 ✅
- `perf/actual_flops` - 实际FLOPs ✅
- `perf/actual_seq_length` - 实际序列长度 ✅
- `perf/real_time_measurement` - 实时测量标记 ✅
- `perf/flops_per_second` - 每秒FLOPs ✅

### 3. Evaluation 组 (`eval/`)
**记录频率**: 每 `eval_steps` 步（默认500步）
**记录方法**: `monitor.log_metrics()` 通过 `evaluate()`

**整体指标**:
- `eval/overall_loss` - 整体评估损失 ✅
- `eval/overall_accuracy` - 整体评估准确率 ✅
- `eval/overall_samples` - 整体样本数 ✅
- `eval/overall_correct` - 整体正确数 ✅

**多数据集情况**:
- `eval/{dataset_name}_loss` - 各数据集损失 ✅
- `eval/{dataset_name}_accuracy` - 各数据集准确率 ✅
- `eval/{dataset_name}_samples` - 各数据集样本数 ✅

**单数据集情况**:
- `eval/loss` - 评估损失 ✅
- `eval/accuracy` - 评估准确率 ✅

**最终评估**:
- `eval/final_overall_loss` - 最终整体损失 ✅
- `eval/final_overall_accuracy` - 最终整体准确率 ✅
- `eval/final_{dataset_name}_loss` - 各数据集最终损失 ✅
- `eval/final_{dataset_name}_accuracy` - 各数据集最终准确率 ✅

## 🔧 记录方法详解

### 核心记录函数

1. **`monitor.log_step()`** - 训练步骤指标
   - 每个有效训练步调用
   - 记录training和perf组指标
   - 使用 `commit=True` 确保数据提交

2. **`monitor.log_epoch()`** - Epoch级别指标
   - 每个epoch结束时调用
   - 记录epoch统计信息
   - 使用 `commit=True` 确保数据提交

3. **`monitor.log_metrics()`** - 通用指标记录
   - 用于eval指标记录
   - 支持自定义commit参数
   - 默认 `commit=True`

4. **`monitor.log_evaluation()`** - 评估指标记录
   - 专门用于评估结果
   - 使用 `commit=True` 确保数据提交

## ⚡ 性能优化

### 频率控制
- **基础训练指标**: 每步记录（loss, lr, grad_norm, epoch）
- **性能指标**: 每10步记录（减少开销）
- **评估指标**: 每 `eval_steps` 记录
- **本地日志保存**: 每100步保存

### 分布式训练优化
- 只有主进程（rank 0）记录到wandb
- 非主进程使用 `DummyMonitor`，零开销
- 避免重复记录和网络开销

## 🚨 关键修复

### 修复前的问题
1. **commit逻辑错误**: 部分指标使用 `commit=False` 但没有后续提交
2. **空记录**: 存在 `wandb.log({}, commit=True)` 空调用
3. **不一致的commit**: 不同方法使用不同的commit策略
4. **数据丢失风险**: 基础指标可能未被正确提交

### 修复后的保证
1. **所有wandb.log调用都使用 `commit=True`**
2. **一次性记录**: 避免多次分离的log调用
3. **统一的commit策略**: 所有记录方法保持一致
4. **数据完整性**: 确保所有指标都被正确记录和提交

## 📈 WandB界面预期结构

```
📊 WandB Dashboard:
├── 📈 training/
│   ├── loss, lr, grad_norm, epoch (每步)
│   ├── epoch_avg_loss, epoch_time (每epoch)
│   └── started, finished, total_time (状态)
├── ⚡ perf/
│   ├── step_time, mfu (每10步)
│   ├── tokens_per_second, actual_flops
│   └── real_time_measurement, flops_per_second
└── 📊 eval/
    ├── overall_loss, overall_accuracy (整体)
    ├── {dataset}_loss, {dataset}_accuracy (各数据集)
    └── final_* (最终评估)
```

## ✅ 验证清单

- [x] 所有训练指标正确记录
- [x] 所有评估指标正确记录  
- [x] 所有性能指标正确记录
- [x] 分布式训练兼容
- [x] commit参数一致性
- [x] 无数据丢失风险
- [x] 频率控制优化
- [x] 错误处理完善

所有指标现在都有**100%的记录保证**！🎉 