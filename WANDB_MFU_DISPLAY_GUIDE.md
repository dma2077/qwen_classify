# WandB中MFU和FLOPs指标显示指南

## 问题描述

用户反映在WandB界面中没有看到MFU和FLOPs等性能指标。

## 原因分析

### 1. 代码问题
- **变量名错误**：`log_step`方法中使用了未定义的`current_flops`变量
- **MFU计算频率**：只有在`step % flops_profile_freq == 0`时才计算MFU，其他步骤MFU为0
- **指标分组**：MFU和FLOPs指标被记录在`perf/`组下

### 2. 配置问题
- **监控频率设置**：`perf_log_freq`可能设置得太高，导致指标记录不频繁
- **FLOPs测量频率**：`flops_profile_freq`设置不当，影响MFU计算
- **WandB配置**：没有明确指定要记录的指标

### 3. 显示问题
- **指标分组**：WandB界面中需要查看正确的分组
- **数据同步**：指标可能没有正确发送到WandB

## 解决方案

### 1. 代码修复

已修复的问题：
```python
# 修复前
"perf/actual_flops": float(current_flops),  # current_flops未定义

# 修复后
current_flops = self.actual_flops if self.actual_flops is not None else 0.0
"perf/actual_flops": float(current_flops),
```

### 2. 配置优化

使用专门的测试配置：
```yaml
monitor:
  freq:
    training_log_freq: 1        # 每步记录训练指标
    perf_log_freq: 1            # 每步记录性能指标
    flops_profile_freq: 5       # 每5步计算MFU
```

### 3. WandB指标配置

明确指定要记录的指标：
```yaml
wandb:
  metrics:
    performance:
      - "perf/mfu"
      - "perf/mfu_percent"
      - "perf/actual_flops"
      - "perf/flops_per_second"
      - "perf/tokens_per_second"
      - "perf/samples_per_second"
```

## 在WandB中查找指标

### 1. 性能指标组 (perf/)

在WandB界面中查找以下指标：

| 指标名称 | 说明 | 单位 |
|----------|------|------|
| `perf/mfu` | Model FLOPs Utilization | 0-1 |
| `perf/mfu_percent` | MFU百分比 | 0-100% |
| `perf/actual_flops` | 实际FLOPs数量 | FLOPs |
| `perf/flops_per_second` | 每秒FLOPs | FLOPs/s |
| `perf/tokens_per_second` | 每秒处理的token数 | tokens/s |
| `perf/samples_per_second` | 每秒处理的样本数 | samples/s |
| `perf/step_time` | 每步耗时 | 秒 |
| `perf/steps_per_second` | 每秒步数 | steps/s |

### 2. 训练指标组 (training/)

| 指标名称 | 说明 |
|----------|------|
| `training/loss` | 训练损失 |
| `training/learning_rate` | 学习率 |
| `training/grad_norm` | 梯度范数 |

### 3. 评估指标组 (eval/)

| 指标名称 | 说明 |
|----------|------|
| `eval/overall_loss` | 整体评估损失 |
| `eval/overall_accuracy` | 整体准确率 |

## 调试步骤

### 1. 检查指标记录

运行测试脚本：
```bash
python test_wandb_mfu_display.py
```

### 2. 检查WandB日志

查看WandB运行日志，确认指标是否成功发送：
```bash
# 在训练日志中查找
grep "perf/mfu" nohup.out
grep "perf/actual_flops" nohup.out
```

### 3. 验证配置

使用测试配置运行训练：
```bash
# 使用修复版本的训练脚本
./scripts/run_deepspeed_fixed.sh --config configs/config_mfu_wandb_test.yaml
```

## 常见问题及解决方案

### 问题1: 指标不显示

**症状**：在WandB界面中看不到任何性能指标

**解决方案**：
1. 检查WandB是否正常初始化
2. 验证指标名称是否正确
3. 确认指标值是否为数值类型
4. 检查WandB界面中的分组设置

### 问题2: MFU值为0

**症状**：MFU指标存在但值始终为0

**解决方案**：
1. 检查`flops_profile_freq`设置
2. 确认`actual_flops`是否正确设置
3. 验证模型引用是否正确
4. 检查profiler是否正常工作

### 问题3: 指标分组不正确

**症状**：指标没有按预期分组显示

**解决方案**：
1. 确保指标名称包含分组前缀 (`perf/`, `training/`, `eval/`)
2. 检查WandB界面中的分组设置
3. 重新启动WandB运行

### 问题4: 指标记录频率过低

**症状**：指标记录不频繁，图表显示不连续

**解决方案**：
1. 降低`perf_log_freq`设置
2. 降低`flops_profile_freq`设置
3. 确保频率设置合理

## 最佳实践

### 1. 配置建议

```yaml
# 开发/调试环境
monitor:
  freq:
    training_log_freq: 1        # 每步记录
    perf_log_freq: 1            # 每步记录
    flops_profile_freq: 5       # 每5步计算MFU

# 生产环境
monitor:
  freq:
    training_log_freq: 10       # 每10步记录
    perf_log_freq: 20           # 每20步记录
    flops_profile_freq: 100     # 每100步计算MFU
```

### 2. 监控建议

1. **实时监控**：使用`watch -n 1 nvidia-smi`监控GPU使用
2. **日志监控**：使用`tail -f nohup.out`监控训练日志
3. **WandB监控**：定期检查WandB界面中的指标更新

### 3. 性能优化

1. **MFU计算频率**：根据需求调整`flops_profile_freq`
2. **指标记录频率**：平衡精度和性能开销
3. **内存管理**：监控GPU内存使用情况

## 测试验证

### 1. 快速测试

```bash
# 运行MFU显示测试
python test_wandb_mfu_display.py
```

### 2. 完整训练测试

```bash
# 使用测试配置运行短训练
./scripts/run_deepspeed_fixed.sh --config configs/config_mfu_wandb_test.yaml
```

### 3. 验证清单

- [ ] WandB正常初始化
- [ ] 训练指标正常记录
- [ ] 性能指标正常记录
- [ ] MFU值不为0（在计算步骤）
- [ ] FLOPs指标正常显示
- [ ] 指标按分组正确显示

## 相关文件

- `training/utils/monitor.py`: 监控器实现
- `configs/config_mfu_wandb_test.yaml`: 测试配置
- `test_wandb_mfu_display.py`: 测试脚本
- `scripts/run_deepspeed_fixed.sh`: 修复版本训练脚本

## 联系支持

如果问题仍然存在，请提供：
1. 完整的训练日志
2. WandB运行链接
3. 配置文件内容
4. 系统环境信息 