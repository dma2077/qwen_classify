# DeepSpeed卡住问题解决方案

## 问题描述

训练脚本在DeepSpeed初始化阶段卡住，通常卡在以下位置：
```
nohup: ignoring input
wandb: Appending key for api.wandb.ai to your netrc file: /root/.netrc
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
🔥 启动多GPU分布式训练...
[2025-07-18 01:45:16,013] [INFO] [real_accelerator.py:254:get_accelerator] Setting ds_accelerator to cuda (auto detect)
```

## 常见原因

### 1. NCCL通信问题
- 网络接口配置错误
- InfiniBand配置问题
- 端口冲突

### 2. 环境变量冲突
- 分布式训练环境变量设置错误
- NCCL配置冲突

### 3. GPU资源问题
- GPU内存不足
- GPU被其他进程占用

### 4. 进程残留
- 之前的训练进程未完全清理
- 端口被占用

## 解决方案

### 方案1: 使用修复版本脚本

```bash
# 使用修复版本的训练脚本
chmod +x scripts/run_deepspeed_fixed.sh
./scripts/run_deepspeed_fixed.sh
```

修复版本的特点：
- 设置NCCL环境变量
- 清理残留进程
- 检查GPU状态
- 使用本地回环接口

### 方案2: 使用简化版本脚本

```bash
# 使用简化版本的训练脚本
chmod +x scripts/run_deepspeed_simple.sh
./scripts/run_deepspeed_simple.sh
```

简化版本的特点：
- 最小化环境变量
- 禁用InfiniBand
- 启用NCCL调试

### 方案3: 手动调试

```bash
# 运行调试脚本
python scripts/debug_deepspeed_init.py
```

调试脚本会检查：
- CUDA环境
- GPU状态
- NCCL通信
- 网络连通性

## 环境变量说明

### 关键环境变量

```bash
# NCCL配置
export NCCL_DEBUG=INFO          # 启用NCCL调试信息
export NCCL_IB_DISABLE=1        # 禁用InfiniBand
export NCCL_SOCKET_IFNAME=lo    # 使用本地回环接口
export NCCL_TIMEOUT=1800        # 设置超时时间
export NCCL_BLOCKING_WAIT=1     # 使用阻塞等待

# 分布式训练配置
export MASTER_ADDR=localhost    # 主节点地址
export MASTER_PORT=29500        # 主节点端口
export WORLD_SIZE=1             # 总进程数
export RANK=0                   # 当前进程rank
```

### 网络接口配置

```bash
# 查看网络接口
ip addr show

# 设置特定网络接口
export NCCL_SOCKET_IFNAME=eth0  # 使用eth0接口
```

## 故障排除步骤

### 1. 检查进程状态

```bash
# 查看是否有残留进程
ps aux | grep deepspeed
ps aux | grep python

# 清理残留进程
pkill -f deepspeed
pkill -f "python.*train.py"
```

### 2. 检查端口占用

```bash
# 检查端口是否被占用
netstat -an | grep 29500
lsof -i :29500

# 如果端口被占用，可以更换端口
export MASTER_PORT=29501
```

### 3. 检查GPU状态

```bash
# 查看GPU使用情况
nvidia-smi

# 查看GPU内存
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
```

### 4. 检查网络连通性

```bash
# 测试本地端口
telnet localhost 29500

# 查看网络接口
ifconfig
ip addr show
```

## 常见错误及解决方案

### 错误1: NCCL初始化失败

```
NCCL error: unhandled system error
```

**解决方案**：
```bash
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
```

### 错误2: 端口被占用

```
Address already in use
```

**解决方案**：
```bash
# 更换端口
export MASTER_PORT=29501

# 或清理占用进程
lsof -ti:29500 | xargs kill -9
```

### 错误3: GPU内存不足

```
CUDA out of memory
```

**解决方案**：
- 减少batch size
- 减少模型大小
- 使用梯度累积

### 错误4: 网络超时

```
NCCL timeout
```

**解决方案**：
```bash
export NCCL_TIMEOUT=1800
export NCCL_BLOCKING_WAIT=1
```

## 最佳实践

### 1. 环境准备

```bash
# 清理环境
pkill -f deepspeed
pkill -f python

# 设置环境变量
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
```

### 2. 启动顺序

```bash
# 1. 检查GPU状态
nvidia-smi

# 2. 检查端口
netstat -an | grep 29500

# 3. 启动训练
./scripts/run_deepspeed_fixed.sh
```

### 3. 监控训练

```bash
# 监控GPU使用
watch -n 1 nvidia-smi

# 监控进程
ps aux | grep deepspeed

# 查看日志
tail -f nohup.out
```

## 相关文件

- `scripts/run_deepspeed_fixed.sh`: 修复版本训练脚本
- `scripts/run_deepspeed_simple.sh`: 简化版本训练脚本
- `scripts/debug_deepspeed_init.py`: 调试脚本
- `configs/ds_s2.json`: DeepSpeed配置文件

## 联系支持

如果问题仍然存在，请提供以下信息：
1. 完整的错误日志
2. 系统环境信息
3. GPU配置信息
4. 网络配置信息 