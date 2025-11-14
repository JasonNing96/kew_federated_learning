# FLQ 联邦学习系统使用指南

## 环境配置（使用 conda activate fed）

### 1. 创建并激活 conda 环境

```bash
# 创建环境
conda env create -f environment.yml

# 激活环境（每次使用前都需要）
conda activate fed

# 验证环境
python test_environment.py
```

### 2. 环境验证输出示例

正确配置后，你应该看到类似输出：

```
========================================
    FLQ 联邦学习环境检查
========================================

✓ Python 版本: Python 3.9.16
✓ Conda 环境: fed
✓ CUDA 支持: CUDA 可用: release 11.4, V11.4.315
✓ TensorFlow GPU: TensorFlow 2.12.0, GPU可用: GPU 0: /physical_device:GPU:0
✓ numpy: numpy 1.21.0
✓ pandas: pandas 1.5.3
✓ flask: flask 2.2.3
✓ requests: requests 2.28.2
✓ openpyxl: openpyxl 3.0.10
✓ scipy: scipy 1.9.3
✓ matplotlib: matplotlib 3.6.3
✓ TensorFlow 功能: TensorFlow 功能测试通过，Loss: 0.8234

========================================
🎉 所有检查通过！环境配置正确。
现在可以运行 FLQ 联邦学习系统了。
========================================
```

## 快速测试（单机模拟分布式）

### 1. 一键测试脚本

```bash
# 确保在 fed 环境中
conda activate fed

# 运行快速测试（会自动启动 1 个 Master + 2 个 Workers）
./quick_test.sh
```

### 2. 手动测试步骤

**终端 1 - 启动 Master**
```bash
conda activate fed
./start_master.sh
```

**终端 2 - 启动 Worker 0**
```bash
conda activate fed
./start_worker.sh --worker_id 0 --master_ip 127.0.0.1
```

**终端 3 - 启动 Worker 1**
```bash
conda activate fed
./start_worker.sh --worker_id 1 --master_ip 127.0.0.1
```

**终端 4 - 监控状态**
```bash
# 查看系统状态
curl http://127.0.0.1:5000/status

# 停止训练
curl -X POST http://127.0.0.1:5000/stop
```

## 多机部署（Jetson Nano 集群）

### 1. 网络配置

确保所有 Jetson Nano 设备在同一网络中，并且可以互相 ping 通：

```bash
# 在每个设备上测试网络连通性
ping 192.168.1.100  # Master IP
ping 192.168.1.101  # Worker 1 IP
ping 192.168.1.102  # Worker 2 IP
# ... 其他 Worker IPs
```

### 2. 修改部署配置

编辑 `deploy_jetson.sh`：

```bash
vim deploy_jetson.sh

# 修改以下变量
MASTER_IP="192.168.1.100"          # 你的 Master 节点 IP
WORKER_IPS=("192.168.1.101" "192.168.1.102" "192.168.1.103" "192.168.1.104")
USERNAME="jetson"                   # Jetson Nano 用户名
```

### 3. 执行自动部署

```bash
# 自动部署到所有节点
./deploy_jetson.sh
```

### 4. 手动启动集群

**在 Master 节点 (192.168.1.100)**
```bash
conda activate fed
./start_master.sh
```

**在各 Worker 节点**
```bash
# Worker 0 (192.168.1.101)
conda activate fed
./start_worker.sh --worker_id 0 --master_ip 192.168.1.100

# Worker 1 (192.168.1.102)
conda activate fed
./start_worker.sh --worker_id 1 --master_ip 192.168.1.100

# Worker 2 (192.168.1.103)
conda activate fed
./start_worker.sh --worker_id 2 --master_ip 192.168.1.100

# Worker 3 (192.168.1.104)
conda activate fed
./start_worker.sh --worker_id 3 --master_ip 192.168.1.100
```

## 配置调整

### 1. 基本配置 (flq_config.json)

```json
{
  "system": {
    "total_workers": 4,        // 总 Worker 数量
    "min_workers": 2,         // 开始训练的最少 Worker 数
    "max_rounds": 1000        // 最大训练轮数
  },
  
  "flq_algorithm": {
    "mode": "laq8",           // 量化模式: fedavg, bbit, bin, laq8
    "b_up": 8,                // 上行量化位数
    "b_down": 8               // 下行量化位数
  },
  
  "dataset": {
    "name": "fmnist",         // 数据集: mnist, fmnist
    "batch_size": 32          // 批大小
  }
}
```

### 2. Jetson Nano 优化配置

```json
{
  "jetson_optimization": {
    "gpu_memory_limit_mb": 2048,    // GPU 内存限制 (MB)
    "enable_mixed_precision": false, // 是否启用混合精度
    "prefetch_size": 2,             // 数据预取大小
    "num_parallel_calls": 2         // 并行调用数
  }
}
```

## 监控和调试

### 1. 实时监控

```bash
# 查看系统状态
watch -n 5 'curl -s http://192.168.1.100:5000/status | python -m json.tool'

# 监控日志
tail -f logs/master_*.log
tail -f logs/worker*_*.log
```

### 2. 性能调试

```bash
# 查看 GPU 使用情况
nvidia-smi

# 查看系统资源
htop

# 查看网络状态
netstat -tuln | grep 5000
```

### 3. 常见问题排查

**问题 1: Worker 连接失败**
```bash
# 检查网络连通性
ping <master_ip>

# 检查端口是否开放
telnet <master_ip> 5000

# 检查防火墙
sudo ufw status
```

**问题 2: CUDA 内存不足**
```bash
# 减少批大小
vim flq_config.json
# 修改 "batch_size": 16

# 减少 GPU 内存限制
# 修改 "gpu_memory_limit_mb": 1024
```

**问题 3: TensorFlow 版本问题**
```bash
# 重新创建环境
conda env remove -n fed
conda env create -f environment.yml
```

## 结果分析

### 1. 结果文件位置

训练结果自动保存在：
- `results/` 目录下的 Excel 文件
- 文件名格式: `flq_results_<dataset>_<mode>_<timestamp>.xlsx`

### 2. 结果内容

Excel 文件包含以下数据：
- `round`: 训练轮次
- `accuracy`: 全局模型准确率
- `loss`: 全局模型损失
- `selected_workers`: 每轮选中的工作节点数
- `total_bits_up`: 累计上行通信量（比特）
- `round_time`: 每轮训练时间（秒）

### 3. 可视化分析

```bash
# 使用现有的绘图脚本
python plot_flq_fed.py --input results/flq_results_fmnist_laq8_*.xlsx
```

## 故障恢复

### 1. 紧急停止

```bash
# 停止所有进程
pkill -f flq_master.py
pkill -f flq_worker.py

# 或者发送 HTTP 停止请求
curl -X POST http://<master_ip>:5000/stop
```

### 2. 清理和重启

```bash
# 清理临时文件
rm -f flq_config_worker*.json
rm -f test_config.json

# 清理日志（可选）
rm -rf logs/*
rm -rf test_logs/*

# 重新启动
conda activate fed
./start_master.sh
```

## 性能优化建议

### 1. Jetson Nano 具体优化

- **GPU 内存**: 设置合适的 `gpu_memory_limit_mb` (1024-2048)
- **批大小**: 根据数据集调整 `batch_size` (16-64)
- **并行度**: 设置 `num_parallel_calls` 为 CPU 核心数
- **网络**: 使用有线网络连接，避免 WiFi

### 2. 算法参数调优

- **量化位数**: `b_up=8, b_down=8` 平衡精度和通信开销
- **懒惰参数**: 调整 `thr_scale` 控制传输频率
- **预算限制**: 设置 `up_budget_bits` 控制通信预算

## 高级功能

### 1. 自定义数据集

修改 `flq_worker.py` 中的 `load_dataset` 函数以支持自定义数据集。

### 2. 自定义模型

修改 `build_model` 函数以使用不同的神经网络架构。

### 3. 动态调整参数

通过 HTTP API 动态调整训练参数（需要扩展 API 接口）。
