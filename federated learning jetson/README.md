# FLQ 联邦学习分布式系统

基于 `flq_fed_v3.py` 算法的分布式联邦学习系统，专门针对 Jetson Nano 设备优化。

## 系统架构

```
Master 节点                    Worker 节点 1,2,3,4...
┌─────────────────┐           ┌─────────────────┐
│  flq_master.py  │◄─────────►│  flq_worker.py  │
│                 │           │                 │
│ • 全局模型管理   │           │ • 本地训练       │
│ • FLQ 聚合策略   │           │ • 梯度量化       │
│ • 预算选择       │           │ • 懒惰传输       │
│ • 性能评估       │           │ • 误差补偿       │
└─────────────────┘           └─────────────────┘
```

## 环境配置

### 1. 创建 Conda 环境

```bash
# 使用提供的环境配置文件
conda env create -f environment.yml

# 激活环境
conda activate fed

# 验证环境
python -c "import tensorflow as tf; print('TF版本:', tf.__version__); print('GPU可用:', tf.config.list_physical_devices('GPU'))"
```

### 2. 验证 Jetson Nano GPU

```bash
# 检查 CUDA
nvcc --version

# 检查 GPU 状态
nvidia-smi

# 测试 TensorFlow GPU
python -c "
import tensorflow as tf
print('TensorFlow 版本:', tf.__version__)
print('可用 GPU 设备:', tf.config.list_physical_devices('GPU'))
if tf.config.list_physical_devices('GPU'):
    print('✓ GPU 可用，TensorFlow 可以使用 CUDA')
else:
    print('✗ GPU 不可用，将使用 CPU')
"
```

## 快速启动

### 单机测试（模拟分布式）

1. **启动 Master**
```bash
conda activate fed
./start_master.sh
```

2. **启动 Workers**（开启多个终端）
```bash
# Terminal 1
conda activate fed
./start_worker.sh --worker_id 0 --master_ip 127.0.0.1

# Terminal 2  
conda activate fed
./start_worker.sh --worker_id 1 --master_ip 127.0.0.1

# Terminal 3
conda activate fed
./start_worker.sh --worker_id 2 --master_ip 127.0.0.1

# Terminal 4
conda activate fed
./start_worker.sh --worker_id 3 --master_ip 127.0.0.1
```

### 多机部署（Jetson Nano 集群）

1. **配置集群 IP**
编辑 `deploy_jetson.sh` 文件：
```bash
MASTER_IP="192.168.1.100"  # Master 节点 IP
WORKER_IPS=("192.168.1.101" "192.168.1.102" "192.168.1.103" "192.168.1.104")
```

2. **执行自动部署**
```bash
./deploy_jetson.sh
```

3. **手动启动**
```bash
# 在 Master 节点 (192.168.1.100)
conda activate fed
./start_master.sh

# 在 Worker 节点 1 (192.168.1.101)
conda activate fed
./start_worker.sh --worker_id 0 --master_ip 192.168.1.100

# 在 Worker 节点 2 (192.168.1.102)
conda activate fed
./start_worker.sh --worker_id 1 --master_ip 192.168.1.100

# ... 其他 Worker 节点类似
```

## 配置参数

编辑 `flq_config.json` 来调整系统参数：

### 系统配置
- `total_workers`: 总工作节点数
- `min_workers`: 开始训练的最少节点数
- `max_rounds`: 最大训练轮数

### FLQ 算法参数
- `mode`: 量化模式 (`fedavg`, `bbit`, `bin`, `laq8`)
- `b_up`: 上行量化位数
- `b_down`: 下行量化位数
- `scale_by_selected`: 是否按选中节点数放大步长

### 懒惰选择参数
- `D`: 历史记忆深度
- `ck`: 能量权重衰减因子
- `C`: 最大懒惰时钟
- `warmup`: 预热轮数

### 预算选择
- `sel_clients`: 固定选择的客户端数（0表示不限制）
- `up_budget_bits`: 上行通信预算（比特）

## 监控和管理

### Web API 接口

Master 节点提供以下 REST API：

```bash
# 获取系统状态
curl http://<master_ip>:5000/status

# 获取全局模型
curl http://<master_ip>:5000/get_global_model

# 停止训练并保存结果
curl -X POST http://<master_ip>:5000/stop
```

### 日志监控

```bash
# 查看 Master 日志
tail -f logs/master_*.log

# 查看 Worker 日志
tail -f logs/worker*_*.log
```

### 结果查看

训练结果保存在 `results/` 目录下的 Excel 文件中，包含：
- 每轮的准确率和损失
- 选中的工作节点数
- 通信开销统计
- 训练时间记录

## 性能优化

### Jetson Nano 专用优化

1. **GPU 内存管理**
   - 自动启用内存增长
   - 限制显存使用（2GB）
   - 避免 OOM 错误

2. **CUDA 优化**
   - 启用 CUDA 缓存
   - 混合精度计算（可选）
   - 并行数据加载

3. **网络优化**
   - 数据预取
   - 批处理优化
   - 异步通信

### 自定义优化

编辑配置文件中的 `jetson_optimization` 部分：
```json
"jetson_optimization": {
  "gpu_memory_limit_mb": 2048,
  "enable_mixed_precision": false,
  "prefetch_size": 2,
  "num_parallel_calls": 2
}
```

## 故障排除

### 常见问题

1. **CUDA 内存不足**
   - 减少 `batch_size`
   - 降低 `gpu_memory_limit_mb`
   - 检查其他进程占用

2. **Worker 连接失败**
   - 检查网络连通性
   - 确认防火墙设置
   - 验证 IP 地址配置

3. **TensorFlow 版本冲突**
   - 重新创建 conda 环境
   - 使用 Jetson 专用 TensorFlow

4. **训练不收敛**
   - 调整学习率 `lr`
   - 修改量化参数 `b_up`, `b_down`
   - 检查数据分布参数 `dir_alpha`

### 调试模式

启用详细日志：
```bash
export TF_CPP_MIN_LOG_LEVEL=0  # 显示所有 TensorFlow 日志
python flq_master.py --config flq_config.json --verbose
```

## 实验配置建议

### 小规模测试
```json
{
  "total_workers": 2,
  "min_workers": 2,
  "max_rounds": 100,
  "batch_size": 16,
  "local_epochs": 1
}
```

### 生产环境
```json
{
  "total_workers": 8,
  "min_workers": 4,
  "max_rounds": 1000,
  "batch_size": 32,
  "local_epochs": 2
}
```

## 文件说明

| 文件 | 功能 |
|------|------|
| `flq_worker.py` | Worker 节点主程序 |
| `flq_master.py` | Master 节点主程序 |
| `flq_config.json` | 系统配置文件 |
| `start_master.sh` | Master 启动脚本 |
| `start_worker.sh` | Worker 启动脚本 |
| `deploy_jetson.sh` | 集群自动部署脚本 |
| `environment.yml` | Conda 环境配置 |
| `flq_fed_v3.py` | 原始算法实现（参考） |

## 许可证

本项目基于 MIT 许可证开源。
