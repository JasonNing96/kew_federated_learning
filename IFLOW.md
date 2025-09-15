# 联邦学习系统 (Federated Learning System)

## 项目概述

这是一个基于Kubernetes的联邦学习系统实现，支持在云边协同环境中进行分布式机器学习训练。系统采用主从架构，包含一个主节点(master)和多个工作节点(worker)，通过联邦平均算法(FedAvg)进行模型聚合。

### 主要特性
- **云边协同部署**: 支持云端和边缘节点的分布式部署
- **联邦学习**: 使用FedAvg算法进行模型聚合
- **量化支持**: 包含多种量化算法实现(4-bit, 8-bit等)
- **内存管理**: 工作节点支持内存压载功能
- **监控指标**: 提供Prometheus格式的监控指标
- **容器化部署**: 完整的Docker和Kubernetes支持

## 系统架构

### 组件结构
```
/home/njh/project/kew_federated_learning/
├── master.py              # 主节点服务程序
├── worker.py              # 工作节点客户端程序
├── FLQ.py                 # 联邦学习量化算法实现
├── Dockerfile.master      # 主节点容器镜像
├── Dockerfile.worker      # 工作节点容器镜像
├── master-deploy.yaml     # 主节点Kubernetes部署配置
├── worker-daemonset.yaml  # 工作节点Kubernetes部署配置
└── python-3.9-slim.tar    # Python基础镜像包
```

### 技术栈
- **后端**: Python 3.9, Flask, NumPy
- **机器学习**: TensorFlow 1.x
- **容器化**: Docker, Kubernetes
- **量化算法**: StocQ, MSQ_v2, SDQ, BetaQ
- **监控**: Prometheus指标格式

## 核心功能

### 1. 主节点 (Master)
- **模型管理**: 维护全局模型状态
- **聚合算法**: 实现FedAvg联邦平均算法
- **API服务**: 提供RESTful接口供工作节点调用
- **监控指标**: 暴露Prometheus格式的监控数据

**主要端点**:
- `POST /update`: 工作节点上报本地训练结果
- `GET /global`: 获取当前全局模型
- `GET /status`: 查看系统状态
- `GET /metrics`: 获取Prometheus监控指标

### 2. 工作节点 (Worker)
- **本地训练**: 在本地数据上进行模型训练
- **梯度计算**: 计算本地模型更新
- **通信管理**: 与主节点进行模型同步
- **内存管理**: 支持内存压载和触页机制

### 3. 量化算法 (Quantization)
- **多种量化方案**: 支持4-bit、8-bit等不同精度量化
- **自适应量化**: 根据数据分布进行自适应量化
- **梯度压缩**: 减少通信开销

## 部署和运行

### 环境要求
- Kubernetes集群
- Docker环境
- Python 3.9+

### 构建镜像
```bash
# 构建主节点镜像
docker build -f Dockerfile.master -t fl-master:local .

# 构建工作节点镜像
docker build -f Dockerfile.worker -t fl-worker:local .
```

### Kubernetes部署
```bash
# 部署主节点
kubectl apply -f master-deploy.yaml

# 部署工作节点(DaemonSet)
kubectl apply -f worker-daemonset.yaml
```

### 本地测试运行
```bash
# 启动主节点
python master.py

# 启动工作节点(设置主节点地址)
export MASTER_ADDR=http://localhost:5000
python worker.py
```

## 配置参数

### 主节点配置
- `MODEL_DIM`: 模型参数维度 (默认: 100,000)
- `AGG_MIN_CLIENTS`: 最小聚合客户端数 (默认: 2)
- `SEED`: 随机种子 (默认: 42)

### 工作节点配置
- `MASTER_ADDR`: 主节点地址 (默认: http://master-service:5000)
- `WORKER_ID`: 工作节点ID (默认: 主机名)
- `MODEL_DIM`: 模型维度 (默认: 100,000)
- `LOCAL_STEPS`: 本地训练步数 (默认: 5)
- `BATCH_SIZE`: 批大小 (默认: 64)
- `DATASET_SIZE`: 本地数据集大小 (默认: 50,000)
- `LR`: 学习率 (默认: 0.05)
- `MEM_MB`: 内存压载大小 (默认: 500MB)

## 算法实现

### 联邦平均算法 (FedAvg)
```python
# 权重聚合公式
w_{t+1} = w_t + sum_k ( (n_k / N_total) * delta_k )
```

### 量化算法
- **StocQ**: 随机量化
- **MSQ_v2**: 改进的最小平方量化
- **SDQ**: 标准差量化
- **BetaQ**: Beta分布量化

## 监控和调试

### 系统状态查询
```bash
# 查看主节点状态
curl http://master-service:5000/status

# 获取Prometheus指标
curl http://master-service:5000/metrics
```

### 日志查看
```bash
# 查看主节点日志
kubectl logs -l app=master

# 查看工作节点日志
kubectl logs -l app=worker
```

## 开发约定

### 代码风格
- 使用英文注释和文档字符串
- 遵循PEP 8 Python编码规范
- 函数和变量命名采用snake_case

### 错误处理
- 网络通信包含超时和重试机制
- 输入参数验证和错误返回
- 异常捕获和日志记录

### 性能优化
- 使用NumPy进行高效的数值计算
- 内存预分配和复用
- 批量处理和向量化操作

## 扩展功能

### 支持的扩展方向
1. **算法扩展**: 支持更多联邦学习算法(如FedProx, FedNova等)
2. **量化优化**: 实现更高效的量化算法
3. **安全增强**: 添加差分隐私和同态加密
4. **异构支持**: 支持不同模型架构和异构设备
5. **自动调优**: 实现超参数自动优化

### 节点亲和性
- 主节点部署在云端节点(role: cloud)
- 工作节点部署在边缘节点(role: edge1, edge2, edge3)

## 注意事项

1. **版本兼容性**: 当前使用TensorFlow 1.x，建议升级到TensorFlow 2.x
2. **网络配置**: 确保Kubernetes服务发现和DNS配置正确
3. **资源限制**: 建议为容器设置合理的CPU和内存限制
4. **数据隐私**: 确保本地数据符合隐私保护要求
5. **模型收敛**: 监控模型收敛情况，调整超参数以获得最佳性能