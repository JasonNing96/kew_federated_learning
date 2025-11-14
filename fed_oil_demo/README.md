# Fed Oil Detection Demo

最小化的分布式联邦学习石油泄漏检测系统演示项目。

## 📁 项目结构

```
fed_oil_demo/
├── train_gpu_only.py        # 中心化训练脚本
├── server.py                 # 联邦学习参数服务器
├── client.py                 # 联邦学习客户端
├── split_dataset.py          # 数据集切分工具
├── start_server.sh           # 启动服务器脚本
├── start_clients.sh          # 启动客户端脚本（本地测试用）
├── stop_all.sh               # 停止所有进程
├── yolov8s.pt                # YOLOv8s 预训练模型（用于中心化训练）
├── yolov8n.pt                # YOLOv8n 预训练模型（用于联邦学习）
├── oil-detection-2-2/        # 完整原始数据集
│   ├── data.yaml             # 数据集配置
│   ├── train/                # 训练集
│   ├── valid/                # 验证集
│   └── test/                 # 测试集
├── client1/                  # 客户端1（运行split_dataset.py后生成）
│   ├── dataset/              # 客户端1的数据子集
│   └── oil.yaml              # 客户端1的数据配置
├── client2/                  # 客户端2
├── client3/                  # 客户端3
├── requirements.txt          # Python依赖
└── README.md                 # 本文档
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 验证 CUDA 可用
python -c "import torch; print(torch.cuda.is_available())"

# 运行环境检查
python check_environment.py
```

### 2. 数据准备

**首次使用前，需要切分数据集：**

```bash
# 将完整数据集切分给3个客户端
python split_dataset.py
```

这会创建 `client1/`, `client2/`, `client3/` 目录，每个包含独立的数据子集。

### 3. 运行中心化训练（对比基准）

```bash
# 使用完整数据集进行传统的中心化训练
python train_gpu_only.py
# 或
./run_centralized.sh
```

### 4. 运行联邦学习训练

#### 方式一：单机多客户端测试（本地模拟）

```bash
# 终端1: 启动服务器
./start_server.sh

# 终端2: 启动所有客户端（后台运行）
./start_clients.sh

# 监控训练日志
tail -f client1.log
tail -f client2.log
tail -f client3.log

# 停止所有进程
./stop_all.sh
```

#### 方式二：多设备分布式训练（真实场景）

**在服务器设备上：**
```bash
# 启动参数服务器（监听所有网络接口）
./start_server.sh
# 或
python server.py

# 获取服务器IP地址
ip addr show | grep "inet "
# 例如：192.168.1.100
```

**在每个客户端设备上：**

1. 复制以下文件到客户端设备：
   - `client.py`
   - `yolov8n.pt`
   - `client1/` 目录（包含数据和配置）

2. 设置服务器地址并启动：
```bash
# 设置服务器地址
export FL_SERVER="http://192.168.1.100:8080"

# 启动客户端
CLIENT_ID=1 python client.py
```

**客户端2、3类似操作，使用相应的 CLIENT_ID 和数据目录。**

## 📊 配置参数

### 中心化训练参数

编辑 `train_gpu_only.py`:
```python
train_args = dict(
    epochs=50,      # 训练轮数
    batch=16,       # 批次大小
    imgsz=640,      # 图像大小
    device=0,       # GPU设备号
)
```

### 联邦学习服务器参数

编辑 `server.py`:
```python
CLIENTS_PER_ROUND = 3  # 每轮参与的客户端数
ROUNDS = 10            # 总训练轮数
MODEL_PATH = "yolov8n.pt"
```

### 联邦学习客户端参数

编辑 `client.py`:
```python
EPR = 5          # Epochs Per Round - 每轮本地训练epoch数
IMGZ = 640       # 图像尺寸
BATCH = 8        # 批次大小
DEVICE = 0       # GPU设备（CPU设为'cpu'）
```

### 数据切分参数

编辑 `split_dataset.py`:
```python
SOURCE_DIR = "oil-detection-2-2"  # 源数据集路径
NUM_CLIENTS = 3                    # 客户端数量
SEED = 42                          # 随机种子（保证可重复性）
```

## 🏗️ 架构说明

### 联邦学习流程

1. **初始化**：服务器加载预训练模型，等待客户端连接
2. **训练循环**（每轮）：
   - 客户端从服务器拉取全局模型
   - 客户端在本地数据上训练 EPR 个epoch
   - 客户端上传更新后的模型权重
   - 服务器使用 FedAvg 算法聚合所有客户端更新
3. **重复**直到完成所有轮次

### FedAvg 聚合算法

```
全局权重 = Σ (客户端权重 × 样本数权重)
其中：样本数权重 = 客户端样本数 / 总样本数
```

## 📈 训练输出

### 中心化训练输出
- **路径**: `runs/oil_spill_gpu/`
- **内容**:
  - `weights/best.pt` - 最佳模型权重
  - `weights/last.pt` - 最后一轮权重
  - 训练曲线图表

### 联邦学习输出
- **全局模型**: `server_checkpoints/global_round_*.pt`
- **客户端结果**: `client*/runs_fed/`
- **训练日志**: `client*.log` (使用start_clients.sh时)

## 🌐 网络配置

### 防火墙设置

如果使用多设备，需要开放服务器端口：

```bash
# Ubuntu/Debian
sudo ufw allow 8080/tcp

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload
```

### 内网访问

1. 服务器会监听 `0.0.0.0:8080`，支持内网访问
2. 客户端通过环境变量 `FL_SERVER` 指定服务器地址
3. 示例：`export FL_SERVER="http://192.168.1.100:8080"`

## 🔧 系统要求

### 硬件要求
- **服务器**: 任何可运行Python的设备（无需GPU）
- **客户端**: NVIDIA GPU推荐（可使用CPU但较慢）
- **网络**: 内网连接（同一局域网）

### 软件要求
- **Python**: 3.8+
- **CUDA**: 11.8+ (客户端GPU训练时)
- **PyTorch**: 2.0.0+
- **Ultralytics**: 8.3.0
- **FastAPI**: 0.104.0+

## 📝 使用场景

### 场景1: 本地测试（单机模拟）
适合开发和调试，使用 `start_server.sh` + `start_clients.sh`

### 场景2: 内网多设备训练
适合实验室/公司内网，多台设备协同训练

### 场景3: 对比实验
同时运行中心化训练和联邦学习，对比效果差异

## ⚠️ 注意事项

1. **数据切分**: 首次使用前必须运行 `split_dataset.py`
2. **网络连通性**: 确保客户端能访问服务器的8080端口
3. **同步问题**: 服务器会等待所有客户端完成当前轮次后才聚合
4. **资源分配**: 多设备时注意每个设备的显存和算力
5. **日志监控**: 使用 `tail -f client*.log` 监控训练进度

## 🐛 故障排除

### 客户端连接不上服务器
```bash
# 检查服务器是否运行
curl http://服务器IP:8080/

# 检查防火墙
sudo ufw status

# 测试网络连通性
ping 服务器IP
```

### 显存不足
- 降低客户端的 `BATCH` 大小（如改为4）
- 使用CPU训练：设置 `DEVICE='cpu'`

### 数据路径错误
- 确保在 `fed_oil_demo` 目录下运行所有脚本
- 运行 `python check_environment.py` 检查环境

### 服务器等待超时
- 确保所有客户端都已启动并连接
- 检查 `CLIENTS_PER_ROUND` 设置是否与实际客户端数匹配

## 📚 API端点

服务器提供以下API：

- `GET /` - 服务器状态查询
- `GET /global` - 拉取全局模型（客户端使用）
- `POST /update` - 上传本地更新（客户端使用）
- `GET /status` - 详细状态监控

访问示例：
```bash
curl http://localhost:8080/status
```

## 🎯 项目特点

- ✅ **真实分布式**: 支持多设备真实联邦学习
- ✅ **易于部署**: 最小化配置，开箱即用
- ✅ **灵活配置**: 支持本地测试和多设备部署
- ✅ **完整工具**: 包含数据切分、训练、监控全流程
- ✅ **详细文档**: 完整的使用说明和故障排除

## 📧 联系方式

如有问题或建议，请联系项目维护者。

## 📖 参考资料

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [FedAvg 论文](https://arxiv.org/abs/1602.05629)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
