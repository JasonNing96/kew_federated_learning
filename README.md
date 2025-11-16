# FLQ-Fed: 联邦学习量化训练框架

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

基于 YOLO 的联邦学习量化训练框架，支持低比特量化和通信压缩。

## 🚀 快速开始

### 安装依赖

```bash
conda activate fedllm  # 或你的 Python 3.9+ 环境
pip install -r requirements.txt
```

### 一键训练

```bash
# 方式1: 使用脚本（推荐）
./scripts/run_fl.sh

# 方式2: 使用 Python 模块
python -m app.runner train

# 方式3: 自定义配置
python -m app.runner train --config configs/custom.yaml
```

### 手动启动

```bash
# 终端1: 启动服务器
python -m app.runner server

# 终端2-4: 启动客户端
python -m app.runner client --id 1
python -m app.runner client --id 2
python -m app.runner client --id 3
```

## 📂 项目结构

```
kew_federated_learning/
├── app/                      # 🎯 核心代码（仅4个文件）
│   ├── runner.py            #    统一入口
│   ├── server.py            #    联邦服务器
│   ├── client.py            #    客户端逻辑
│   ├── model_utils.py       #    量化/聚合工具
│   └── config.py            #    配置加载
│
├── configs/                  # 📋 配置文件
│   └── flq_config.yaml      #    主配置
│
├── scripts/                  # 🛠️ 便捷脚本
│   ├── run_fl.sh            #    一键启动
│   ├── stop_fl.sh           #    停止训练
│   ├── status.sh            #    查看状态
│   └── split_dataset.py     #    数据切分
│
├── models/                   # 🤖 预训练模型
│   └── yolov8n.pt
│
├── data/                     # 📊 数据集
│   ├── client1/
│   ├── client2/
│   └── client3/
│
├── outputs/                  # 📁 训练输出
│   ├── server/              #    全局模型
│   └── client*/             #    客户端结果
│
├── tests/                    # ✅ 单元测试
├── docs/                     # 📚 文档
└── legacy/                   # 🗄️ 旧版代码
```

## 📋 配置说明

编辑 `configs/flq_config.yaml`:

```yaml
training:
  rounds: 5                   # 训练轮数
  clients_per_round: 3        # 每轮客户端数
  local_epochs: 1             # 本地训练轮数

quantization:
  enabled: false              # 量化开关
  bits: 8                     # 量化比特 (1/4/8)

model:
  name: "models/yolov8n.pt"  # 模型路径
  device: "cuda:0"            # 训练设备

client:
  batch_size: 8
  workers: 0                  # DataLoader进程数（推荐0）
  verbose: true
  enable_val: false           # 加快训练
  enable_plots: false         # 加快训练
```

## 🔧 常用命令

```bash
# 查看训练状态
./scripts/status.sh
curl http://localhost:8087/status

# 停止所有训练
./scripts/stop_fl.sh

# 监控 GPU
watch -n 1 nvidia-smi

# 查看日志
tail -f outputs/server/logs/server.log
tail -f outputs/client1/logs/client1.log

# 数据准备（首次使用）
python scripts/split_dataset.py --clients 3
```

## 📊 性能指标

| 配置 | 上行/轮 | 下行/轮 | 压缩率 | mAP50 |
|------|---------|---------|--------|-------|
| FP32基线 | 0.29 Gbit | 0.29 Gbit | 1.0x | 0.83 |
| 8-bit量化 | 0.07 Gbit | 0.07 Gbit | 4.0x | 0.82 |
| 1-bit量化 | 0.01 Gbit | 0.07 Gbit | 7.3x | 0.78 |

## 🐛 故障排除

### 端口被占用

```bash
./scripts/stop_fl.sh
# 或手动: pkill -f "app.runner"
```

### 客户端卡住

- 检查 `workers` 设置（推荐 0）
- 关闭 `enable_val` 和 `enable_plots`
- 查看客户端日志定位问题

### GPU 未使用

- 确认 `device: "cuda:0"` 配置
- 检查 CUDA 是否可用: `python -c "import torch; print(torch.cuda.is_available())"`

## 📚 文档

- [项目检查点](docs/CHECKPOINTS.md)
- [API 文档](docs/API.md)
- [开发指南](docs/DEVELOPMENT.md)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 🙏 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [FastAPI](https://fastapi.tiangolo.com/)
- [PyTorch](https://pytorch.org/)
