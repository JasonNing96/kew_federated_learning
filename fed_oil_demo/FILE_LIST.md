# Fed Oil Demo - 完整文件清单

## 📋 文件列表

### 核心训练脚本
- `train_gpu_only.py` - GPU专用中心化训练脚本
- `server.py` - 联邦学习参数服务器
- `client.py` - 联邦学习客户端

### 工具脚本
- `split_dataset.py` - 数据集切分工具（IID分片）
- `check_environment.py` - 环境检查脚本

### 启动脚本
- `start_server.sh` - 启动联邦学习服务器
- `start_clients.sh` - 启动3个客户端（本地测试用）
- `run_centralized.sh` - 启动中心化训练
- `stop_all.sh` - 停止所有联邦学习进程

### 模型文件
- `yolov8s.pt` (22MB) - YOLOv8s预训练模型（中心化训练用）
- `yolov8n.pt` (6.2MB) - YOLOv8n预训练模型（联邦学习用）

### 文档
- `README.md` - 完整项目文档
- `QUICKSTART.md` - 快速开始指南
- `FILE_LIST.md` - 本文件（文件清单）

### 配置
- `requirements.txt` - Python依赖列表

### 数据集

#### 原始完整数据集
```
oil-detection-2-2/          (158MB)
├── data.yaml               - 数据集配置
├── train/                  - 训练集（2488张图片）
│   ├── images/
│   └── labels/
├── valid/                  - 验证集（164张图片）
│   ├── images/
│   └── labels/
└── test/                   - 测试集
    ├── images/
    └── labels/
```

#### 客户端数据（运行split_dataset.py后生成）
```
client1/                    (55MB)
├── dataset/
│   ├── train/              - 829张图片
│   │   ├── images/
│   │   └── labels/
│   └── val/                - 164张图片
│       ├── images/
│       └── labels/
└── oil.yaml                - 客户端1数据配置

client2/                    (56MB)
├── dataset/
│   ├── train/              - 829张图片
│   └── val/                - 164张图片
└── oil.yaml

client3/                    (56MB)
├── dataset/
│   ├── train/              - 830张图片
│   └── val/                - 164张图片
└── oil.yaml
```

### 训练输出（运行后生成）

#### 中心化训练输出
```
runs/
└── oil_spill_gpu/
    ├── weights/
    │   ├── best.pt
    │   └── last.pt
    ├── results.png
    └── ...
```

#### 联邦学习输出
```
server_checkpoints/         - 服务器聚合模型
├── global_round_1.pt
├── global_round_2.pt
└── ...

client1/runs_fed/           - 客户端1训练记录
client2/runs_fed/           - 客户端2训练记录
client3/runs_fed/           - 客户端3训练记录

client1.log                 - 客户端1日志
client2.log                 - 客户端2日志
client3.log                 - 客户端3日志
```

## 📊 总大小

- 完整项目（含数据）: ~351MB
- 原始数据集: 158MB
- 客户端数据（3个）: 167MB
- 模型文件: 28MB
- 脚本和文档: <1MB

## 🔧 首次使用必须文件

必须存在的文件：
- ✅ `train_gpu_only.py`
- ✅ `server.py`
- ✅ `client.py`
- ✅ `split_dataset.py`
- ✅ `yolov8s.pt`
- ✅ `yolov8n.pt`
- ✅ `oil-detection-2-2/` 目录

首次使用必须运行：
```bash
python split_dataset.py  # 生成client1/2/3目录
```

## 📦 多设备部署文件分发

### 服务器设备需要的文件
```
- server.py
- yolov8n.pt
- client1/oil.yaml (仅用于读取nc配置)
- start_server.sh (可选)
```

### 客户端1设备需要的文件
```
- client.py
- yolov8n.pt
- client1/ (完整目录)
  ├── dataset/
  └── oil.yaml
```

### 客户端2设备需要的文件
```
- client.py
- yolov8n.pt
- client2/ (完整目录)
```

### 客户端3设备需要的文件
```
- client.py
- yolov8n.pt
- client3/ (完整目录)
```

## 🎯 可选文件

以下文件是可选的，不影响核心功能：
- `check_environment.py` - 环境检查（建议保留）
- `README.md` - 文档（建议保留）
- `QUICKSTART.md` - 快速指南（建议保留）
- `FILE_LIST.md` - 本文件（可删除）
- `*.sh` 脚本 - 可以手动运行Python命令代替

## ⚠️ 不要删除的文件

以下文件删除后会导致无法运行：
- ❌ `yolov8n.pt` / `yolov8s.pt` - 预训练模型
- ❌ `oil-detection-2-2/` - 原始数据集
- ❌ `client*/dataset/` - 客户端数据
- ❌ `*.yaml` - 数据配置文件
- ❌ 核心Python脚本

## 🔄 可重新生成的文件

以下文件可以删除后重新生成：
- ✅ `client1/`, `client2/`, `client3/` - 运行 `split_dataset.py` 重新生成
- ✅ `server_checkpoints/` - 训练输出
- ✅ `runs/` - 训练输出
- ✅ `*.log` - 日志文件

