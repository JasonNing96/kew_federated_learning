# FLQ-Fed 联邦学习测试指南

**日期**: 2025-11-15  
**环境**: fedllm (Python 3.10.9 + Ultralytics 8.3.50)

---

## 🔧 已修复的问题

### 1. **客户端全局变量问题** ✅
- **文件**: `core/client.py`
- **问题**: 使用了未定义的 `FL_SERVER` 变量
- **修复**: 统一使用 `SERVER` 变量

### 2. **服务器配置加载问题** ✅
- **文件**: `core/server.py`
- **问题**: 配置加载后全局变量未更新
- **修复**: 简化配置加载逻辑，调整日志级别

### 3. **输出显示问题** ✅
- **文件**: `flq-fed.py`
- **问题**: 日志只保存到文件，终端无输出
- **修复**: 添加实时输出功能（同时保存日志和显示终端）

### 4. **GPU使用问题** ✅
- **客户端**: 已设置为 `cuda:0` (使用GPU训练)
- **服务器**: 使用CPU进行聚合（无需GPU）

### 5. **模型和数据路径** ✅
- **模型**: `models/yolov8n.pt` ✓
- **数据**: `data/client1/oil.yaml` ✓
- **端口**: 8087 ✓

---

## 🚀 快速测试

### 方法1: 使用测试脚本（推荐）

```bash
# 运行快速测试（1轮训练）
./test_fl_quick.sh
```

### 方法2: 手动运行

```bash
# 设置Python环境
export PYTHON=/home/njh/miniconda3/envs/fedllm/bin/python

# 完整训练（自动启动服务器和客户端）
$PYTHON flq-fed.py train --config configs/flq_config.yaml
```

### 方法3: 分别启动（用于调试）

```bash
# 终端1: 启动服务器
/home/njh/miniconda3/envs/fedllm/bin/python flq-fed.py server --config configs/flq_config.yaml

# 终端2-4: 启动3个客户端
/home/njh/miniconda3/envs/fedllm/bin/python flq-fed.py client --id 1 --server http://localhost:8087
/home/njh/miniconda3/envs/fedllm/bin/python flq-fed.py client --id 2 --server http://localhost:8087
/home/njh/miniconda3/envs/fedllm/bin/python flq-fed.py client --id 3 --server http://localhost:8087
```

---

## 📊 当前配置 (configs/flq_config.yaml)

```yaml
training:
  rounds: 1                     # 训练轮数（快速测试）
  clients_per_round: 3          # 每轮参与的客户端数
  local_epochs: 2               # 客户端本地训练轮数

quantization:
  enabled: false                # 是否启用量化（false=FP32基线）
  bits: 8                       # 量化比特数
  
model:
  name: "models/yolov8n.pt"     # YOLO模型
  device: "cuda:0"              # 训练设备

server:
  host: "0.0.0.0"
  port: 8087
```

---

## 📁 输出文件

训练完成后，结果保存在以下位置：

```
outputs/
├── server/
│   ├── checkpoints/
│   │   └── global_round_1.pt      # 全局模型
│   └── logs/
│       └── server.log              # 服务器日志
├── client1/
│   ├── runs/                       # 训练结果
│   └── logs/
│       └── client1.log             # 客户端1日志
├── client2/
│   └── logs/
│       └── client2.log
└── client3/
    └── logs/
        └── client3.log
```

---

## 🔍 监控命令

### 查看实时日志

```bash
# 服务器日志
tail -f outputs/server/logs/server.log

# 客户端1日志
tail -f outputs/client1/logs/client1.log

# GPU使用情况
watch -n 1 nvidia-smi
```

### 检查进程

```bash
# 查看联邦学习进程
ps aux | grep flq-fed.py

# 查看端口占用
lsof -i:8087
```

---

## 🧪 预期输出

### 服务器输出示例

```
[HH:MM:SS] 🚀 初始化参数服务器...
[HH:MM:SS] 📦 加载基础模型: models/yolov8n.pt
[HH:MM:SS] 🏷️  类别数: 2
[HH:MM:SS] ✅ 服务器就绪，等待客户端连接...
[HH:MM:SS] 📊 配置: 3客户端/轮 × 1轮
[HH:MM:SS] 📤 客户端拉取全局模型 (Round 0)
[HH:MM:SS] 📥 收到客户端更新: 样本数=100, 已收集=1/3
[HH:MM:SS] 🔄 开始第 1 轮聚合...
```

### 客户端输出示例

```
[HH:MM:SS] 🚀 客户端 #1 启动
[HH:MM:SS] 🌐 服务器: http://localhost:8087
[HH:MM:SS] 📁 数据配置: data/client1/oil.yaml
[HH:MM:SS] 🖥️  训练设备: cuda:0
[HH:MM:SS] 📥 拉取全局模型成功 (Round 0)
[HH:MM:SS] 🎯 开始本地训练 Round 0...
Epoch 1/2: 100%|████████| 10/10 [00:15<00:00]
[HH:MM:SS] 📤 上传本地更新成功 (样本数=100)
[HH:MM:SS] 🎉 联邦训练完成
```

---

## ⚠️ 常见问题

### 1. 端口被占用

```bash
# 查找占用8087端口的进程
lsof -i:8087

# 杀死进程
kill -9 <PID>
```

### 2. CUDA内存不足

修改 `configs/flq_config.yaml`:
```yaml
client:
  batch_size: 4  # 减小batch size
```

### 3. 客户端无法连接服务器

检查服务器是否启动：
```bash
curl http://localhost:8087/status
```

### 4. 数据集不存在

确保数据已分片：
```bash
ls -la data/client1/
ls -la data/client2/
ls -la data/client3/
```

---

## 📈 性能指标

### 预期指标（1轮训练，3客户端）

| 指标 | 值 |
|------|-----|
| 训练时间 | ~2-5分钟 (取决于GPU) |
| 上行通信 | ~0.2 GB (FP32) |
| 下行通信 | ~0.6 GB |
| GPU显存 | ~2-4 GB (batch_size=8) |
| 参数数量 | ~3M (YOLOv8n) |

---

## 🎯 下一步

测试成功后，可以：

1. **增加训练轮数**: 修改 `configs/flq_config.yaml` 中的 `rounds`
2. **启用量化**: 设置 `quantization.enabled: true`
3. **对比实验**: 运行不同量化比特数(1, 4, 8)的实验
4. **评估模型**: 使用验证集评估聚合后的模型

---

**创建时间**: 2025-11-15  
**最后更新**: 2025-11-15  
**状态**: ✅ 就绪测试

