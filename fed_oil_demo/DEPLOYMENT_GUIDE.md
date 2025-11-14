# 多设备部署指南

本指南详细说明如何在多台设备上部署联邦学习系统。

## 🏗️ 架构概览

```
服务器设备 (Server Device)
    ↓
    8080端口 (FastAPI服务)
    ↓
    ├─→ 客户端设备1 (Client 1) - 训练829张图片
    ├─→ 客户端设备2 (Client 2) - 训练829张图片
    └─→ 客户端设备3 (Client 3) - 训练830张图片
```

## 📦 准备工作

### 1. 在开发机器上切分数据

首先在主开发机器上准备好所有数据：

```bash
cd fed_oil_demo
python split_dataset.py
```

这会生成 `client1/`, `client2/`, `client3/` 三个目录。

### 2. 确认网络环境

所有设备必须在同一局域网内，可以互相访问。

```bash
# 在各设备上测试网络连通性
ping 服务器IP
```

## 🖥️ 服务器设备部署

### 需要的文件
```
server_device/
├── server.py
├── yolov8n.pt
├── client1/
│   └── oil.yaml  (仅用于读取nc配置)
└── requirements.txt
```

### 安装步骤

```bash
# 1. 创建目录
mkdir -p fed_server
cd fed_server

# 2. 复制文件（从开发机器）
scp user@dev_machine:/path/to/fed_oil_demo/server.py .
scp user@dev_machine:/path/to/fed_oil_demo/yolov8n.pt .
scp -r user@dev_machine:/path/to/fed_oil_demo/client1 .
scp user@dev_machine:/path/to/fed_oil_demo/requirements.txt .

# 3. 安装依赖
pip install -r requirements.txt

# 4. 获取服务器IP
ip addr show | grep "inet " | grep -v "127.0.0.1"
# 记录下你的局域网IP，例如: 192.168.1.100
```

### 启动服务器

```bash
python server.py
```

服务器会在 `0.0.0.0:8080` 上监听，输出类似：
```
[23:45:12] 🚀 初始化参数服务器...
[23:45:12] 📦 加载基础模型: yolov8n.pt
[23:45:12] 🏷️  类别数: 2
[23:45:13] ✅ 服务器就绪，等待客户端连接...
[23:45:13] 📊 配置: 3客户端/轮 × 10轮
```

### 防火墙配置

如果无法连接，需要开放端口：

**Ubuntu/Debian:**
```bash
sudo ufw allow 8080/tcp
sudo ufw status
```

**CentOS/RHEL:**
```bash
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload
```

## 💻 客户端设备部署

### 客户端1设备

#### 需要的文件
```
client1_device/
├── client.py
├── yolov8n.pt
├── client1/
│   ├── dataset/
│   │   ├── train/
│   │   └── val/
│   └── oil.yaml
└── requirements.txt
```

#### 部署步骤

```bash
# 1. 创建目录
mkdir -p fed_client1
cd fed_client1

# 2. 复制文件（从开发机器）
scp user@dev_machine:/path/to/fed_oil_demo/client.py .
scp user@dev_machine:/path/to/fed_oil_demo/yolov8n.pt .
scp -r user@dev_machine:/path/to/fed_oil_demo/client1 .
scp user@dev_machine:/path/to/fed_oil_demo/requirements.txt .

# 3. 安装依赖
pip install -r requirements.txt

# 4. 设置服务器地址并启动
export FL_SERVER="http://192.168.1.100:8080"
CLIENT_ID=1 python client.py
```

### 客户端2设备

```bash
mkdir -p fed_client2
cd fed_client2

# 复制client2相关文件
scp user@dev_machine:/path/to/fed_oil_demo/client.py .
scp user@dev_machine:/path/to/fed_oil_demo/yolov8n.pt .
scp -r user@dev_machine:/path/to/fed_oil_demo/client2 .
scp user@dev_machine:/path/to/fed_oil_demo/requirements.txt .

pip install -r requirements.txt

export FL_SERVER="http://192.168.1.100:8080"
CLIENT_ID=2 python client.py
```

### 客户端3设备

```bash
mkdir -p fed_client3
cd fed_client3

# 复制client3相关文件
scp user@dev_machine:/path/to/fed_oil_demo/client.py .
scp user@dev_machine:/path/to/fed_oil_demo/yolov8n.pt .
scp -r user@dev_machine:/path/to/fed_oil_demo/client3 .
scp user@dev_machine:/path/to/fed_oil_demo/requirements.txt .

pip install -r requirements.txt

export FL_SERVER="http://192.168.1.100:8080"
CLIENT_ID=3 python client.py
```

## 📊 监控训练进度

### 在服务器上查看状态

```bash
# 实时监控
curl http://localhost:8080/status

# 或者使用watch
watch -n 5 "curl -s http://localhost:8080/status"
```

### 在客户端上查看日志

客户端会输出训练进度：
```
[23:46:01] 🚀 启动联邦学习客户端 #1
[23:46:01] 🌐 服务器地址: http://192.168.1.100:8080
[23:46:02] 📥 拉取全局模型成功 (Round 0)
[23:46:02] 🎯 开始本地训练 Round 0...
[23:47:15] ✅ 本地训练完成 (mAP50: 0.823)
[23:47:16] 📤 上传本地更新成功 (样本数=829)
```

## 🔧 故障排除

### 问题1: 客户端无法连接服务器

**症状**: `连接被拒绝` 或 `超时`

**解决方案**:
```bash
# 1. 检查服务器是否运行
curl http://192.168.1.100:8080/

# 2. 检查防火墙
sudo ufw status
sudo firewall-cmd --list-all

# 3. 测试网络连通性
ping 192.168.1.100
telnet 192.168.1.100 8080
```

### 问题2: 显存不足

**症状**: `CUDA out of memory`

**解决方案**:
编辑 `client.py`，降低BATCH大小：
```python
BATCH = 4  # 或更小，如2
```

### 问题3: 训练太慢

**客户端使用CPU**:
检查CUDA是否可用：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**网络慢**:
- 确保使用千兆网络
- 减少每轮训练的epoch数（修改 `EPR`）

### 问题4: 服务器等待客户端超时

**症状**: 服务器一直显示 "等待客户端..."

**原因**: 客户端数量不足

**解决方案**:
确保启动的客户端数量 = `CLIENTS_PER_ROUND` (默认3)

## 📈 性能优化

### 网络优化
- 使用千兆以太网连接
- 服务器和客户端在同一交换机下

### 计算优化
- 客户端使用GPU加速
- 调整 `BATCH` 大小以充分利用显存
- 多GPU可设置不同的 `DEVICE`

### 并行优化
如果有多个GPU设备，可以让同一个客户端使用不同GPU：
```python
# 客户端1使用GPU 0
DEVICE = 0

# 客户端2使用GPU 1 (如果有)
DEVICE = 1
```

## 🎯 推荐硬件配置

### 服务器设备
- **CPU**: 任意（仅做模型聚合）
- **RAM**: 4GB+
- **网络**: 千兆网卡
- **GPU**: 不需要

### 客户端设备
- **CPU**: 4核+
- **RAM**: 8GB+
- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **网络**: 千兆网卡

## 📝 启动顺序

1. **先启动服务器**（等待客户端连接）
2. **再启动所有客户端**（可以依次启动）
3. **等待训练完成**（10轮 × 5 epochs/轮 × 3客户端）

## 🛑 停止训练

### 优雅停止
在客户端按 `Ctrl+C`，客户端会完成当前epoch后退出。

### 强制停止
```bash
# 服务器
pkill -f "python server.py"

# 客户端
pkill -f "python client.py"
```

## 📂 训练结果

### 服务器
```
server_checkpoints/
├── global_round_1.pt
├── global_round_2.pt
└── ...
```

### 客户端
```
client1/runs_fed/
client2/runs_fed/
client3/runs_fed/
```

## 🔐 安全建议

1. **仅在受信任的内网环境中使用**
2. **不要暴露8080端口到公网**
3. **考虑使用VPN连接远程设备**
4. **定期备份训练checkpoint**

## 📞 获取帮助

如果遇到问题，请检查：
1. 所有设备的Python版本 (>=3.8)
2. 所有设备的依赖版本一致
3. 网络防火墙配置
4. 服务器和客户端日志输出

