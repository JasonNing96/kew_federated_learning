# 快速开始指南

## 📦 准备工作

### 1. 安装依赖
```bash
cd fed_oil_demo
pip install -r requirements.txt
```

### 2. 检查环境
```bash
python check_environment.py
```

### 3. 切分数据集（首次使用必须）
```bash
python split_dataset.py
```

这会将完整数据集切分给3个客户端，每个客户端获得独立的训练数据子集。

## 🚀 运行训练

### 方式一：中心化训练（传统方法）

```bash
python train_gpu_only.py
# 或
./run_centralized.sh
```

结果保存在 `runs/oil_spill_gpu/`

### 方式二：联邦学习训练（单机测试）

**终端1 - 启动服务器：**
```bash
./start_server.sh
```

**终端2 - 启动客户端：**
```bash
./start_clients.sh
```

**监控训练：**
```bash
# 查看客户端日志
tail -f client1.log
tail -f client2.log
tail -f client3.log

# 查看服务器状态
curl http://localhost:8080/status
```

**停止训练：**
```bash
./stop_all.sh
```

### 方式三：联邦学习训练（多设备真实场景）

#### 在服务器设备上：

```bash
# 启动服务器
./start_server.sh

# 查看服务器IP
ip addr show | grep "inet "
# 假设得到: 192.168.1.100
```

#### 在客户端设备1上：

```bash
# 1. 复制必要文件到设备
# - client.py
# - yolov8n.pt
# - client1/ 目录

# 2. 设置服务器地址
export FL_SERVER="http://192.168.1.100:8080"

# 3. 启动客户端
CLIENT_ID=1 python client.py
```

#### 在客户端设备2上：

```bash
export FL_SERVER="http://192.168.1.100:8080"
CLIENT_ID=2 python client.py
```

#### 在客户端设备3上：

```bash
export FL_SERVER="http://192.168.1.100:8080"
CLIENT_ID=3 python client.py
```

## 📊 查看结果

### 中心化训练
```bash
cd runs/oil_spill_gpu/
ls weights/best.pt        # 最佳模型
ls results.png            # 训练曲线
```

### 联邦学习
```bash
# 全局聚合模型
ls server_checkpoints/

# 各客户端训练结果
ls client1/runs_fed/
ls client2/runs_fed/
ls client3/runs_fed/

# 训练日志
cat client1.log
```

## ⚙️ 快速调参

### 显存不足？
编辑 `client.py`:
```python
BATCH = 4  # 降低batch size
```

### 训练太慢？
编辑 `client.py`:
```python
EPR = 3    # 减少每轮epoch数
```

编辑 `server.py`:
```python
ROUNDS = 5  # 减少总轮数
```

### 改变客户端数量？
1. 编辑 `split_dataset.py`:
```python
NUM_CLIENTS = 5  # 改为5个客户端
```

2. 重新切分数据：
```bash
python split_dataset.py
```

3. 编辑 `server.py`:
```python
CLIENTS_PER_ROUND = 5
```

## 🐛 常见问题

### 1. 客户端连接不上服务器
```bash
# 检查服务器是否运行
curl http://服务器IP:8080/

# 检查防火墙（Ubuntu）
sudo ufw allow 8080/tcp

# 测试网络
ping 服务器IP
```

### 2. "client1/oil.yaml 不存在"
```bash
# 运行数据切分脚本
python split_dataset.py
```

### 3. CUDA out of memory
```bash
# 编辑client.py，降低BATCH大小
BATCH = 4  # 或更小
```

### 4. 服务器一直等待客户端
- 确保启动的客户端数量 = `CLIENTS_PER_ROUND`
- 检查客户端日志：`tail -f client*.log`

## 🎯 完整工作流示例

```bash
# 1. 环境准备
cd fed_oil_demo
pip install -r requirements.txt
python check_environment.py

# 2. 数据切分
python split_dataset.py

# 3. 启动联邦学习（新终端分别运行）
## 终端1
./start_server.sh

## 终端2
./start_clients.sh

# 4. 监控进度（新终端）
watch -n 5 "curl -s http://localhost:8080/status | jq"
tail -f client1.log

# 5. 等待训练完成（10轮 × 3个客户端）

# 6. 停止所有进程
./stop_all.sh

# 7. 查看结果
ls -lh server_checkpoints/
```

## 📚 更多信息

详细文档请查看 `README.md`
