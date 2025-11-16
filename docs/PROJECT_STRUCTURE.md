# 项目结构说明

## 📂 目录树

```
kew_federated_learning/
│
├── 📱 app/                          # 核心应用（仅5个文件）
│   ├── __init__.py                 #   包初始化
│   ├── runner.py                   #   命令行入口（280行）
│   ├── server.py                   #   联邦服务器（260行）
│   ├── client.py                   #   客户端逻辑（220行）
│   ├── model_utils.py              #   工具函数（180行）
│   └── config.py                   #   配置加载（100行）
│
├── ⚙️ configs/                      # 配置文件
│   └── flq_config.yaml             #   主配置（所有参数）
│
├── 🛠️ scripts/                      # 便捷脚本
│   ├── run_fl.sh                   #   一键启动训练
│   ├── stop_fl.sh                  #   停止所有进程
│   ├── status.sh                   #   查看训练状态
│   └── split_dataset.py            #   数据集切分
│
├── 🤖 models/                       # 预训练模型
│   └── yolov8n.pt                  #   YOLO基础模型
│
├── 📊 data/                         # 数据集
│   ├── oil.yaml                    #   主数据配置
│   ├── client1/                    #   客户端1数据
│   ├── client2/                    #   客户端2数据
│   └── client3/                    #   客户端3数据
│
├── 📁 outputs/                      # 所有输出（自动生成）
│   ├── server/
│   │   ├── checkpoints/            #   全局模型
│   │   └── logs/                   #   服务器日志
│   ├── client1/
│   │   ├── runs/                   #   训练结果
│   │   └── logs/                   #   客户端日志
│   ├── client2/ (同上)
│   └── client3/ (同上)
│
├── ✅ tests/                        # 单元测试
│   ├── test_quantization.py
│   ├── test_aggregation.py
│   └── test_utils.py
│
├── 📚 docs/                         # 文档
│   ├── CHECKPOINTS.md              #   开发检查点
│   ├── PROJECT_STRUCTURE.md        #   本文档
│   └── API.md                      #   API文档
│
├── 🗄️ legacy/                       # 旧版代码（归档）
│   ├── core/                       #   旧版核心模块
│   ├── flq_modules/                #   旧版量化模块
│   ├── fed_oil_demo/               #   旧版示例
│   └── flq-fed.py                  #   旧版入口
│
├── 📄 README.md                     # 项目说明
├── 📄 QUICKSTART.md                 # 快速开始
├── 📄 MIGRATION.md                  # 迁移指南
└── 📄 requirements.txt              # 依赖列表
```

## 🎯 核心文件说明

### app/runner.py

**职责**: 统一命令行入口

**功能**:
- 解析命令参数 (`server`, `client`, `train`)
- 进程管理（启动/停止/监控）
- 端口检查和清理
- 日志重定向

**关键函数**:
```python
def start_server_mode(config_path)      # 启动服务器
def start_client_mode(client_id, ...)   # 启动客户端
def train_full_mode(config_path, ...)   # 完整训练流程
def kill_existing_processes()           # 清理旧进程
```

### app/server.py

**职责**: 联邦学习服务器

**功能**:
- 维护全局模型
- 接收客户端更新
- FedAvg 聚合
- Checkpoint 保存

**关键类/函数**:
```python
class ServerState:                      # 服务器状态管理
    def add_update(...)                 # 添加客户端更新
    def _aggregate_and_advance()        # 聚合并推进轮次

def create_app(config, model)           # 创建 FastAPI 应用
    @app.get("/global")                 # 下发全局模型
    @app.post("/update")                # 接收客户端更新
    @app.get("/status")                 # 查询训练状态
```

### app/client.py

**职责**: 联邦学习客户端

**功能**:
- 拉取全局模型
- 本地训练
- 上传更新

**关键函数**:
```python
def pull_global_model(model)            # 下载全局模型
def train_local(model, round_id, cfg)   # 本地训练
def push_update(model, n_samples, ...)  # 上传本地更新
def start_client(client_id, ...)        # 客户端主循环
```

**执行流程**:
```
初始化 → while True:
    1. pull_global_model()   # 拉取
    2. train_local()         # 训练
    3. push_update()         # 上传
    4. 检查完成标志
```

### app/model_utils.py

**职责**: 核心工具函数

**功能模块**:

1. **模型转换**
   ```python
   model_to_vector(model)           # 模型 → 向量
   vector_to_model(vector, model)   # 向量 → 模型
   state_dict_to_vector(sd)         # state_dict → 向量
   vector_to_state_dict(vec, tpl)   # 向量 → state_dict
   ```

2. **量化/反量化**
   ```python
   quantize_vector(vec, bits)       # 向量量化
   dequantize_vector(q, scale, ...)  # 向量反量化
   ```

3. **聚合算法**
   ```python
   fedavg_aggregate(updates, weights)  # FedAvg聚合
   ```

4. **统计计算**
   ```python
   compute_model_size(sd, bits)     # 模型大小
   compute_compression_ratio(...)   # 压缩率
   ```

5. **误差反馈（可选）**
   ```python
   class ErrorFeedback:
       compress_with_feedback(...)  # 带误差反馈的压缩
   ```

### app/config.py

**职责**: 配置加载和访问

**特点**:
- 统一从 YAML 加载
- 属性访问（IDE 自动补全）
- 默认值处理

**用法**:
```python
config = Config("configs/flq_config.yaml")

# 训练参数
config.rounds                # 训练轮数
config.clients_per_round     # 客户端数

# 量化参数
config.quant_enabled         # 是否量化
config.quant_bits            # 量化比特

# 模型参数
config.model_name            # 模型路径
config.device                # 训练设备

# 客户端参数
config.batch_size            # 批大小
config.workers               # 进程数
```

## 🔄 数据流

### 训练流程

```
1. 服务器初始化
   └─> 加载基础模型 (models/yolov8n.pt)
   └─> 保存初始 global_state

2. 客户端启动
   └─> 拉取 global_state (/global)
   └─> 加载到本地模型

3. 本地训练
   └─> YOLO.train() 在本地数据上训练
   └─> 获得更新后的 state_dict

4. 上传更新
   └─> 序列化 state_dict
   └─> POST /update 发送到服务器

5. 服务器聚合
   └─> 收集 N 个客户端更新
   └─> FedAvg 加权平均
   └─> 更新 global_state
   └─> 保存 checkpoint

6. 下一轮
   └─> 客户端拉取新的 global_state
   └─> 重复 2-5
```

### 通信 API

```
GET  /                     # 服务器信息
GET  /status              # 训练状态
GET  /global              # 下载全局模型
POST /update              # 上传客户端更新
```

## 📊 配置文件结构

### configs/flq_config.yaml

```yaml
training:
  rounds: 5               # 训练轮数
  clients_per_round: 3    # 每轮客户端数
  local_epochs: 1         # 本地epoch数

quantization:
  enabled: false          # 量化开关
  bits: 8                 # 量化比特 (1/4/8/32)
  use_error_feedback: true

model:
  name: "models/yolov8n.pt"
  device: "cuda:0"

server:
  host: "0.0.0.0"
  port: 8087
  save_dir: "outputs/server/checkpoints"

client:
  batch_size: 8
  workers: 0              # DataLoader 进程数
  verbose: true
  enable_val: false       # 是否验证
  enable_plots: false     # 是否绘图
```

## 🛠️ 脚本说明

### scripts/run_fl.sh

一键启动完整训练:
```bash
./scripts/run_fl.sh [config_file]
```

### scripts/stop_fl.sh

停止所有训练进程:
```bash
./scripts/stop_fl.sh
```

### scripts/status.sh

查看训练状态:
```bash
./scripts/status.sh
```

输出内容:
- 服务器状态 (API)
- 运行中的进程
- 端口占用情况
- GPU 使用情况

### scripts/split_dataset.py

切分数据集:
```bash
python scripts/split_dataset.py --clients 3
```

## 📁 输出目录

### outputs/ 结构

```
outputs/
├── server/
│   ├── checkpoints/
│   │   ├── global_round_1.pt       # 第1轮全局模型
│   │   ├── global_round_2.pt       # 第2轮全局模型
│   │   └── ...
│   └── logs/
│       └── server.log              # 服务器完整日志
│
└── client1/
    ├── runs/
    │   ├── round_0/
    │   │   ├── weights/
    │   │   │   └── best.pt         # 最佳模型
    │   │   ├── results.csv         # 训练指标
    │   │   ├── results.png         # 指标曲线
    │   │   ├── confusion_matrix.png
    │   │   └── ...
    │   └── round_1/ (同上)
    └── logs/
        └── client1.log             # 客户端完整日志
```

## 🧪 测试目录

### tests/ 结构

```
tests/
├── test_quantization.py            # 量化模块测试
├── test_aggregation.py             # 聚合算法测试
├── test_utils.py                   # 工具函数测试
└── test_integration.py             # 集成测试
```

## 🗄️ Legacy 归档

旧版代码统一归档到 `legacy/`，包括:
- 旧版核心模块 (`core/`)
- 旧版量化模块 (`flq_modules/`)
- 旧版入口 (`flq-fed.py`)
- 旧版示例 (`fed_oil_demo/`)
- 旧版脚本

保留目的:
- 参考旧代码
- 对比新旧实现
- 向后兼容测试

## 📝 最佳实践

### 代码组织

1. **服务器逻辑** → 全部放 `app/server.py`
2. **客户端逻辑** → 全部放 `app/client.py`
3. **工具函数** → 全部放 `app/model_utils.py`
4. **配置管理** → 全部放 `app/config.py`
5. **入口调度** → 全部放 `app/runner.py`

### 文件原则

- **单一职责**: 每个文件只负责一个模块
- **自包含**: 函数定义和使用在同一文件
- **线性流程**: 从上到下阅读即可理解
- **最小依赖**: 减少模块间相互依赖

### 调试建议

1. 查看日志: `tail -f outputs/*/logs/*.log`
2. 检查状态: `./scripts/status.sh`
3. 定位代码: 只需查看 `app/` 下的 5 个文件
4. 单步调试: 在 `app/client.py` 的主循环设置断点

## 🔍 代码导航

遇到问题时的查找路径:

| 问题类型 | 查看文件 |
|---------|----------|
| 训练不启动 | `app/runner.py` → `train_full_mode()` |
| 客户端卡住 | `app/client.py` → `start_client()` |
| 模型未保存 | `app/server.py` → `_aggregate_and_advance()` |
| 量化错误 | `app/model_utils.py` → `quantize_vector()` |
| 配置无效 | `app/config.py` → 对应属性 |
| 端口被占用 | `app/runner.py` → `check_port_available()` |

总代码量: ~1000 行（不含注释和空行），清晰易懂！

