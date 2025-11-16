# FLQ-Fed 标准项目结构

## 🎯 设计原则

1. **路径固定化** - 所有路径都基于项目根目录，不再使用相对路径猜测
2. **分类清晰** - 代码、数据、模型、输出严格分离
3. **避免重复** - 所有资源只保留一份
4. **便于部署** - 结构简单，易于打包和发布

## 📂 标准目录结构

```
kew_federated_learning/          # 项目根目录
│
├── flq-fed.py                   # ⭐ 主入口（唯一启动脚本）
├── README.md                    # 项目主文档
├── requirements.txt             # Python依赖
├── .gitignore                   # Git忽略配置
│
├── configs/                     # 📋 配置文件
│   └── flq_config.yaml          # 训练/量化配置
│
├── core/                        # 🔧 核心代码
│   ├── __init__.py
│   ├── server.py                # 联邦学习服务器
│   └── client.py                # 联邦学习客户端
│
├── flq_modules/                 # 📦 量化模块
│   ├── __init__.py
│   ├── quantization.py          # 量化算法
│   ├── aggregation.py           # 聚合算法
│   ├── utils.py                 # 工具函数
│   └── config.py                # 配置加载器
│
├── models/                      # 🤖 模型文件目录
│   ├── yolov8n.pt               # ⭐ 预训练模型（从fed_oil_demo移过来）
│   ├── yolov8s.pt               # 其他模型（可选）
│   └── README.md                # 模型说明
│
├── data/                        # 📊 数据集目录
│   ├── oil.yaml                 # ⭐ 数据集配置（主配置）
│   ├── client1/                 # 客户端1数据
│   │   ├── oil.yaml             # 客户端1配置（指向本地数据）
│   │   └── dataset/             # 实际数据
│   │       ├── images/
│   │       └── labels/
│   ├── client2/                 # 客户端2数据
│   │   ├── oil.yaml
│   │   └── dataset/
│   ├── client3/                 # 客户端3数据
│   │   ├── oil.yaml
│   │   └── dataset/
│   └── README.md                # 数据说明
│
├── outputs/                     # 📁 所有输出统一存放
│   ├── server/                  # 服务器输出
│   │   ├── checkpoints/         # 全局模型checkpoint
│   │   └── logs/                # 服务器日志
│   ├── client1/                 # 客户端1输出
│   │   ├── runs/                # 训练结果
│   │   └── logs/                # 客户端日志
│   ├── client2/                 # 客户端2输出
│   ├── client3/                 # 客户端3输出
│   └── experiments/             # 实验记录
│
├── scripts/                     # 🛠️ 辅助脚本
│   ├── split_dataset.py         # 数据切分工具
│   ├── visualize_results.py     # 结果可视化
│   └── cleanup.sh               # 清理临时文件
│
├── tests/                       # ✅ 单元测试
│   ├── test_quantization.py
│   ├── test_aggregation.py
│   └── test_utils.py
│
├── docs/                        # 📚 文档
│   ├── CHECKPOINTS.md           # 开发检查点
│   ├── API.md                   # API文档
│   └── DEPLOYMENT.md            # 部署指南
│
└── legacy/                      # 🗄️ 旧代码归档（不参与运行）
    ├── flq_fed_v4.py            # 原始FLQ代码
    ├── fed_oil_demo/            # 原始demo（参考用）
    └── README.md                # 归档说明

```

## 🔑 关键设计

### 1. 固定路径配置

所有代码使用项目根目录作为基准：

```python
# core/server.py, core/client.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 固定路径（不再动态查找）
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
```

### 2. 配置文件路径

```yaml
# configs/flq_config.yaml
model:
  pretrained: "models/yolov8n.pt"          # 固定路径
  
data:
  base_config: "data/oil.yaml"             # 主配置
  client_dirs: "data/client{id}/"          # 客户端数据模板

output:
  server_checkpoint: "outputs/server/checkpoints/"
  server_log: "outputs/server/logs/server.log"
  client_dir: "outputs/client{id}/"
```

### 3. 输出组织

```
outputs/
├── server/
│   ├── checkpoints/
│   │   ├── global_round_1.pt      # 按轮次保存
│   │   └── global_round_2.pt
│   └── logs/
│       └── server.log              # 统一日志
├── client1/
│   ├── runs/                       # YOLO训练输出
│   │   ├── round_0/
│   │   └── round_1/
│   └── logs/
│       └── client1.log
└── experiments/
    └── 2025-11-14_fp32_baseline.log  # 实验记录
```

## 🚀 迁移步骤

### 第1步：创建标准目录

```bash
mkdir -p models data outputs/server/{checkpoints,logs} scripts docs legacy
mkdir -p outputs/{client1,client2,client3}/{runs,logs}
```

### 第2步：移动模型文件

```bash
# 复制预训练模型到统一位置
cp fed_oil_demo/yolov8n.pt models/
cp fed_oil_demo/yolov8s.pt models/  # 可选
```

### 第3步：移动数据文件

```bash
# 移动数据集
mv fed_oil_demo/client1 data/
mv fed_oil_demo/client2 data/
mv fed_oil_demo/client3 data/
cp oil.yaml data/  # 主配置
```

### 第4步：移动辅助脚本

```bash
mv fed_oil_demo/split_dataset.py scripts/
mv plot_flq_fed.py scripts/visualize_results.py
```

### 第5步：归档旧代码

```bash
mkdir -p legacy
mv flq_fed_v3.py flq_fed_v4.py test_v4.py legacy/
mv fed_oil_demo legacy/  # 整个目录归档
```

### 第6步：更新所有代码路径

修改 `core/server.py`, `core/client.py`, `flq-fed.py` 中的路径引用。

## 📋 路径映射表

| 旧路径 | 新路径 | 说明 |
|--------|--------|------|
| `yolov8n.pt` / `fed_oil_demo/yolov8n.pt` | `models/yolov8n.pt` | 预训练模型 |
| `oil.yaml` | `data/oil.yaml` | 数据集主配置 |
| `fed_oil_demo/client1/` | `data/client1/` | 客户端1数据 |
| `client1/oil.yaml` | `data/client1/oil.yaml` | 客户端配置 |
| `server_checkpoints/` | `outputs/server/checkpoints/` | 全局模型 |
| `client1/runs_fed/` | `outputs/client1/runs/` | 客户端训练输出 |
| `logs/` | `outputs/server/logs/` 或 `outputs/client1/logs/` | 日志 |
| `split_dataset.py` | `scripts/split_dataset.py` | 工具脚本 |
| `flq_fed_v4.py` | `legacy/flq_fed_v4.py` | 旧代码参考 |

## ✅ 验证清单

- [ ] 所有模型文件在 `models/` 目录
- [ ] 所有数据文件在 `data/` 目录
- [ ] 所有输出写入 `outputs/` 目录
- [ ] 工具脚本在 `scripts/` 目录
- [ ] 旧代码归档到 `legacy/` 目录
- [ ] 配置文件使用固定路径（不再动态查找）
- [ ] 测试运行 `python flq-fed.py train` 成功
- [ ] 无路径错误、无重复下载

## 🎯 优势

1. **清晰** - 目录用途一目了然
2. **稳定** - 路径固定，不会因工作目录变化出错
3. **易维护** - 代码、数据、输出分离
4. **易部署** - 复制整个项目即可使用
5. **易清理** - `rm -rf outputs/*` 即可清空所有输出

---

**创建时间**: 2025-11-15 00:15  
**状态**: 设计完成，待实施

