# FLQ-Fed 文件结构说明

## 📂 顶层文件

| 文件 | 说明 | 用途 |
|------|------|------|
| `flq-fed.py` | 🎯 **主入口脚本** | 统一启动服务器/客户端/完整训练 |
| `README.md` | 项目主文档 | 快速开始、配置说明、性能指标 |
| `CHECKPOINTS.md` | 开发检查点 | 记录开发进度和状态 |
| `PROJECT_PLAN.md` | 项目计划 | 详细的开发计划和里程碑 |
| `REFACTOR_SUMMARY.md` | 重构总结 | 项目重构的详细说明 |

## 📦 核心模块

### `core/` - 核心服务器/客户端逻辑
- `server.py` - 联邦学习服务器（FastAPI）
- `client.py` - 联邦学习客户端（YOLO训练）
- `__init__.py` - 模块初始化

### `flq_modules/` - FLQ量化模块
- `quantization.py` - 量化算法（1/4/8-bit）
- `aggregation.py` - 聚合算法（FedAvg, 量化聚合）
- `utils.py` - 工具函数（模型转换、通信计算）
- `config.py` - 配置加载器
- `__init__.py` - 模块导出

### `configs/` - 配置文件
- `flq_config.yaml` - 默认配置（训练、量化、模型参数）

### `tests/` - 单元测试
- `test_quantization.py` - 量化模块测试（6个测试）
- `test_aggregation.py` - 聚合模块测试（9个测试）
- `test_utils.py` - 工具模块测试（9个测试）

### `examples/` - 示例代码
- `fed_oil_demo/` - 石油泄漏检测完整示例
  - 原始server.py, client.py（向后兼容）
  - 启动脚本、数据切分工具
  - 完整的README和文档

### `logs/` - 日志输出
- `server.log` - 服务器日志
- `client1.log, client2.log, ...` - 客户端日志

### `experiments/` - 实验记录
- `block_summary.md` - 模块开发总结
- `stage2_summary.md` - 阶段2总结
- `*.log` - 测试日志

## 🗂️ 其他目录

| 目录 | 说明 |
|------|------|
| `docker/` | Docker容器化部署（可选） |
| `federated_learning_jetson/` | Jetson设备部署（可选） |
| `data/` | 数据集目录（自动生成） |

## 🚀 使用流程

### 开发者视角

```
1. 阅读文档
   └── README.md（快速开始）
   └── CHECKPOINTS.md（了解进度）

2. 查看核心代码
   └── core/server.py, client.py
   └── flq_modules/（量化逻辑）

3. 运行测试
   └── tests/test_*.py

4. 配置训练
   └── configs/flq_config.yaml

5. 启动训练
   └── python flq-fed.py train
```

### 用户视角

```
1. 阅读README.md
2. 准备数据（examples/fed_oil_demo/split_dataset.py）
3. 修改配置（configs/flq_config.yaml）
4. 运行训练（python flq-fed.py train）
5. 查看结果（logs/, server_checkpoints/）
```

## 📌 重要文件说明

### 必须保留
- `flq-fed.py` - 主入口
- `core/` - 核心逻辑
- `flq_modules/` - 量化模块
- `configs/flq_config.yaml` - 配置文件

### 可选文件
- `examples/` - 示例代码（可根据需要删除）
- `tests/` - 测试代码（发布时可选）
- `experiments/` - 实验记录（开发用）

### 自动生成
- `logs/` - 运行时生成
- `server_checkpoints/` - 训练时生成
- `client*/runs_fed/` - 训练时生成

---

**更新时间**: 2025-11-14 23:25
