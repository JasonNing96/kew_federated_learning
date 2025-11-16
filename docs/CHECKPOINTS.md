# FLQ-YOLO 开发检查点

## 🔧 环境

```bash
conda activate fedllm  # Python 3.9.16 + PyTorch 2.0.0+cu117
```

## 📋 开发进度

### ✅ 阶段0: 基线验证

- [x] **0.1** FLQ算法验证 - `python flq_fed_v4.py`
- [x] **0.2** 单节点YOLO - `python train_gpu_only.py`  
- [x] **0.3** 多节点联邦训练 - ✅ **已完成！** (2025-11-14 22:40)

**修复过程**:
1. 启用GPU: `DEVICE = 'cuda:0'` (原CPU)
2. 修复数据路径: `fed_oil_detection` → `kew_federated_learning`
3. 显示训练进度: `verbose=True`
4. **添加统计信息**：每轮显示 Tround, BitUp, BitDown, Clients, Params, Compress

**结果**: 
- 2轮训练完成，GPU峰值97%
- 模型保存在 `server_checkpoints/`
- 每轮统计：~0.29 Gbits上行 + 0.29 Gbits下行 = **0.58 Gbits/轮**

### ✅ 阶段1: 量化模块移植（1周）

- [x] **1.1** ✅ `flq_modules/quantization.py` - 量化函数（6/6测试通过）
- [x] **1.2** ✅ `flq_modules/utils.py` - 模型↔向量转换（9/9测试通过）
- [x] **1.3** ✅ `flq_modules/aggregation.py` - 聚合逻辑（9/9测试通过）

**完成时间**: 2025-11-14 23:00  
**测试总结**: 24/24全部通过 ✅

### ✅ 阶段2: 服务器集成（1周）

- [x] **2.1** ✅ 配置系统（`configs/flq_config.yaml` + `flq_modules/config.py`）
- [x] **2.2** ✅ 服务器集成（`server.py` 添加量化支持）
  - 配置加载正常
  - 统计输出显示压缩率  
  - FP32基线模式测试通过
- [x] **2.3** ✅ 端到端测试（FP32基线）
  - 1轮×3客户端训练成功
  - Checkpoint生成正常 (12MB)
  - 系统运行稳定

**完成时间**: 2025-11-14 23:10  
**状态**: 服务器集成完成，系统可正常运行 ✅

### 🔄 项目重构：统一入口（2025-11-14 23:20）

- [x] ✅ 创建统一入口 `flq-fed.py`
  - 支持 `server`, `client`, `train` 三种模式
  - 命令行参数驱动
  - 自动进程管理
- [x] ✅ 重新整理项目结构
  - `core/` - 核心服务器/客户端逻辑
  - `examples/fed_oil_demo/` - 示例代码（向后兼容）
  - `logs/` - 日志目录
- [x] ✅ 创建项目主README
  - 使用说明
  - 配置文档
  - 性能指标
- [x] ✅ 修复路径问题（2025-11-14 23:56）
  - 智能路径查找（支持多种目录结构）
  - 测试脚本验证通过
  - 所有启动方式正常工作

### 🏗️ 项目结构标准化（2025-11-15 00:20）⭐

**问题诊断**：
1. 文件散乱：模型、数据、输出混杂在多个目录
2. 路径混乱：需要"智能查找"才能定位文件
3. 资源重复：YOLO模型多处存在，重复下载
4. 难以维护：不利于部署和版本管理

**解决方案**：实施标准化目录结构（2025-11-15 17:30 更新：结构精简方案已经确认，核心文件 ≤5 个，具体如下）

```
kew_federated_learning/          # 项目根目录
│
├── flq-fed.py                   # ⭐ 主入口（唯一，后续将拆成 runner.py）
├── README.md                    # 项目文档
├── requirements.txt             # 依赖管理
│
├── configs/                     # 📋 配置
│   └── flq_config.yaml
│
├── app/                         # 🔧 核心代码（精简版）
│   ├── runner.py                # 统一入口（server/client/train 子命令）
│   ├── server.py                # FastAPI服务器 + 聚合
│   ├── client.py                # 客户端控制（拉取→训练→上传）
│   └── model_utils.py           # 公共工具（向量化、量化辅助）
│
├── flq_modules/                 # 📦 量化模块
│   ├── quantization.py
│   ├── aggregation.py
│   └── utils.py
│
├── models/                      # 🤖 预训练模型（⭐新增）
│   └── yolov8n.pt               # 统一存放，避免重复下载
│
├── data/                        # 📊 数据集（⭐新增）
│   ├── oil.yaml                 # 主配置
│   └── client1/client2/client3/ # 客户端数据
│
├── outputs/                     # 📁 所有输出（⭐新增）
│   ├── server/                  # 服务器输出
│   │   ├── checkpoints/         # 全局模型
│   │   └── logs/                # 服务器日志
│   └── client1/client2/client3/ # 客户端输出
│       ├── runs/                # 训练结果
│       └── logs/                # 客户端日志
│
├── scripts/                     # 🛠️ 工具脚本
│   ├── split_dataset.py         # 数据切分
│   ├── run_fl.sh                # 一键训练（调用 runner）
│   └── stop_fl.sh               # 清理残留进程 / 端口
│
├── tests/                       # ✅ 测试
├── docs/                        # 📚 文档（⭐新增）
└── legacy/                      # 🗄️ 旧代码归档（⭐新增）
```

**实施步骤**（精简版）：

1. `app/` 目录下创建四个核心脚本：`runner.py`、`server.py`、`client.py`、`model_utils.py`
2. 将现有 `core/*.py` 与 `flq-fed.py` 中的逻辑迁移/拆分到上述脚本
3. 更新 `flq-fed.py` 仅作为兼容入口（内部 `from app.runner import main`）
4. 调整 `scripts/run_fl.sh` / `stop_fl.sh` 以调用新的 runner
5. 执行 `python -m app.runner train --config configs/flq_config.yaml` 验证

**核心改进**：

1. **路径固定化** - 所有路径基于项目根目录，不再动态查找
2. **资源统一** - 模型、数据、输出各有专属目录
3. **避免重复** - YOLO模型只保留一份（`models/yolov8n.pt`）
4. **易于部署** - 目录结构清晰，易于打包发布
5. **便于维护** - 代码、数据、文档分离

**路径映射**：

| 旧路径 | 新路径 | 说明 |
|--------|--------|------|
| `fed_oil_demo/yolov8n.pt` | `models/yolov8n.pt` | 模型统一存放 |
| `oil.yaml` | `data/oil.yaml` | 数据配置 |
| `fed_oil_demo/client1/` | `data/client1/` | 客户端数据 |
| `server_checkpoints/` | `outputs/server/checkpoints/` | 全局模型 |
| `client1/runs_fed/` | `outputs/client1/runs/` | 训练输出 |
| `logs/` | `outputs/*/logs/` | 日志统一 |
| `split_dataset.py` | `scripts/split_dataset.py` | 工具脚本 |
| `flq_fed_v4.py` | `legacy/flq_fed_v4.py` | 旧代码归档 |

**完成标志**：

- ✅ 运行 `python flq-fed.py train` 无路径错误
- ✅ 不会重复下载已有的YOLO模型
- ✅ 所有输出统一写入 `outputs/` 目录
- ✅ 项目结构清晰，易于理解和维护

**详细文档**: 见 `docs/PROJECT_STRUCTURE_STANDARD.md`（待更新为精简版结构说明）

### ⏳ 阶段3: 客户端适配（1周）

- [ ] **3.1** 客户端支持量化上传
- [ ] **3.2** 端到端测试

---

## 🚀 快速命令

### 标准化结构后（⭐推荐）

```bash
# 完整训练（一键启动）
python flq-fed.py train --config configs/flq_config.yaml

# 或分别启动
python flq-fed.py server                    # 终端1
python flq-fed.py client --id 1             # 终端2
python flq-fed.py client --id 2             # 终端3
python flq-fed.py client --id 3             # 终端4

# 监控和调试
watch -n 1 nvidia-smi                       # GPU监控
tail -f outputs/server/logs/server.log      # 服务器日志
tail -f outputs/client1/logs/client1.log    # 客户端日志

# 运行测试
python tests/test_quantization.py          # 单元测试
bash scripts/cleanup.sh                     # 清理输出

# 数据准备
python scripts/split_dataset.py --clients 3
```

### 过渡期（临时兼容）

```bash
# 方式1：从fed_oil_demo目录运行（旧方式）
cd fed_oil_demo
./start_server.sh  
./start_clients.sh

# 方式2：从项目根目录运行（带路径查找）
python flq-fed.py train   # 会自动查找fed_oil_demo/client*/
```

**⚠️ 注意**: 完成结构标准化后，统一使用新路径（`data/`, `models/`, `outputs/`）

---

## 📊 目标指标

| 阶段 | 上行/轮 | 下行/轮 | 总通信/轮 | mAP50 | 压缩率 |
|------|--------|---------|----------|-------|--------|
| **基线(FedAvg)** | **0.29 Gbit** | **0.29 Gbit** | **0.58 Gbit** | 0.83 | 1.0x |
| 8-bit量化 | 0.07 Gbit | 0.07 Gbit | 0.14 Gbit | 0.82 | 4.0x ↓75% |
| 4-bit量化 | 0.04 Gbit | 0.07 Gbit | 0.11 Gbit | 0.80 | 5.3x ↓81% |
| 1-bit量化 | 0.01 Gbit | 0.07 Gbit | 0.08 Gbit | 0.78 | 7.3x ↓86% |
| 1-bit+懒惰 | 0.006 Gbit | 0.05 Gbit | 0.056 Gbit | 0.78 | 10.4x ↓90% |

**注**: 基于YOLO8n模型 (3,011,238参数 ≈ 12MB FP32)

---

## 📈 服务器统计输出示例

```
======================================================================
                        📊 Round 1 统计信息                           
======================================================================
  ⏱️  Tround    : 15.23 秒
  📤 BitUp     : 0.289 Gbits (36.1 MB)
  📥 BitDown   : 0.289 Gbits (36.1 MB)
  📊 BitTotal  : 0.578 Gbits
  👥 Clients   : 3/3
  📦 Params    : 3,011,238 (12.0 MB)
  🗜️  Compress  : 1.00x (FP32 baseline)
======================================================================
```

---

## 📌 当前状态

**最后更新**: 2025-11-15 20:00

**已完成**:
- ✅ 阶段0: 基线验证（多节点联邦训练）
- ✅ 阶段1: 量化模块移植（24/24测试通过）
- ✅ 阶段2: 服务器集成（FP32基线运行正常）
- ✅ 项目重构: 统一入口 `flq-fed.py`
- ✅ **项目架构简化** (2025-11-15 20:00) ⭐
  - 核心代码精简为 5 个文件（~1000 行）
  - 线性化流程，易于理解和调试
  - 新增便捷脚本（run/stop/status）
  - 完整文档（README/QUICKSTART/MIGRATION）
  - 所有测试通过 ✅

**待完成**:
- [ ] 阶段3: 客户端量化适配（启用量化训练）
- [ ] 阶段4: 性能测试（对比不同量化比特数）
- [ ] 阶段5: 论文撰写

**下一步**: 
1. 使用新架构进行完整训练测试: `./scripts/run_fl.sh`
2. 验证多轮训练稳定性
3. 准备启用量化功能
