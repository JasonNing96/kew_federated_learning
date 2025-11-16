# FLQ-Fed 项目开发计划

## 🎯 最终目标

创建一个规范的、可发布的联邦学习量化框架，核心脚本为 `flq-fed.py`

## 📂 项目结构

```
kew_federated_learning/
├── flq-fed.py                 # 🎯 主脚本（联邦学习训练入口）
├── flq_modules/               # 核心模块目录
│   ├── __init__.py
│   ├── quantization.py        # 量化算法
│   ├── aggregation.py         # 聚合算法
│   ├── client.py              # 客户端逻辑
│   ├── server.py              # 服务器逻辑
│   └── utils.py               # 工具函数
├── tests/                     # 测试模块
│   ├── test_quantization.py
│   ├── test_aggregation.py
│   └── test_integration.py
├── configs/                   # 配置文件
│   ├── fedavg.yaml            # FedAvg基线配置
│   ├── flq_8bit.yaml          # 8-bit量化配置
│   └── flq_1bit.yaml          # 1-bit量化配置
├── experiments/               # 实验记录
│   └── baseline_results.md
├── CHECKPOINTS.md             # 开发检查点
├── README.md                  # 项目说明
└── requirements.txt           # 依赖列表

fed_oil_demo/                  # 原型代码（保留作为参考）
flq_fed_v4.py                  # 原始FLQ实现（参考）
```

## 📋 开发步骤（逐模块实现）

### Block 1: 量化模块 ✅
**文件**: `flq_modules/quantization.py`
**测试**: `tests/test_quantization.py`
**功能**:
- [ ] 相对域量化 `quantize_relative()`
- [ ] 随机舍入 `_stochastic_round()`
- [ ] 二值量化 `_binary_quantize()`
- [ ] LAQ量化 `quantize_laq()`

**验收标准**: 
- 单元测试通过
- 量化误差 < 1e-5
- 性能测试: 量化1M参数 < 100ms

---

### Block 2: 工具函数 ✅
**文件**: `flq_modules/utils.py`
**测试**: `tests/test_utils.py`
**功能**:
- [ ] 模型→向量 `model_to_vector()`
- [ ] 向量→模型 `vector_to_model()`
- [ ] 计算比特数 `compute_bits()`
- [ ] 参数形状提取 `get_param_shapes()`

**验收标准**:
- YOLO模型往返转换无损
- 支持多种模型架构

---

### Block 3: 聚合模块 ✅
**文件**: `flq_modules/aggregation.py`
**测试**: `tests/test_aggregation.py`
**功能**:
- [ ] FedAvg聚合 `fedavg_aggregate()`
- [ ] 量化感知聚合 `flq_aggregate()`
- [ ] 误差反馈 `ErrorFeedback` 类
- [ ] 懒惰选择 `LazySelector` 类

**验收标准**:
- 3客户端聚合结果正确
- 误差反馈有效（量化误差<10%）

---

### Block 4: 服务器模块 ✅
**文件**: `flq_modules/server.py`
**测试**: `tests/test_server.py`
**功能**:
- [ ] `FLQServer` 类
- [ ] 模型分发 `broadcast_model()`
- [ ] 接收更新 `collect_updates()`
- [ ] 统计输出 `print_stats()`

**验收标准**:
- 支持配置文件加载
- 统计输出完整

---

### Block 5: 客户端模块 ✅
**文件**: `flq_modules/client.py`
**测试**: `tests/test_client.py`
**功能**:
- [ ] `FLQClient` 类
- [ ] 本地训练 `train_local()`
- [ ] 量化上传 `upload_quantized()`
- [ ] 自动重连机制

**验收标准**:
- 单客户端训练正常
- 网络中断自动恢复

---

### Block 6: 主脚本集成 ✅
**文件**: `flq-fed.py`
**功能**:
- [ ] 参数解析
- [ ] 配置加载
- [ ] 服务器/客户端启动
- [ ] 日志记录

**验收标准**:
- 命令行友好
- 端到端训练成功

---

## 🧪 测试策略

每个Block完成后：
1. ✅ 运行单元测试
2. ✅ 记录测试结果到 `experiments/block_X_test.log`
3. ✅ 更新 CHECKPOINTS.md
4. ✅ Git commit（描述清晰）

## 📊 验收指标

最终系统需要达到：
- ✅ 所有单元测试通过
- ✅ 端到端训练成功（3客户端×10轮）
- ✅ 8-bit量化：通信降低75%，精度下降<2%
- ✅ 代码符合PEP8规范
- ✅ 文档完整（README + API文档）

---

**创建时间**: 2025-11-14 22:45
**预计完成**: 2025-11-21

