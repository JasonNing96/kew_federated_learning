# 更新日志

## [2.0-simplified] - 2025-11-15

### 🎯 重大重构：简化架构

**核心改进**：
- 从分散的 10+ 个文件简化为 5 个核心文件（~1000 行）
- 线性化代码流程，便于阅读和调试
- 统一日志输出到 `outputs/` 目录
- 新增便捷脚本（一键启动/停止/状态查询）

### ✨ 新增

- **app/** 目录 - 精简核心模块
  - `runner.py` - 统一命令行入口
  - `server.py` - 集中的服务器逻辑
  - `client.py` - 线性化的客户端流程
  - `model_utils.py` - 整合的工具函数
  - `config.py` - 简化的配置加载

- **scripts/** 目录 - 便捷脚本
  - `run_fl.sh` - 一键启动训练
  - `stop_fl.sh` - 停止所有进程
  - `status.sh` - 查看训练状态
  - `test_setup.py` - 架构测试脚本

- **文档**
  - `README.md` - 重写项目说明
  - `QUICKSTART.md` - 快速开始指南
  - `MIGRATION.md` - 迁移指南
  - `docs/PROJECT_STRUCTURE.md` - 详细结构说明

### 🔄 变更

- 移动 `core/` → `legacy/core/` （旧版归档）
- 移动 `flq_modules/` → `legacy/flq_modules/`（旧版归档）
- 移动 `flq-fed.py` → `legacy/flq-fed.py`（旧版入口）
- 更新配置文件增加 `workers`、`enable_val`、`enable_plots` 参数

### 🐛 修复

- 修复多客户端并发时 DataLoader 进程过多导致卡住的问题（设置 `workers: 0`）
- 修复端口占用检测和自动清理
- 修复日志输出不实时的问题
- 修复客户端训练进度不可见的问题

### 🗑️ 移除

- 移除冗余的中间层抽象
- 移除分散的配置访问方式
- 移除复杂的进程通信逻辑

### 📊 性能优化

- 减少 DataLoader worker 数量（默认 0）
- 可选关闭验证和绘图（`enable_val: false`, `enable_plots: false`）
- 优化日志输出性能

---

## [1.0] - 2025-11-14

### ✨ 初始版本

- 基础联邦学习框架
- YOLO 目标检测支持
- 量化压缩（1/4/8 bit）
- FedAvg 聚合算法
- FastAPI 服务器
- 多客户端支持

---

## 版本说明

### 版本号规则

- **主版本号** (2.x): 重大架构变更
- **次版本号** (x.0): 新功能添加
- **修订号** (x.x.1): Bug修复

### 标签说明

- 🎯 重大重构
- ✨ 新增功能
- 🔄 变更
- 🐛 Bug修复
- 🗑️ 移除
- 📊 性能优化
- 📚 文档更新
- ⚠️  重要提示

