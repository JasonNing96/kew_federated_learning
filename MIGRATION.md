# 迁移指南：从旧版到简化版

## 📋 概览

本次重构将项目从复杂的多文件结构简化为易于理解的 4 个核心文件。

## 🔄 主要变更

### 文件结构对比

| 旧版 | 新版 | 说明 |
|------|------|------|
| `flq-fed.py` | `app/runner.py` | 统一入口，功能相同 |
| `core/server.py` | `app/server.py` | 服务器逻辑，简化并集中 |
| `core/client.py` | `app/client.py` | 客户端逻辑，线性流程 |
| `flq_modules/*` | `app/model_utils.py` | 工具函数，整合为单文件 |
| `flq_modules/config.py` | `app/config.py` | 配置加载，简化访问 |
| 无 | `scripts/*.sh` | 新增便捷脚本 |

### 命令变更

| 旧版命令 | 新版命令 |
|---------|---------|
| `python flq-fed.py train` | `python -m app.runner train` 或 `./scripts/run_fl.sh` |
| `python flq-fed.py server` | `python -m app.runner server` |
| `python flq-fed.py client --id 1` | `python -m app.runner client --id 1` |
| 无 | `./scripts/stop_fl.sh` (新增) |
| 无 | `./scripts/status.sh` (新增) |

## 🎯 核心改进

### 1. 文件数量大幅减少

**旧版** (分散在多个目录):
- `core/` - 2 个文件
- `flq_modules/` - 5+ 个文件
- `tests/` - 多个测试文件
- 根目录 - 多个脚本

**新版** (集中在 app/):
- `app/runner.py` - 入口 (280 行)
- `app/server.py` - 服务器 (260 行)
- `app/client.py` - 客户端 (220 行)
- `app/model_utils.py` - 工具 (180 行)
- `app/config.py` - 配置 (100 行)

**总计**: ~1000 行核心代码，结构清晰

### 2. 逻辑流程线性化

**旧版** `core/client.py`:
- 函数分散定义
- 全局变量隐藏
- 需要跳转多次才能理解流程

**新版** `app/client.py`:
- 从上到下阅读即可
- 主流程一目了然:
  ```python
  def start_client(...):
      # 初始化
      while True:
          pull_global_model()  # 1. 拉取
          train_local()         # 2. 训练
          push_update()         # 3. 上传
  ```

### 3. 配置访问简化

**旧版**:
```python
from flq_modules.config import FLQConfig
config = FLQConfig("configs/flq_config.yaml")
rounds = config.training['rounds']  # 字典访问
```

**新版**:
```python
from app.config import Config
config = Config("configs/flq_config.yaml")
rounds = config.rounds  # 属性访问，IDE 自动补全
```

### 4. 新增便捷脚本

- `scripts/run_fl.sh` - 一键启动
- `scripts/stop_fl.sh` - 一键停止
- `scripts/status.sh` - 状态查询

## 📦 迁移步骤

### 如果你有自定义代码

1. **备份旧代码**
   ```bash
   cp -r core/ core_backup/
   cp -r flq_modules/ flq_modules_backup/
   ```

2. **查找依赖**
   ```bash
   # 查找项目中使用旧模块的地方
   grep -r "from core" .
   grep -r "from flq_modules" .
   ```

3. **更新导入**
   ```python
   # 旧版
   from core.server import start_fl_server
   from flq_modules.utils import model_to_vector
   
   # 新版
   from app.server import start_server
   from app.model_utils import model_to_vector
   ```

### 如果你使用标准流程

直接使用新版即可，命令基本兼容。

## 🔍 调试指南

### 旧版调试路径

查找问题需要跳转多个文件:
1. `flq-fed.py` → `core/server.py` → `flq_modules/aggregation.py` → ...

### 新版调试路径

所有逻辑集中，容易定位:
1. 问题在服务器? → 打开 `app/server.py`
2. 问题在客户端? → 打开 `app/client.py`
3. 问题在工具函数? → 打开 `app/model_utils.py`

### 日志位置

**统一输出目录**:
```
outputs/
├── server/logs/server.log
├── client1/logs/client1.log
├── client2/logs/client2.log
└── client3/logs/client3.log
```

## 🚨 Breaking Changes

### 1. 模块路径变更

所有导入从 `core.*` 和 `flq_modules.*` 改为 `app.*`

### 2. 配置文件增强

新增客户端配置项:
```yaml
client:
  workers: 0           # 新增
  enable_val: false    # 新增
  enable_plots: false  # 新增
```

### 3. 命令行参数

- `python flq-fed.py` → `python -m app.runner`
- 参数名保持不变

## ✅ 兼容性保证

### 保持不变

1. **配置文件格式** - `configs/flq_config.yaml` 保持兼容
2. **数据目录** - `data/client*/` 路径不变
3. **输出目录** - `outputs/` 结构不变
4. **API 端点** - `/global`, `/update`, `/status` 不变

### 需要更新

1. **导入语句** - 从 `core.*` 改为 `app.*`
2. **启动命令** - 使用新的脚本或模块路径
3. **测试代码** - 更新模块导入

## 💡 最佳实践

### 代码组织

- 服务器逻辑 → 全部放 `app/server.py`
- 客户端逻辑 → 全部放 `app/client.py`
- 工具函数 → 全部放 `app/model_utils.py`

### 调试流程

1. 查看日志 → `tail -f outputs/*/logs/*.log`
2. 检查状态 → `./scripts/status.sh`
3. 定位代码 → 只需查看 4 个核心文件

### 性能优化

- 设置 `workers: 0` 避免多进程竞争
- 关闭 `enable_val` 和 `enable_plots` 加快训练
- 减少 `local_epochs` 快速验证

## 📞 获取帮助

遇到迁移问题？

1. 查看 [QUICKSTART.md](QUICKSTART.md)
2. 查看 [README.md](README.md)
3. 查看 [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md)
4. 提交 Issue

## 🎉 迁移完成

完成迁移后，你的项目将:

- ✅ 结构更清晰（4 个核心文件）
- ✅ 代码更易读（线性流程）
- ✅ 调试更简单（集中管理）
- ✅ 启动更方便（便捷脚本）
- ✅ 维护更轻松（减少跳转）

