# 快速开始指南

## 🎯 5分钟上手

### 1. 环境准备

```bash
# 激活 Python 环境
conda activate fedllm

# 确认依赖已安装
pip install -r requirements.txt

# 检查 CUDA
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

### 2. 数据准备（首次使用）

```bash
# 切分数据集为3个客户端
python scripts/split_dataset.py --clients 3

# 确认数据已生成
ls data/client*/oil.yaml
```

### 3. 开始训练

```bash
# 一键启动（最简单）
./scripts/run_fl.sh

# 完成后查看结果
ls outputs/server/checkpoints/
ls outputs/client1/runs/
```

## 📊 查看状态

### 实时监控

```bash
# 终端1: 服务器日志
tail -f outputs/server/logs/server.log

# 终端2: 客户端日志
tail -f outputs/client1/logs/client1.log

# 终端3: GPU监控
watch -n 1 nvidia-smi

# 终端4: 训练状态
./scripts/status.sh
```

### API 查询

```bash
# 服务器状态
curl http://localhost:8087/status | python -m json.tool

# 示例输出:
{
    "current_round": 2,
    "total_rounds": 5,
    "training_done": false,
    "buffered_updates": 1,
    "clients_per_round": 3,
    "waiting_for": 2
}
```

## 🛑 停止训练

```bash
# 方式1: 使用脚本（推荐）
./scripts/stop_fl.sh

# 方式2: 手动停止
pkill -f "app.runner"

# 方式3: Ctrl+C（如果在前台运行）
```

## ⚙️ 自定义配置

编辑 `configs/flq_config.yaml`:

```yaml
training:
  rounds: 10              # 增加训练轮数
  local_epochs: 2         # 增加本地epoch

client:
  batch_size: 16          # 调整batch size
  enable_val: true        # 启用验证
```

然后重新训练:

```bash
./scripts/run_fl.sh configs/flq_config.yaml
```

## 🐛 常见问题

### 端口被占用

```bash
# 问题: ERROR: address already in use
# 解决:
./scripts/stop_fl.sh
```

### 训练无输出

```bash
# 检查日志文件
cat outputs/server/logs/server.log
cat outputs/client1/logs/client1.log

# 查看进程状态
ps aux | grep "app.runner"
```

### GPU 不工作

```bash
# 检查配置
grep device configs/flq_config.yaml

# 应该显示: device: "cuda:0"

# 检查 CUDA
nvidia-smi
```

## 📁 输出文件说明

```
outputs/
├── server/
│   ├── checkpoints/
│   │   └── global_round_N.pt     # 全局模型（每轮保存）
│   └── logs/
│       └── server.log            # 服务器日志
│
└── client1/
    ├── runs/
    │   └── round_N/
    │       ├── weights/
    │       │   └── best.pt       # 最佳模型
    │       ├── results.csv       # 训练指标
    │       └── *.png             # 可视化图片
    └── logs/
        └── client1.log           # 客户端日志
```

## 🚀 下一步

1. **调整超参数**: 编辑 `configs/flq_config.yaml`
2. **启用量化**: 设置 `quantization.enabled: true`
3. **查看文档**: 阅读 `docs/CHECKPOINTS.md`
4. **运行测试**: `python tests/test_quantization.py`

## 💡 提示

- 首次训练建议使用默认配置（1轮×3客户端）
- 设置 `workers: 0` 避免多进程冲突
- 关闭 `enable_val` 和 `enable_plots` 可加快训练
- 使用 `./scripts/status.sh` 随时查看状态
- 日志文件会实时更新，可用 `tail -f` 查看

## 📞 获取帮助

```bash
# 查看命令帮助
python -m app.runner --help
python -m app.runner train --help

# 查看脚本用法
./scripts/run_fl.sh --help
```

