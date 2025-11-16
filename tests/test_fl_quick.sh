#!/bin/bash
# FLQ-Fed 快速测试脚本

echo "========================================================================"
echo "  FLQ-Fed 快速测试 (1轮训练，3客户端)"
echo "========================================================================"
echo ""

# 进入项目根目录
cd "$(dirname "$0")/.."

# 设置Python环境（使用fedllm）
PYTHON=/home/njh/miniconda3/envs/fedllm/bin/python

# 检查环境
echo "📋 检查环境..."
echo "   Python: $($PYTHON --version)"
echo "   Ultralytics: $($PYTHON -c 'import ultralytics; print(ultralytics.__version__)' 2>/dev/null || echo '未安装')"
echo "   PyTorch: $($PYTHON -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '未安装')"
echo ""

# 确认继续
echo "🚀 将启动联邦学习训练（1轮×3客户端）"
echo "   - 配置: configs/flq_config.yaml"
echo "   - 端口: 8087"
echo "   - 设备: CUDA (客户端) + CPU (服务器)"
echo ""
echo "⏳ 3秒后开始..."
sleep 3

# 运行训练
echo ""
echo "========================================================================"
echo "  开始训练"
echo "========================================================================"
$PYTHON flq-fed.py train --config configs/flq_config.yaml

echo ""
echo "========================================================================"
echo "  测试完成"
echo "========================================================================"
echo "📂 查看结果:"
echo "   - 服务器日志: outputs/server/logs/server.log"
echo "   - 客户端日志: outputs/client1/logs/client1.log"
echo "   - 模型checkpoint: outputs/server/checkpoints/"
echo ""

