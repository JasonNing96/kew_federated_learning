#!/bin/bash
# FLQ-Fed 一键启动脚本

set -e

cd "$(dirname "$0")/.."

echo "========================================================================"
echo "  FLQ-Fed 联邦学习训练"
echo "========================================================================"

# 检查 Python 环境
if ! command -v python &> /dev/null; then
    echo "❌ Python 未找到"
    exit 1
fi

# 检查配置文件
CONFIG="${1:-configs/flq_config.yaml}"
if [ ! -f "$CONFIG" ]; then
    echo "❌ 配置文件不存在: $CONFIG"
    echo "💡 用法: $0 [config_file]"
    exit 1
fi

echo "📋 配置文件: $CONFIG"
echo ""

# 启动训练
python -m app.runner train --config "$CONFIG"

echo ""
echo "========================================================================"
echo "  训练完成"
echo "========================================================================"
echo "📁 查看结果:"
echo "   - 服务器日志: outputs/server/logs/server.log"
echo "   - 客户端日志: outputs/client*/logs/client*.log"
echo "   - 全局模型: outputs/server/checkpoints/"
echo "   - 训练结果: outputs/client*/runs/"
echo "========================================================================"

