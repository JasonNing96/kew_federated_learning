#!/bin/bash
# FLQ-Fed 单节点快速测试脚本

set -e

cd "$(dirname "$0")/.."

echo "========================================================================"
echo "  FLQ-Fed 单节点快速测试"
echo "========================================================================"
echo ""
echo "💡 本脚本将在本地顺序运行 server 和 clients"
echo "   - Server 运行在后台线程"
echo "   - Clients 依次执行（避免资源竞争）"
echo "   - 配置: 1轮 × 3客户端 × 1epoch"
echo ""
echo "========================================================================"
echo ""

# 清理旧进程
echo "🧹 清理旧进程..."
pkill -f "app.server" 2>/dev/null || true
pkill -f "app.client" 2>/dev/null || true
sleep 1

# 检查 Python 环境
PYTHON="${PYTHON:-python}"

# 检查 ultralytics 是否可用
if ! $PYTHON -c "import ultralytics" 2>/dev/null; then
    echo "❌ 当前 Python 环境缺少 ultralytics"
    echo "💡 请先激活正确的环境: conda activate fedllm"
    exit 1
fi

# 运行测试
echo "🚀 开始测试..."
echo ""

$PYTHON -m app.local_train --config configs/local_test.yaml --clients 3

echo ""
echo "========================================================================"
echo "  测试完成"
echo "========================================================================"
echo ""
echo "📊 检查结果:"
echo "   1. 查看全局模型:"
echo "      ls -lh outputs/server/checkpoints/"
echo ""
echo "   2. 查看客户端训练结果:"
echo "      ls outputs/client1/runs/"
echo ""
echo "   3. 查看日志:"
echo "      tail outputs/server/logs/server.log"
echo "      tail outputs/client1/logs/client1.log"
echo ""
echo "========================================================================"

