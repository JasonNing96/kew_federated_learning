#!/bin/bash
# FLQ-Fed 状态查询脚本

echo "========================================================================"
echo "  FLQ-Fed 训练状态"
echo "========================================================================"

# 检查服务器状态
if curl -s http://localhost:8087/status >/dev/null 2>&1; then
    echo "📡 服务器状态:"
    curl -s http://localhost:8087/status | python -m json.tool
    echo ""
else
    echo "❌ 服务器未运行 (端口 8087 无响应)"
    echo ""
fi

# 检查进程
echo "🔧 运行中的进程:"
ps aux | grep -E "(app\.(server|client|runner)|flq-fed\.py)" | grep -v grep || echo "  无相关进程"
echo ""

# 检查端口占用
echo "🌐 端口占用:"
lsof -i:8087 2>/dev/null || echo "  端口 8087 未被占用"
echo ""

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU 状态:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
    echo ""
fi

echo "========================================================================"

