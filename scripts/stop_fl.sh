#!/bin/bash
# FLQ-Fed 停止脚本 - 清理所有训练进程

echo "🛑 停止 FLQ-Fed 训练进程..."

# 停止所有相关进程
pkill -f "app.server" 2>/dev/null && echo "  ✅ 停止 server 进程"
pkill -f "app.client" 2>/dev/null && echo "  ✅ 停止 client 进程"
pkill -f "flq-fed.py" 2>/dev/null && echo "  ✅ 停止旧版进程"
# 注意: 不停止 app.runner，因为用户可能在使用它

# 检查 8087 端口
if lsof -i:8087 >/dev/null 2>&1; then
    echo "  ⚠️  端口 8087 仍被占用"
    lsof -i:8087
else
    echo "  ✅ 端口 8087 已释放"
fi

echo ""
echo "✅ 清理完成"

