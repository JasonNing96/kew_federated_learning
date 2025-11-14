#!/bin/bash
# 启动3个联邦学习客户端

cd "$(dirname "$0")"

echo "=========================================="
echo "启动联邦学习客户端"
echo "=========================================="
echo "客户端数量: 3"
echo "服务器地址: http://127.0.0.1:8080"
echo "训练设备: GPU (device=0)"
echo "每轮训练: 5 epochs"
echo "=========================================="

# 检查数据分片是否存在
for i in 1 2 3; do
    if [ ! -f "client${i}/oil.yaml" ]; then
        echo "❌ 错误: client${i}/oil.yaml 不存在"
        echo "💡 请先运行: python split_dataset.py"
        exit 1
    fi
done

# 清空旧日志
echo ""
echo "🗑️  清空旧日志..."
> client1.log
> client2.log
> client3.log

# 启动3个客户端（后台运行，日志重定向，UTF-8编码）
echo ""
echo "🚀 启动客户端..."

PYTHONIOENCODING=utf-8 CLIENT_ID=1 python client.py > client1.log 2>&1 &
CLIENT1_PID=$!
echo "✓ 客户端 #1 已启动 (PID: $CLIENT1_PID)"

PYTHONIOENCODING=utf-8 CLIENT_ID=2 python client.py > client2.log 2>&1 &
CLIENT2_PID=$!
echo "✓ 客户端 #2 已启动 (PID: $CLIENT2_PID)"

PYTHONIOENCODING=utf-8 CLIENT_ID=3 python client.py > client3.log 2>&1 &
CLIENT3_PID=$!
echo "✓ 客户端 #3 已启动 (PID: $CLIENT3_PID)"

echo ""
echo "=========================================="
echo "✅ 所有客户端已启动"
echo "=========================================="
echo "进程ID: $CLIENT1_PID, $CLIENT2_PID, $CLIENT3_PID"
echo ""
echo "监控日志:"
echo "  tail -f client1.log"
echo "  tail -f client2.log"
echo "  tail -f client3.log"
echo ""
echo "停止所有客户端:"
echo "  pkill -f 'python client.py'"
echo "=========================================="

