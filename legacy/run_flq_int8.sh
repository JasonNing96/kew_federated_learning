#!/bin/bash
# FLQ-Federated MNIST int8量化实验脚本
# 运行5个worker，FLQ_MODE=int8（8位整数量化）

echo "=== 启动FLQ-Federated MNIST int8量化实验 ==="
echo "模式: int8量化 (FLQ_MODE=int8)"
echo "Worker数量: 5"
echo "本地步数: 5"
echo "批次大小: 64"

# 启动master
echo "启动master..."
python master.py &
MASTER_PID=$!
sleep 3

# 启动5个worker（int8量化模式）
echo "启动worker（int8量化模式）..."
for i in {1..5}; do
    WORKER_ID="worker_int8_$i"
    echo "启动 $WORKER_ID..."
    
    MASTER_ADDR=http://127.0.0.1:5000 \
    WORKER_ID=$WORKER_ID \
    FLQ_MODE=int8 \
    LOCAL_STEPS=5 \
    BATCH_SIZE=64 \
    LR=0.05 \
    python worker_mnist.py &
    
    sleep 0.5
done

echo ""
echo "=== 实验运行中 ==="
echo "监控master状态: curl http://127.0.0.1:5000/status"
echo "监控Prometheus指标: curl http://127.0.0.1:5000/metrics"
echo ""
echo "按 Ctrl+C 停止实验..."

# 等待中断
trap "echo '停止实验...'; kill $MASTER_PID; pkill -f worker_mnist.py; exit 0" INT
wait $MASTER_PID