#!/bin/bash
# PyTorch版本FLQ快速测试脚本

echo "=== PyTorch版本FLQ-Federated MNIST快速测试 ==="

# 创建日志目录
mkdir -p logs

# 启动master
echo "1. 启动master..."
python master.py > logs/master_pytorch.log 2>&1 &
MASTER_PID=$!
echo "Master PID: $MASTER_PID"
sleep 3

# 测试基线模式
echo "2. 测试基线模式 (FLQ_MODE=off)..."
MASTER_ADDR=http://127.0.0.1:5000 \
WORKER_ID=worker_baseline_pytorch \
FLQ_MODE=off \
LOCAL_STEPS=2 \
python worker_mnist_pytorch_simple.py > logs/worker_baseline_pytorch.log 2>&1 &
BASELINE_PID=$!
sleep 8

# 测试sign1量化
echo "3. 测试sign1量化 (FLQ_MODE=sign1)..."
MASTER_ADDR=http://127.0.0.1:5000 \
WORKER_ID=worker_sign1_pytorch \
FLQ_MODE=sign1 \
LOCAL_STEPS=2 \
python worker_mnist_pytorch_simple.py > logs/worker_sign1_pytorch.log 2>&1 &
SIGN1_PID=$!
sleep 8

# 测试int8量化
echo "4. 测试int8量化 (FLQ_MODE=int8)..."
MASTER_ADDR=http://127.0.0.1:5000 \
WORKER_ID=worker_int8_pytorch \
FLQ_MODE=int8 \
LOCAL_STEPS=2 \
python worker_mnist_pytorch_simple.py > logs/worker_int8_pytorch.log 2>&1 &
INT8_PID=$!
sleep 8

# 测试Lloyd-Max量化
echo "5. 测试Lloyd-Max量化 (FLQ_MODE=lloyd-max)..."
MASTER_ADDR=http://127.0.0.1:5000 \
WORKER_ID=worker_lloydmax_pytorch \
FLQ_MODE=lloyd-max \
LOCAL_STEPS=2 \
python worker_mnist_pytorch_simple.py > logs/worker_lloydmax_pytorch.log 2>&1 &
LLOYD_PID=$!
sleep 8

# 检查状态
echo "6. 检查系统状态..."
echo "Master状态:"
curl -s http://127.0.0.1:5000/status | python -m json.tool || echo "无法获取状态"

echo ""
echo "Prometheus指标:"
curl -s http://127.0.0.1:5000/metrics | head -20

# 分析日志
echo ""
echo "7. 分析worker日志..."
for logfile in logs/worker_*_pytorch.log; do
    if [ -f "$logfile" ]; then
        echo "=== $logfile ==="
        tail -3 "$logfile"
        echo ""
    fi
done

# 清理
echo "8. 清理进程..."
kill $MASTER_PID $BASELINE_PID $SIGN1_PID $INT8_PID $LLOYD_PID 2>/dev/null || true
sleep 2
pkill -f "python.*master.py" 2>/dev/null || true
pkill -f "python.*worker" 2>/dev/null || true

echo ""
echo "=== PyTorch版本测试完成 ==="
echo "日志文件保存在 logs/ 目录"
echo "可以运行 python analyze_results.py 分析详细结果"