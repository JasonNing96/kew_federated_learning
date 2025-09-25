#!/bin/bash
# FLQ 联邦学习快速测试脚本
# 在单机上模拟分布式训练

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    FLQ 联邦学习快速测试${NC}"
echo -e "${BLUE}========================================${NC}"

# 检查 conda 环境
if [[ "$CONDA_DEFAULT_ENV" != "fed" ]]; then
    echo -e "${YELLOW}正在激活 conda fed 环境...${NC}"
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate fed
fi

echo -e "${GREEN}✓ Conda 环境: $CONDA_DEFAULT_ENV${NC}"

# 环境测试
echo -e "\n${BLUE}== 步骤 1: 环境检查 ==${NC}"
python test_environment.py
if [ $? -ne 0 ]; then
    echo -e "${RED}环境检查失败，请先修复环境问题${NC}"
    exit 1
fi

# 创建测试配置
echo -e "\n${BLUE}== 步骤 2: 创建测试配置 ==${NC}"
cat > test_config.json << 'EOF'
{
  "system": {
    "total_workers": 2,
    "min_workers": 2,
    "max_rounds": 20,
    "master_url": "http://127.0.0.1:5000"
  },
  "dataset": {
    "name": "mnist",
    "batch_size": 32,
    "partition": "iid",
    "dir_alpha": 0.1
  },
  "model": {
    "lr": 0.01,
    "l2": 0.0005,
    "local_epochs": 1
  },
  "flq_algorithm": {
    "mode": "laq8",
    "b_up": 8,
    "b_down": 8,
    "clip_global": 0.0,
    "scale_by_selected": true,
    "sel_ref": 1.0
  },
  "lazy_selection": {
    "D": 10,
    "ck": 0.8,
    "C": 50,
    "warmup": 5,
    "thr_scale": 0.0
  },
  "budget_selection": {
    "sel_clients": 0,
    "up_budget_bits": 0.0
  },
  "jetson_optimization": {
    "gpu_memory_limit_mb": 1024,
    "enable_mixed_precision": false,
    "prefetch_size": 2,
    "num_parallel_calls": 2
  }
}
EOF

echo -e "${GREEN}✓ 测试配置已创建: test_config.json${NC}"

# 创建结果和日志目录
mkdir -p test_results test_logs

# 启动 Master
echo -e "\n${BLUE}== 步骤 3: 启动 Master 节点 ==${NC}"
echo -e "${YELLOW}启动 Master (后台运行)...${NC}"

python flq_master.py \
    --config test_config.json \
    --host 127.0.0.1 \
    --port 5000 \
    > test_logs/master.log 2>&1 &

MASTER_PID=$!
echo -e "${GREEN}✓ Master 已启动 (PID: $MASTER_PID)${NC}"

# 等待 Master 启动
echo -e "${YELLOW}等待 Master 启动...${NC}"
sleep 3

# 检查 Master 是否正常运行
if ! curl -s http://127.0.0.1:5000/status >/dev/null; then
    echo -e "${RED}Master 启动失败，检查日志: test_logs/master.log${NC}"
    kill $MASTER_PID 2>/dev/null || true
    exit 1
fi

echo -e "${GREEN}✓ Master 运行正常${NC}"

# 启动 Workers
echo -e "\n${BLUE}== 步骤 4: 启动 Worker 节点 ==${NC}"

WORKER_PIDS=()

for worker_id in 0 1; do
    echo -e "${YELLOW}启动 Worker $worker_id...${NC}"
    
    python flq_worker.py \
        --config test_config.json \
        --worker_id $worker_id \
        > test_logs/worker${worker_id}.log 2>&1 &
    
    WORKER_PIDS+=($!)
    echo -e "${GREEN}✓ Worker $worker_id 已启动 (PID: ${WORKER_PIDS[$worker_id]})${NC}"
    sleep 1
done

# 清理函数
cleanup() {
    echo -e "\n${YELLOW}正在停止所有进程...${NC}"
    
    # 停止 Master
    if kill -0 $MASTER_PID 2>/dev/null; then
        kill $MASTER_PID
        echo -e "${GREEN}✓ Master 已停止${NC}"
    fi
    
    # 停止 Workers
    for pid in "${WORKER_PIDS[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            kill $pid
        fi
    done
    echo -e "${GREEN}✓ 所有 Workers 已停止${NC}"
    
    # 清理临时文件
    rm -f test_config.json
}

trap cleanup EXIT

# 监控训练过程
echo -e "\n${BLUE}== 步骤 5: 监控训练过程 ==${NC}"
echo -e "${YELLOW}训练进行中，按 Ctrl+C 停止...${NC}"

START_TIME=$(date +%s)
LAST_ROUND=-1

while true; do
    sleep 5
    
    # 获取状态
    STATUS=$(curl -s http://127.0.0.1:5000/status 2>/dev/null || echo "{}")
    CURRENT_ROUND=$(echo "$STATUS" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('current_round', -1))" 2>/dev/null || echo -1)
    CONNECTED_WORKERS=$(echo "$STATUS" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('connected_workers', 0))" 2>/dev/null || echo 0)
    
    if [[ $CURRENT_ROUND -gt $LAST_ROUND ]]; then
        ELAPSED=$(($(date +%s) - START_TIME))
        echo -e "${GREEN}Round $CURRENT_ROUND/20 完成，已用时 ${ELAPSED}s，连接 Workers: $CONNECTED_WORKERS${NC}"
        LAST_ROUND=$CURRENT_ROUND
    fi
    
    # 检查是否完成
    if [[ $CURRENT_ROUND -ge 20 ]]; then
        echo -e "${GREEN}✓ 训练完成！${NC}"
        break
    fi
    
    # 检查进程是否还在运行
    if ! kill -0 $MASTER_PID 2>/dev/null; then
        echo -e "${RED}Master 进程已停止，检查日志${NC}"
        break
    fi
done

# 获取最终结果
echo -e "\n${BLUE}== 步骤 6: 获取结果 ==${NC}"

# 停止训练并保存结果
curl -s -X POST http://127.0.0.1:5000/stop > /dev/null 2>&1 || true

sleep 2

# 检查结果文件
if ls flq_results_*.xlsx 1> /dev/null 2>&1; then
    RESULT_FILE=$(ls -t flq_results_*.xlsx | head -n1)
    mv "$RESULT_FILE" test_results/
    echo -e "${GREEN}✓ 结果已保存: test_results/$RESULT_FILE${NC}"
else
    echo -e "${YELLOW}⚠ 未找到结果文件${NC}"
fi

# 显示日志摘要
echo -e "\n${BLUE}== 日志摘要 ==${NC}"
echo -e "${YELLOW}Master 日志:${NC}"
tail -n 5 test_logs/master.log 2>/dev/null || echo "无日志"

echo -e "\n${YELLOW}Worker 0 日志:${NC}"
tail -n 3 test_logs/worker0.log 2>/dev/null || echo "无日志"

echo -e "\n${YELLOW}Worker 1 日志:${NC}"
tail -n 3 test_logs/worker1.log 2>/dev/null || echo "无日志"

# 总结
TOTAL_TIME=$(($(date +%s) - START_TIME))
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}🎉 测试完成！${NC}"
echo -e "${GREEN}总用时: ${TOTAL_TIME}s${NC}"
echo -e "${GREEN}结果目录: test_results/${NC}"
echo -e "${GREEN}日志目录: test_logs/${NC}"
echo -e "${BLUE}========================================${NC}"
