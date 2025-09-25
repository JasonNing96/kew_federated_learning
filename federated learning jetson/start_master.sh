#!/bin/bash
# FLQ Master 启动脚本 - Jetson Nano 优化版本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    FLQ 联邦学习 Master 启动脚本${NC}"
echo -e "${BLUE}========================================${NC}"

# 检查 conda 环境
if [[ "$CONDA_DEFAULT_ENV" != "fed" ]]; then
    echo -e "${YELLOW}警告: 当前不在 fed 环境中，正在激活...${NC}"
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate fed
fi

echo -e "${GREEN}✓ Conda 环境: $CONDA_DEFAULT_ENV${NC}"

# 检查配置文件
CONFIG_FILE="flq_config.json"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo -e "${RED}错误: 找不到配置文件 $CONFIG_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 配置文件: $CONFIG_FILE${NC}"

# Jetson Nano 专用优化
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

# 设置内存限制（避免 OOM）
export TF_GPU_MEMORY_LIMIT=2048

# CUDA 优化
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_PATH=/tmp/cuda_cache

echo -e "${GREEN}✓ Jetson Nano GPU 环境已配置${NC}"

# 创建结果目录
mkdir -p results
mkdir -p logs

# 启动参数
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-5000}

echo -e "${BLUE}启动参数:${NC}"
echo -e "  配置文件: $CONFIG_FILE"
echo -e "  监听地址: $HOST:$PORT"
echo -e "  结果目录: ./results"
echo -e "  日志目录: ./logs"

# 生成启动时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/master_$TIMESTAMP.log"

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}🚀 启动 FLQ Master...${NC}"
echo -e "${YELLOW}日志文件: $LOG_FILE${NC}"
echo -e "${BLUE}========================================${NC}"

# 启动 master（后台运行并记录日志）
python flq_master.py \
    --config "$CONFIG_FILE" \
    --host "$HOST" \
    --port "$PORT" \
    2>&1 | tee "$LOG_FILE"

echo -e "${RED}Master 已停止${NC}"
