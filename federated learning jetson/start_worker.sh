#!/bin/bash
# FLQ Worker 启动脚本 - Jetson Nano 优化版本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    FLQ 联邦学习 Worker 启动脚本${NC}"
echo -e "${BLUE}========================================${NC}"

# 解析命令行参数
WORKER_ID=""
MASTER_IP=""
CONFIG_FILE="flq_config.json"

while [[ $# -gt 0 ]]; do
    case $1 in
        --worker_id)
            WORKER_ID="$2"
            shift 2
            ;;
        --master_ip)
            MASTER_IP="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            echo "用法: $0 --worker_id <ID> --master_ip <IP> [--config <config_file>]"
            exit 1
            ;;
    esac
done

# 检查必要参数
if [[ -z "$WORKER_ID" ]]; then
    echo -e "${RED}错误: 必须指定 --worker_id${NC}"
    echo "用法: $0 --worker_id <ID> --master_ip <IP>"
    exit 1
fi

if [[ -z "$MASTER_IP" ]]; then
    echo -e "${RED}错误: 必须指定 --master_ip${NC}"
    echo "用法: $0 --worker_id <ID> --master_ip <IP>"
    exit 1
fi

# 检查 conda 环境
if [[ "$CONDA_DEFAULT_ENV" != "fed" ]]; then
    echo -e "${YELLOW}警告: 当前不在 fed 环境中，正在激活...${NC}"
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate fed
fi

echo -e "${GREEN}✓ Conda 环境: $CONDA_DEFAULT_ENV${NC}"

# 检查配置文件
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo -e "${RED}错误: 找不到配置文件 $CONFIG_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 配置文件: $CONFIG_FILE${NC}"

# 动态更新配置文件中的 master URL
TEMP_CONFIG="flq_config_worker${WORKER_ID}.json"
cp "$CONFIG_FILE" "$TEMP_CONFIG"

# 使用 Python 更新配置
python3 -c "
import json
with open('$TEMP_CONFIG', 'r') as f:
    config = json.load(f)
config['system']['master_url'] = 'http://${MASTER_IP}:5000'
with open('$TEMP_CONFIG', 'w') as f:
    json.dump(config, f, indent=2)
print('已更新 master URL 为: http://${MASTER_IP}:5000')
"

# Jetson Nano 专用优化
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_MEMORY_LIMIT=1024  # Worker 使用较少显存

# CUDA 优化
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_PATH=/tmp/cuda_cache

echo -e "${GREEN}✓ Jetson Nano GPU 环境已配置${NC}"

# 创建日志目录
mkdir -p logs

# 获取本机 IP（用于标识）
LOCAL_IP=$(hostname -I | awk '{print $1}')

echo -e "${BLUE}启动参数:${NC}"
echo -e "  Worker ID: $WORKER_ID"
echo -e "  Master IP: $MASTER_IP"
echo -e "  Local IP: $LOCAL_IP"
echo -e "  配置文件: $TEMP_CONFIG"

# 生成启动时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/worker${WORKER_ID}_$TIMESTAMP.log"

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}🚀 启动 Worker $WORKER_ID...${NC}"
echo -e "${YELLOW}日志文件: $LOG_FILE${NC}"
echo -e "${BLUE}========================================${NC}"

# 清理函数
cleanup() {
    echo -e "\n${YELLOW}正在清理临时文件...${NC}"
    rm -f "$TEMP_CONFIG"
    echo -e "${RED}Worker $WORKER_ID 已停止${NC}"
}
trap cleanup EXIT

# 启动 worker
python flq_worker.py \
    --config "$TEMP_CONFIG" \
    --worker_id "$WORKER_ID" \
    2>&1 | tee "$LOG_FILE"
