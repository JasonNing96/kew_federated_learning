#!/bin/bash
# GPU训练验证脚本

echo "=========================================="
echo "  GPU训练验证"
echo "=========================================="

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedllm

echo ""
echo "1️⃣  检查GPU状态..."
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

echo ""
echo "2️⃣  检查client.py配置..."
grep "DEVICE = " client.py

echo ""
echo "3️⃣  运行快速GPU训练测试（1 epoch, 小图像）..."
python test_local_training.py

echo ""
echo "=========================================="
echo "  验证完成"
echo "=========================================="

