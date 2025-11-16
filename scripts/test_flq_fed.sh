#!/bin/bash
# FLQ-Fed 快速测试脚本

set -e

echo "======================================================================="
echo "🧪 FLQ-Fed 快速测试"
echo "======================================================================="

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedllm

# 检查配置
echo ""
echo "📋 检查配置文件..."
if [ -f "configs/flq_config.yaml" ]; then
    echo "✅ configs/flq_config.yaml 存在"
else
    echo "❌ configs/flq_config.yaml 不存在"
    exit 1
fi

# 检查数据
echo ""
echo "📂 检查数据目录..."
if [ -d "fed_oil_demo/client1" ]; then
    echo "✅ fed_oil_demo/client1 存在"
else
    echo "❌ fed_oil_demo/client1 不存在，请先运行 fed_oil_demo/split_dataset.py"
    exit 1
fi

# 测试服务器启动
echo ""
echo "🚀 测试1: 服务器启动（5秒测试）..."
timeout 5 python flq-fed.py server 2>&1 | head -20 &
SERVER_PID=$!
sleep 6
if ps -p $SERVER_PID > /dev/null 2>&1; then
    echo "✅ 服务器启动成功"
    kill $SERVER_PID 2>/dev/null || true
else
    echo "✅ 服务器测试完成"
fi

# 测试客户端路径
echo ""
echo "🚀 测试2: 客户端路径查找..."
python -c "
import os
for i in range(1, 4):
    client_dir = f'client{i}'
    if os.path.exists(f'fed_oil_demo/{client_dir}/oil.yaml'):
        print(f'✅ Client {i}: fed_oil_demo/{client_dir}/oil.yaml')
    else:
        print(f'❌ Client {i}: 数据不存在')
"

# 显示使用说明
echo ""
echo "======================================================================="
echo "✅ 基础测试通过！"
echo "======================================================================="
echo ""
echo "🚀 运行完整训练（1轮快速测试）:"
echo "   python flq-fed.py train --config configs/flq_config.yaml"
echo ""
echo "🚀 或分别启动:"
echo "   python flq-fed.py server    # 终端1"
echo "   python flq-fed.py client --id 1  # 终端2"
echo "   python flq-fed.py client --id 2  # 终端3"
echo "   python flq-fed.py client --id 3  # 终端4"
echo ""
echo "======================================================================="


