#!/bin/bash
# 清理训练输出和临时文件

echo "🧹 清理训练输出..."
rm -rf outputs/server/checkpoints/*
rm -rf outputs/server/logs/*
rm -rf outputs/client*/runs/*
rm -rf outputs/client*/logs/*
rm -rf outputs/experiments/*
echo "✅ 清理完成"
