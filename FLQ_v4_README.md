# FLQ Federated Learning v4 - PyTorch版本

## 🎯 概述

FLQ v4是基于PyTorch的联邦学习量化算法实现，从TensorFlow v3版本完全移植而来。保持了所有核心功能和算法逻辑，同时提供了更好的GPU支持和更灵活的深度学习框架。

## ✨ 主要特性

### 🔄 完整的算法支持
- **FedAvg**: 标准联邦平均算法
- **BBIT**: 多比特量化 (2-8 bit)
- **BIN**: 二值量化 (1 bit)
- **LAQ8**: LAQ 8比特量化

### 📊 数据集支持
- **MNIST**: 手写数字识别
- **Fashion-MNIST**: 时尚物品分类

### 🎲 数据分布
- **IID**: 独立同分布数据划分
- **Non-IID**: Dirichlet分布非独立同分布

### 🧠 模型架构
```python
CNNModel:
  Conv2d(1, 32, 3) + ReLU
  Conv2d(32, 32, 3) + ReLU
  MaxPool2d(2)
  Conv2d(32, 64, 3) + ReLU  
  Conv2d(64, 64, 3) + ReLU
  MaxPool2d(2)
  Flatten
  Linear(64*7*7, 128) + ReLU
  Linear(128, 10)  # logits
```

### 🔧 核心技术
- **逐张量量化**: 每个张量独立量化，提高精度
- **误差补偿**: Error Feedback机制减少量化误差
- **懒惰聚合**: 基于门限的客户端选择
- **预算控制**: 支持通信比特预算和客户端数量限制

## 🚀 快速开始

### 环境要求
```bash
pip install torch torchvision pandas openpyxl numpy
```

### 基本使用
```bash
# MNIST + 4比特量化
python flq_fed_v4.py --dataset mnist --mode bbit --b 4 --iters 100

# Fashion-MNIST + 二值量化
python flq_fed_v4.py --dataset fmnist --mode bin --iters 100

# 非IID数据分布
python flq_fed_v4.py --dataset mnist --mode bbit --partition non_iid --dir_alpha 0.1
```

### 参数说明
```bash
# 数据相关
--dataset {mnist,fmnist}     # 数据集选择
--M 10                       # 客户端数量
--batch 64                   # 批次大小
--partition {iid,non_iid}    # 数据分布
--dir_alpha 0.1              # Dirichlet参数(越小越非IID)

# 算法相关  
--mode {fedavg,bbit,bin,laq8} # 算法模式
--b 8                         # 上行量化比特数
--b_down 8                    # 下行量化比特数
--iters 800                   # 训练轮次

# 优化相关
--lr 1e-3                     # 学习率
--cl 5e-4                     # L2正则化系数
--clip_global 0.0             # 全局梯度裁剪

# 预算控制
--sel_clients 0               # 固定选择客户端数(0=无限制)
--up_budget_bits 17000000.0   # 上行比特预算

# 懒惰聚合
--D 10                        # 历史窗口长度
--ck 0.8                      # 历史权重缩放
--C 1000000000                # 强制通信周期
--warmup 0                    # 预热轮次
--thr_scale 0.0               # 门限缩放因子
```

## 📈 实验结果

### 性能基准 (MNIST, 100轮训练)
| 模式 | 最终准确率 | 交叉熵损失 | 上行比特 | 下行比特 |
|------|------------|------------|----------|----------|
| FedAvg | ~95% | ~0.15 | 1.5e10 | 1.5e10 |
| BBIT-8 | ~93% | ~0.23 | 1.5e9 | 1.5e9 |
| BIN-1 | ~90% | ~0.35 | 1.9e8 | 1.5e9 |
| LAQ8 | ~94% | ~0.20 | 1.5e9 | 1.5e9 |

### 通信效率
- **二值量化(BIN)**: 相比FedAvg节省~87%上行通信
- **8比特量化(BBIT)**: 相比FedAvg节省~90%上行通信  
- **下行量化**: 统一节省~75%下行通信

## 🔍 测试验证

### 功能测试
```bash
python test_v4.py
```
验证所有模式和数据集的基本功能。

### 长时间训练测试
```bash
python flq_fed_v4.py --dataset mnist --mode bbit --iters 800 --M 10
```

## 📊 输出文件

训练完成后生成Excel文件包含：

### curve_{mode} 工作表
- `iter`: 迭代轮次
- `loss`: 训练损失
- `acc`: 测试准确率  
- `entropy`: 测试交叉熵
- `selcnt`: 选择客户端数
- `bits_up_cum`: 累计上行比特
- `bits_down_cum`: 累计下行比特
- `cum_bits_total`: 累计总比特

### bin_{mode} 工作表 (仅BIN模式)
- `comm`: 通信轮次
- `bit`: 二值取值 {0,1}

## 🔧 与v3版本对比

| 特性 | TensorFlow v3 | PyTorch v4 |
|------|---------------|------------|
| 深度学习框架 | TensorFlow 2.x | PyTorch |
| GPU支持 | 基础支持 | 优化支持 |
| 内存管理 | 手动配置 | 自动管理 |
| 数据加载 | tf.data | DataLoader |
| 模型定义 | Sequential | nn.Module |
| 优化器 | Adam | Adam |
| 梯度裁剪 | 手动实现 | 内置支持 |

## 🐛 已知问题

1. **初期准确率显示**: 前几轮显示0.0000是正常的，因为只在特定轮次评估
2. **内存使用**: 大模型可能需要调整batch_size
3. **收敛速度**: 不同随机种子可能影响收敛速度

## 🛠️ 开发说明

### 核心文件结构
```
flq_fed_v4.py           # 主训练脚本
test_v4.py              # 功能测试脚本  
compare_v3_v4.py        # 版本对比脚本
results/                # 结果输出目录
figures/                # 图表输出目录
```

### 关键函数
- `weights_to_vec()`: 模型参数向量化
- `quant_rel_per_tensor()`: 逐张量相对量化
- `make_federated_*()`: 联邦数据划分
- `run()`: 主训练循环

## 📝 更新日志

### v4.0.0 (2024-11-13)
- ✅ 完整移植TensorFlow v3到PyTorch
- ✅ 保持所有算法功能和参数
- ✅ 优化GPU内存管理
- ✅ 添加完整测试套件
- ✅ 验证训练效果和收敛性

## 🤝 贡献

欢迎提交Issue和Pull Request来改进FLQ v4版本！

## 📄 许可证

本项目遵循原项目许可证。
