#!/usr/bin/env python3
"""
PyTorch版本的MNIST联邦学习Worker
支持FLQ量化：sign1(1位)和int8(8位)
"""

import requests
import numpy as np
import time
import os
import random
import socket
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# ---------------------
# 可配置参数
# ---------------------
MASTER_ADDR = os.environ.get("MASTER_ADDR", "http://master-service:5000")
WORKER_ID = os.environ.get("WORKER_ID") or socket.gethostname()
MODEL_DIM = int(os.getenv("MODEL_DIM", 100_000))  # 将由模型实际维度覆盖
LOCAL_STEPS = int(os.getenv("LOCAL_STEPS", 5))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
DATASET_SIZE = int(os.getenv("DATASET_SIZE", 50_000))
LR = float(os.getenv("LR", 0.05))
SEED = int(os.getenv("SEED", 1234))
FLQ_MODE = os.environ.get("FLQ_MODE", "off")  # off, sign1, int8

# 目标内存（MB）
MEM_MB = int(os.getenv("MEM_MB", 500))
TOUCH_STRIDE_BYTES = int(os.getenv("TOUCH_STRIDE_BYTES", 4096))

# 设置随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# 设备配置（优先使用CPU进行联邦学习）
DEVICE = torch.device("cpu")
print(f"[{WORKER_ID}] 使用设备: {DEVICE}")

# ---------------------
# MNIST 数据加载与预处理
# ---------------------
def load_mnist_data(worker_id=0, total_workers=1, noniid=False):
    """加载MNIST数据集，支持IID切分"""
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化
    ])
    
    # 加载训练集
    full_train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    # IID切分：每个worker获得均等数据
    dataset_size = len(full_train_dataset)
    samples_per_worker = dataset_size // total_workers
    start_idx = worker_id * samples_per_worker
    end_idx = (worker_id + 1) * samples_per_worker
    
    # 创建子数据集
    indices = list(range(start_idx, end_idx))
    worker_dataset = torch.utils.data.Subset(full_train_dataset, indices)
    
    # 加载测试集（所有worker共享）
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    return worker_dataset, test_dataset

# ---------------------
# PyTorch模型定义
# ---------------------
class MNISTCNN(nn.Module):
    """简单的MNIST CNN模型"""
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 14x14 -> 14x14
        self.pool = nn.MaxPool2d(2, 2)  # 2x2池化
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # 计算全连接层输入维度
        # 经过两次池化: 28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 第一层卷积+池化
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        
        # 第二层卷积+池化
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

def build_model():
    """构建模型并移动到设备"""
    model = MNISTCNN().to(DEVICE)
    return model

# ---------------------
# 模型权重与向量互转
# ---------------------
def model_to_vector(model):
    """将模型权重转换为向量"""
    vec = []
    for param in model.parameters():
        vec.append(param.data.cpu().numpy().flatten())
    return np.concatenate(vec).astype(np.float32)

def vector_to_model(vec, model):
    """将向量转换回模型权重"""
    start_idx = 0
    for param in model.parameters():
        param_data = param.data.cpu().numpy()
        shape = param_data.shape
        size = np.prod(shape)
        
        # 更新参数
        new_data = vec[start_idx:start_idx + size].reshape(shape)
        param.data = torch.from_numpy(new_data).to(DEVICE)
        start_idx += size
    return model

def get_model_dim(model):
    """获取模型参数维度"""
    return sum(np.prod(param.shape) for param in model.parameters())

# ---------------------
# FLQ 量化函数
# ---------------------
def apply_flq_quantization(delta, mode):
    """应用FLQ量化"""
    if mode == "off":
        return delta, 32 * len(delta)  # 32 bits per parameter
    
    elif mode == "sign1":
        # sign1量化：只保留符号信息，乘以平均绝对值
        alpha = np.mean(np.abs(delta))
        q = alpha * np.sign(delta)
        logical_bits = 1 * len(delta)
        return q, logical_bits
    
    elif mode == "int8":
        # int8量化：对称量化到-128~127
        max_val = np.max(np.abs(delta))
        if max_val > 0:
            scale = max_val / 127.0
            q_int8 = np.clip(np.round(delta / scale), -128, 127)
            q = q_int8 * scale  # 反量化回FP32
        else:
            q = delta
        logical_bits = 8 * len(delta)
        return q, logical_bits
    
    else:
        raise ValueError(f"Unknown FLQ_MODE: {mode}")

# ---------------------
# 内存压载
# ---------------------
def allocate_ballast(mem_mb: int):
    """分配内存压载"""
    n_bytes = mem_mb * 1024 * 1024
    ballast = np.zeros(n_bytes, dtype=np.uint8)
    
    def touch():
        stride = TOUCH_STRIDE_BYTES
        for i in range(0, n_bytes, stride):
            ballast[i] = (ballast[i] + 1) & 0xFF
    return ballast, touch

ballast, touch_pages = allocate_ballast(MEM_MB)

# ---------------------
# 拉取全局模型
# ---------------------
def pull_global():
    """拉取全局模型"""
    try:
        r = requests.get(f"{MASTER_ADDR}/global", timeout=10)
        j = r.json()
        gw = np.array(j["weights"], dtype=np.float32)
        gr = int(j["round"])
        return gw, gr
    except Exception as e:
        print(f"[{WORKER_ID}] pull_global error: {e}")
        return None, None

# ---------------------
# 上报本地更新
# ---------------------
def push_update(round_id, delta, num_samples, loss, accuracy, logical_bits, compute_factor):
    """上报本地更新"""
    try:
        payload = {
            "worker_id": WORKER_ID,
            "round": int(round_id),
            "num_samples": int(num_samples),
            "loss": float(loss),
            "delta": delta.astype(np.float32).tolist(),
            "compute_factor": float(compute_factor)
        }
        r = requests.post(f"{MASTER_ADDR}/update", json=payload, timeout=15)
        return r.json()
    except Exception as e:
        print(f"[{WORKER_ID}] push_update error: {e}")
        return None

# ---------------------
# 本地训练
# ---------------------
def local_training(global_weights, model, train_loader, steps):
    """执行本地训练并返回权重更新"""
    # 设置全局权重到本地模型
    vector_to_model(global_weights, model)
    
    # 保存初始权重
    initial_weights = global_weights.copy()
    
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # 训练指定步数
    for step, (data, target) in enumerate(train_loader):
        if step >= steps:
            break
            
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct_predictions += pred.eq(target.view_as(pred)).sum().item()
        total_samples += data.size(0)
        
        # 内存触页
        touch_pages()
        
        # 模拟计算延迟
        cf = random.uniform(0.5, 1.5)
        time.sleep(max(0.01, 0.05 / cf))
    
    avg_loss = total_loss / steps
    avg_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    
    # 计算权重更新
    final_weights = model_to_vector(model)
    delta = final_weights - initial_weights
    
    return delta, avg_loss, avg_accuracy

# ---------------------
# 主循环
# ---------------------
def main():
    """主函数"""
    # 预热内存
    touch_pages()
    
    # 加载MNIST数据
    worker_dataset, test_dataset = load_mnist_data(
        worker_id=hash(WORKER_ID) % 10,  # 简单的worker ID映射
        total_workers=10
    )
    
    # 创建数据加载器
    train_loader = DataLoader(worker_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 构建模型
    model = build_model()
    model_dim = get_model_dim(model)
    
    print(f"[{WORKER_ID}] PyTorch MNIST worker started")
    print(f"[{WORKER_ID}] FLQ_MODE={FLQ_MODE}, LOCAL_STEPS={LOCAL_STEPS}, BATCH_SIZE={BATCH_SIZE}")
    print(f"[{WORKER_ID}] Model dimension: {model_dim}")
    print(f"[{WORKER_ID}] Training data: {len(worker_dataset)} samples")
    print(f"[{WORKER_ID}] Device: {DEVICE}")
    
    round_count = 0
    
    while True:
        try:
            # 拉取全局模型
            global_weights, global_round = pull_global()
            
            if global_weights is None:
                time.sleep(2)
                continue
                
            if len(global_weights) != model_dim:
                print(f"[{WORKER_ID}] Dimension mismatch: expected {model_dim}, got {len(global_weights)}")
                # 如果是第一轮，初始化模型
                if global_round == 0:
                    model = build_model()
                    global_weights = model_to_vector(model)
                else:
                    time.sleep(2)
                    continue
                    
        except Exception as e:
            print(f"[{WORKER_ID}] pull_global error: {e}")
            time.sleep(2)
            continue
        
        # 本地训练
        try:
            delta, loss, accuracy = local_training(
                global_weights, model, train_loader, LOCAL_STEPS
            )
            
            # FLQ量化处理
            quantized_delta, logical_bits = apply_flq_quantization(delta, FLQ_MODE)
            
            # 计算算力因子
            compute_factor = random.uniform(0.5, 1.5)
            
            # 上报更新
            resp = push_update(
                global_round, 
                quantized_delta, 
                len(worker_dataset),  # num_samples
                loss, 
                accuracy,
                logical_bits,
                compute_factor
            )
            
            round_count += 1
            
            # 详细日志：包含理论通信比特数
            compression_ratio = 32.0 * len(delta) / logical_bits if logical_bits > 0 else 0
            print(f"[{WORKER_ID}] round={global_round}, loss={loss:.4f}, acc={accuracy:.4f}, "
                  f"logical_bits={logical_bits}, compression={compression_ratio:.1f}:1, "
                  f"flq_mode={FLQ_MODE}, resp={resp}")
            
        except Exception as e:
            print(f"[{WORKER_ID}] training/push error: {e}")
            time.sleep(1)
            continue
        
        # 间隔一会进入下一轮
        time.sleep(0.5)

if __name__ == "__main__":
    main()