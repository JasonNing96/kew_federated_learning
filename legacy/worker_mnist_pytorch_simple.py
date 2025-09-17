#!/usr/bin/env python3
"""
PyTorch简化版MNIST Worker，用于快速测试FLQ量化功能
基于论文《Federated Optimal Framework with Low-bitwidth Quantization》
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 环境变量
MASTER_ADDR = os.environ.get("MASTER_ADDR", "http://127.0.0.1:5000")
WORKER_ID = os.environ.get("WORKER_ID") or socket.gethostname()
FLQ_MODE = os.environ.get("FLQ_MODE", "off")  # off, sign1, int8, lloyd-max
LOCAL_STEPS = int(os.getenv("LOCAL_STEPS", 3))
MODEL_DIM = 1000  # 简化模型维度

print(f"[{WORKER_ID}] 启动PyTorch简化MNIST Worker")
print(f"  MASTER_ADDR: {MASTER_ADDR}")
print(f"  FLQ_MODE: {FLQ_MODE}")
print(f"  LOCAL_STEPS: {LOCAL_STEPS}")

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 设备配置
DEVICE = torch.device("cpu")

# ---------------------
# 根据论文优化的FLQ量化算法
# ---------------------
def apply_flq_quantization(delta, mode):
    """应用FLQ量化算法 - 基于论文实现"""
    if mode == "off":
        return delta, 32 * len(delta)
    
    elif mode == "sign1":
        # 论文中的二进制梯度量化：G_B(θ) = |∇f(θ)|_l * sign(∇f(θ))
        # 其中l是标量归一化因子，这里使用平均绝对值
        gradient_magnitude = np.mean(np.abs(delta))
        quantized = gradient_magnitude * np.sign(delta)
        logical_bits = 1 * len(delta)  # 每参数1位
        return quantized, logical_bits
    
    elif mode == "int8":
        # 论文中的8位低比特量化
        max_val = np.max(np.abs(delta))
        if max_val > 0:
            scale = max_val / 127.0  # 映射到[-127, 127]
            quantized_int8 = np.clip(np.round(delta / scale), -128, 127)
            quantized = quantized_int8 * scale  # 反量化
        else:
            quantized = delta
        logical_bits = 8 * len(delta)  # 每参数8位
        return quantized, logical_bits
    
    elif mode == "lloyd-max":
        # Lloyd-Max量化器 - 论文中用于最优量化
        # 简化的Lloyd-Max量化，使用2个聚类中心（1位量化）
        centroid_positive = np.mean(delta[delta > 0]) if np.any(delta > 0) else 1.0
        centroid_negative = np.mean(delta[delta < 0]) if np.any(delta < 0) else -1.0
        
        quantized = np.where(delta >= 0, centroid_positive, centroid_negative)
        logical_bits = 1 * len(delta)  # 每参数1位
        return quantized, logical_bits
    
    else:
        raise ValueError(f"Unknown FLQ_MODE: {mode}")

def lloyd_max_quantizer(data, num_levels=2):
    """
    Lloyd-Max量化器实现
    论文：使用Lloyd-Max算法进行移动性聚合量化
    """
    if num_levels == 2:
        # 二进制量化：找到最优阈值
        threshold = np.median(data)
        centroid1 = np.mean(data[data <= threshold])
        centroid2 = np.mean(data[data > threshold])
        
        # 分配量化级别
        quantized = np.where(data <= threshold, centroid1, centroid2)
        return quantized, 1 * len(data)  # 1位量化
    
    else:
        # 多级别量化（简化版）
        min_val, max_val = np.min(data), np.max(data)
        step = (max_val - min_val) / num_levels
        
        quantized_levels = []
        for i in range(num_levels):
            level_min = min_val + i * step
            level_max = min_val + (i + 1) * step
            centroid = (level_min + level_max) / 2
            quantized_levels.append(centroid)
        
        # 分配最近的量化级别
        quantized = np.zeros_like(data)
        for i, value in enumerate(data):
            distances = [abs(value - level) for level in quantized_levels]
            closest_idx = np.argmin(distances)
            quantized[i] = quantized_levels[closest_idx]
        
        bits_per_param = int(np.ceil(np.log2(num_levels)))
        return quantized, bits_per_param * len(data)

# ---------------------
# 简化的PyTorch MNIST模型
# ---------------------
class SimpleMNISTModel(nn.Module):
    """简化的MNIST模型"""
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(SimpleMNISTModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = x.view(-1, 784)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def create_synthetic_data(num_samples=1000):
    """创建合成MNIST-like数据用于测试"""
    X = torch.randn(num_samples, 1, 28, 28)
    y = torch.randint(0, 10, (num_samples,))
    return X, y

# ---------------------
# 模型权重操作函数
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
        
        new_data = vec[start_idx:start_idx + size].reshape(shape)
        param.data = torch.from_numpy(new_data)
        start_idx += size
    return model

def get_model_dim(model):
    """获取模型参数维度"""
    return sum(np.prod(param.shape) for param in model.parameters())

# ---------------------
# 通信函数
# ---------------------
def pull_global():
    """拉取全局模型"""
    try:
        r = requests.get(f"{MASTER_ADDR}/global", timeout=5)
        j = r.json()
        return np.array(j["weights"], dtype=np.float32), int(j["round"])
    except Exception as e:
        print(f"[{WORKER_ID}] 拉取全局模型失败: {e}")
        return None, None

def push_update(round_id, delta, logical_bits, loss=0.3, accuracy=0.85):
    """上报本地更新"""
    try:
        payload = {
            "worker_id": WORKER_ID,
            "round": int(round_id),
            "num_samples": 100,
            "loss": float(loss),
            "delta": delta.astype(np.float32).tolist(),
            "compute_factor": float(accuracy)
        }
        r = requests.post(f"{MASTER_ADDR}/update", json=payload, timeout=5)
        return r.json()
    except Exception as e:
        print(f"[{WORKER_ID}] 上报更新失败: {e}")
        return None

# ---------------------
# 模拟训练函数（使用PyTorch）
# ---------------------
def simulate_training_pytorch(global_weights, model, steps):
    """使用PyTorch模拟本地训练"""
    # 设置全局权重
    vector_to_model(global_weights, model)
    initial_weights = global_weights.copy()
    
    # 创建优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # 创建合成数据
    X, y = create_synthetic_data(100)
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model.train()
    total_loss = 0.0
    
    for step, (data, target) in enumerate(dataloader):
        if step >= steps:
            break
            
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        time.sleep(0.1)  # 模拟计算延迟
    
    avg_loss = total_loss / steps if steps > 0 else 0.0
    
    # 计算权重更新
    final_weights = model_to_vector(model)
    delta = final_weights - initial_weights
    
    return delta, avg_loss, 0.88  # 模拟准确率

def simulate_training_simple(global_weights, steps):
    """简化的训练模拟（无PyTorch依赖）"""
    # 简单的权重扰动模拟训练
    delta = np.random.randn(len(global_weights)).astype(np.float32) * 0.01
    loss = random.uniform(0.1, 0.5)
    accuracy = random.uniform(0.85, 0.95)
    
    # 模拟训练时间
    for i in range(steps):
        time.sleep(0.1)
    
    return delta, loss, accuracy

# ---------------------
# 主循环
# ---------------------
def main():
    """主循环"""
    # 构建模型
    model = SimpleMNISTModel()
    model_dim = get_model_dim(model)
    
    print(f"[{WORKER_ID}] PyTorch模型构建完成")
    print(f"[{WORKER_ID}] 模型维度: {model_dim}")
    print(f"[{WORKER_ID}] 量化模式: {FLQ_MODE}")
    
    round_count = 0
    
    while round_count < 10:  # 限制测试轮次
        try:
            # 拉取全局模型
            global_weights, global_round = pull_global()
            if global_weights is None:
                time.sleep(2)
                continue
            
            if len(global_weights) != model_dim:
                print(f"[{WORKER_ID}] 维度不匹配: 期望{model_dim}, 实际{len(global_weights)}")
                # 如果是第0轮，使用随机初始化
                if global_round == 0:
                    global_weights = np.random.randn(model_dim).astype(np.float32)
                else:
                    time.sleep(2)
                    continue
            
            print(f"[{WORKER_ID}] 开始第{global_round}轮训练...")
            
            # 本地训练
            try:
                delta, loss, accuracy = simulate_training_pytorch(global_weights, model, LOCAL_STEPS)
                print(f"[{WORKER_ID}] PyTorch训练完成: loss={loss:.4f}, acc={accuracy:.4f}")
            except Exception as e:
                print(f"[{WORKER_ID}] PyTorch训练失败，使用简化训练: {e}")
                delta, loss, accuracy = simulate_training_simple(global_weights, LOCAL_STEPS)
            
            # FLQ量化处理
            if FLQ_MODE == "lloyd-max":
                quantized_delta, logical_bits = lloyd_max_quantizer(delta, num_levels=2)
            else:
                quantized_delta, logical_bits = apply_flq_quantization(delta, FLQ_MODE)
            
            # 上报更新
            resp = push_update(global_round, quantized_delta, logical_bits, loss, accuracy)
            
            if resp:
                compression_ratio = 32.0 * len(delta) / logical_bits if logical_bits > 0 else 0
                print(f"[{WORKER_ID}] 轮次{global_round}: loss={loss:.4f}, acc={accuracy:.4f}, "
                      f"logical_bits={logical_bits}, compression={compression_ratio:.1f}:1, "
                      f"flq_mode={FLQ_MODE}, resp={resp}")
                round_count += 1
            
            time.sleep(1)
            
        except KeyboardInterrupt:
            print(f"[{WORKER_ID}] 收到中断信号，停止训练")
            break
        except Exception as e:
            print(f"[{WORKER_ID}] 训练循环错误: {e}")
            time.sleep(2)
            continue

if __name__ == "__main__":
    main()