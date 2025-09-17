#!/usr/bin/env python3
"""
PyTorch简化版MNIST Worker，用于快速测试FLQ量化功能
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

# 环境变量
MASTER_ADDR = os.environ.get("MASTER_ADDR", "http://127.0.0.1:5000")
WORKER_ID = os.environ.get("WORKER_ID") or socket.gethostname()
FLQ_MODE = os.environ.get("FLQ_MODE", "off")  # off, sign1, int8
LOCAL_STEPS = int(os.getenv("LOCAL_STEPS", 3))
MODEL_DIM = 1000  # 简化模型维度

print(f"[{WORKER_ID}] 启动PyTorch简化MNIST Worker")
print(f"  MASTER_ADDR: {MASTER_ADDR}")
print(f"  FLQ_MODE: {FLQ_MODE}")
print(f"  LOCAL_STEPS: {LOCAL_STEPS}")

# 设备配置
DEVICE = torch.device("cpu")

def apply_flq_quantization(delta, mode):
    """FLQ量化函数"""
    if mode == "off":
        return delta, 32 * len(delta)
    elif mode == "sign1":
        alpha = np.mean(np.abs(delta))
        q = alpha * np.sign(delta)
        logical_bits = 1 * len(delta)
        return q, logical_bits
    elif mode == "int8":
        max_val = np.max(np.abs(delta))
        if max_val > 0:
            scale = max_val / 127.0
            q_int8 = np.clip(np.round(delta / scale), -128, 127)
            q = q_int8 * scale
        else:
            q = delta
        logical_bits = 8 * len(delta)
        return q, logical_bits
    else:
        raise ValueError(f"Unknown FLQ_MODE: {mode}")

def pull_global():
    """拉取全局模型"""
    try:
        r = requests.get(f"{MASTER_ADDR}/global", timeout=5)
        j = r.json()
        return np.array(j["weights"], dtype=np.float32), int(j["round"])
    except Exception as e:
        print(f"[{WORKER_ID}] 拉取全局模型失败: {e}")
        return None, None

def push_update(round_id, delta, logical_bits):
    """上报本地更新"""
    try:
        payload = {
            "worker_id": WORKER_ID,
            "round": int(round_id),
            "num_samples": 100,  # 固定样本数
            "loss": random.uniform(0.1, 0.5),  # 随机损失
            "delta": delta.astype(np.float32).tolist(),
            "compute_factor": random.uniform(0.8, 1.2)
        }
        r = requests.post(f"{MASTER_ADDR}/update", json=payload, timeout=5)
        return r.json()
    except Exception as e:
        print(f"[{WORKER_ID}] 上报更新失败: {e}")
        return None

def simulate_pytorch_training(global_weights, steps):
    """使用PyTorch模拟本地训练"""
    # 创建简单的PyTorch模型
    model = nn.Sequential(
        nn.Linear(MODEL_DIM, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, MODEL_DIM)
    ).to(DEVICE)
    
    # 将全局权重设置为模型参数
    param_list = list(model.parameters())
    start_idx = 0
    for param in param_list:
        param_data = param.data.cpu().numpy()
        shape = param_data.shape
        size = np.prod(shape)
        new_data = global_weights[start_idx:start_idx + size].reshape(shape)
        param.data = torch.from_numpy(new_data).to(DEVICE)
        start_idx += size
    
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    model.train()
    total_loss = 0.0
    
    # 模拟训练步骤
    for step in range(steps):
        # 创建随机输入和目标
        input_data = torch.randn(32, MODEL_DIM).to(DEVICE)
        target = torch.randn(32, MODEL_DIM).to(DEVICE)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 模拟计算延迟
        time.sleep(0.1)
    
    avg_loss = total_loss / steps if steps > 0 else 0.0
    accuracy = random.uniform(0.85, 0.95)  # 模拟准确率
    
    # 计算权重更新
    final_weights = []
    for param in model.parameters():
        final_weights.append(param.data.cpu().numpy().flatten())
    final_weights_flat = np.concatenate(final_weights)
    delta = final_weights_flat - global_weights
    
    return delta, avg_loss, accuracy

def main():
    """主循环"""
    round_count = 0
    
    while round_count < 10:  # 限制测试轮次
        try:
            # 拉取全局模型
            global_weights, global_round = pull_global()
            if global_weights is None:
                time.sleep(2)
                continue
            
            if len(global_weights) != MODEL_DIM:
                print(f"[{WORKER_ID}] 维度不匹配: 期望{MODEL_DIM}, 实际{len(global_weights)}")
                # 如果是第0轮，使用随机初始化
                if global_round == 0:
                    global_weights = np.random.randn(MODEL_DIM).astype(np.float32)
                else:
                    time.sleep(2)
                    continue
            
            print(f"[{WORKER_ID}] 开始第{global_round}轮训练...")
            
            # 本地训练
            delta, loss, accuracy = simulate_pytorch_training(global_weights, LOCAL_STEPS)
            
            # FLQ量化
            quantized_delta, logical_bits = apply_flq_quantization(delta, FLQ_MODE)
            
            # 上报更新
            resp = push_update(global_round, quantized_delta, logical_bits)
            
            if resp:
                compression_ratio = 32.0 * len(delta) / logical_bits if logical_bits > 0 else 0
                print(f"[{WORKER_ID}] 轮次{global_round}: loss={loss:.4f}, acc={accuracy:.4f}, "
                      f"logical_bits={logical_bits}, 压缩比={compression_ratio:.1f}:1")
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