import requests
import numpy as np
import time
import os
import random
import socket
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# ---------------------
# 可配置参数
# ---------------------
MASTER_ADDR = os.environ.get("MASTER_ADDR", "http://master-service:5000")
WORKER_ID = os.environ.get("WORKER_ID") or socket.gethostname()
MODEL_DIM = int(os.getenv("MODEL_DIM", 100_000))  # 将由模型自动推断
LOCAL_STEPS = int(os.getenv("LOCAL_STEPS", 5))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
DATASET_SIZE = int(os.getenv("DATASET_SIZE", 50_000))
LR = float(os.getenv("LR", 0.05))
SEED = int(os.getenv("SEED", 1234))
FLQ_MODE = os.environ.get("FLQ_MODE", "off")  # off, sign1, int8

# 目标内存（MB）
MEM_MB = int(os.getenv("MEM_MB", 500))
TOUCH_STRIDE_BYTES = int(os.getenv("TOUCH_STRIDE_BYTES", 4096))

rng = np.random.default_rng(SEED)

# ---------------------
# MNIST 数据加载与预处理
# ---------------------
def load_mnist_data(worker_id=0, total_workers=1, noniid=False):
    """加载MNIST数据，支持IID切分"""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # 归一化
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    #  reshape for CNN
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # 转换为one-hot编码
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # IID切分：每个worker获得均等数据
    samples_per_worker = len(x_train) // total_workers
    start_idx = worker_id * samples_per_worker
    end_idx = (worker_id + 1) * samples_per_worker
    
    x_local = x_train[start_idx:end_idx]
    y_local = y_train[start_idx:end_idx]
    
    return (x_local, y_local), (x_test, y_test)

# ---------------------
# 构建MNIST模型
# ---------------------
def build_mnist_model():
    """构建简单的MNIST CNN模型"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=LR),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ---------------------
# 模型权重与向量互转
# ---------------------
def model_to_vector(model):
    """将模型权重转换为向量"""
    weights = model.get_weights()
    vec = np.array([])
    for w in weights:
        vec = np.concatenate([vec, w.flatten()])
    return vec.astype(np.float32)

def vector_to_model(vec, model):
    """将向量转换回模型权重"""
    weights = []
    start_idx = 0
    for layer_weight in model.get_weights():
        shape = layer_weight.shape
        size = np.prod(shape)
        weights.append(vec[start_idx:start_idx + size].reshape(shape))
        start_idx += size
    model.set_weights(weights)
    return model

def get_model_dim(model):
    """获取模型参数维度"""
    return sum(np.prod(w.shape) for w in model.get_weights())

# ---------------------
# FLQ量化函数
# ---------------------
def apply_flq_quantization(delta, mode):
    """应用FLQ量化"""
    if mode == "off":
        return delta, 32 * len(delta)  # 32 bits per parameter
    
    elif mode == "sign1":
        # sign1量化：只保留符号信息，乘以平均绝对值
        alpha = np.mean(np.abs(delta))
        q = alpha * np.sign(delta)
        logical_bits = 1 * len(delta)  # 1 bit per parameter
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
            scale = 1.0
        logical_bits = 8 * len(delta)  # 8 bits per parameter
        return q, logical_bits
    
    else:
        raise ValueError(f"Unknown FLQ_MODE: {mode}")

# ---------------------
# 内存压载
# ---------------------
def allocate_ballast(mem_mb):
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
    r = requests.get(f"{MASTER_ADDR}/global", timeout=10)
    j = r.json()
    gw = np.array(j["weights"], dtype=np.float32)
    gr = int(j["round"])
    return gw, gr

# ---------------------
# 上报本地更新
# ---------------------
def push_update(round_id, delta, num_samples, loss, acc, logical_bits, compute_factor):
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

# ---------------------
# 本地训练
# ---------------------
def local_training(global_weights, model, x_train, y_train, steps):
    """执行本地训练并返回权重更新"""
    # 设置全局权重到本地模型
    vector_to_model(global_weights, model)
    
    initial_weights = global_weights.copy()
    
    # 随机选择训练批次
    dataset_size = len(x_train)
    batch_size = min(BATCH_SIZE, dataset_size)
    
    total_loss = 0.0
    total_acc = 0.0
    
    for step in range(steps):
        # 随机选择batch
        indices = rng.choice(dataset_size, batch_size, replace=False)
        x_batch = x_train[indices]
        y_batch = y_train[indices]
        
        # 训练一步
        loss, acc = model.train_on_batch(x_batch, y_batch)
        total_loss += loss
        total_acc += acc
        
        # 内存触页
        touch_pages()
        
        # 模拟计算延迟
        cf = random.uniform(0.5, 1.5)
        time.sleep(max(0.01, 0.05 / cf))
    
    avg_loss = total_loss / steps
    avg_acc = total_acc / steps
    
    # 计算权重更新
    final_weights = model_to_vector(model)
    delta = final_weights - initial_weights
    
    return delta, avg_loss, avg_acc

# ---------------------
# 主循环
# ---------------------
def main():
    # 预热内存
    touch_pages()
    
    # 加载MNIST数据
    (x_train, y_train), (x_test, y_test) = load_mnist_data(
        worker_id=hash(WORKER_ID) % 10,  # 简单的worker ID映射
        total_workers=10
    )
    
    # 构建模型
    model = build_mnist_model()
    model_dim = get_model_dim(model)
    
    print(f"[{WORKER_ID}] MNIST worker started")
    print(f"[{WORKER_ID}] FLQ_MODE={FLQ_MODE}, LOCAL_STEPS={LOCAL_STEPS}, BATCH_SIZE={BATCH_SIZE}")
    print(f"[{WORKER_ID}] Model dimension: {model_dim}")
    print(f"[{WORKER_ID}] Training data: {len(x_train)} samples")
    
    round_count = 0
    
    while True:
        try:
            # 拉取全局模型
            global_weights, global_round = pull_global()
            
            if len(global_weights) != model_dim:
                print(f"[{WORKER_ID}] Dimension mismatch: expected {model_dim}, got {len(global_weights)}")
                # 如果是第一轮，初始化模型
                if global_round == 0:
                    model = build_mnist_model()
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
            delta, loss, acc = local_training(
                global_weights, model, x_train, y_train, LOCAL_STEPS
            )
            
            # FLQ量化处理
            quantized_delta, logical_bits = apply_flq_quantization(delta, FLQ_MODE)
            
            # 计算算力因子
            compute_factor = random.uniform(0.5, 1.5)
            
            # 上报更新
            resp = push_update(
                global_round, 
                quantized_delta, 
                len(x_train),  # num_samples
                loss, 
                acc,
                logical_bits,
                compute_factor
            )
            
            round_count += 1
            
            # 详细日志：包含理论通信比特数
            print(f"[{WORKER_ID}] round={global_round}, loss={loss:.4f}, acc={acc:.4f}, "
                  f"logical_bits={logical_bits}, flq_mode={FLQ_MODE}, resp={resp}")
            
        except Exception as e:
            print(f"[{WORKER_ID}] training/push error: {e}")
            time.sleep(1)
            continue
        
        # 间隔一会进入下一轮
        time.sleep(0.5)

if __name__ == "__main__":
    main()