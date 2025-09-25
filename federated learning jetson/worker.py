import requests
import numpy as np
import time
import os
import random
import socket

# ---------------------
# 可配置参数
# ---------------------
MASTER_ADDR = os.environ.get("MASTER_ADDR", "http://master-service:5000")
WORKER_ID = os.environ.get("WORKER_ID") or socket.gethostname()
MODEL_DIM = int(os.getenv("MODEL_DIM", 100_000))
LOCAL_STEPS = int(os.getenv("LOCAL_STEPS", 5))         # 每轮本地 step 数
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))          # 仅用于 num_samples 估算
DATASET_SIZE = int(os.getenv("DATASET_SIZE", 50_000))  # 本地数据规模（影响 num_samples 权重）
LR = float(os.getenv("LR", 0.05))
SEED = int(os.getenv("SEED", 1234))

# 目标内存（MB）——用于内存压载
MEM_MB = int(os.getenv("MEM_MB", 500))
TOUCH_STRIDE_BYTES = int(os.getenv("TOUCH_STRIDE_BYTES", 4096))  # 触页步长（字节）

rng = np.random.default_rng(SEED)

# ---------------------
# 内存压载：确保常驻 ~500MB
# ---------------------
def allocate_ballast(mem_mb: int):
    n_bytes = mem_mb * 1024 * 1024
    # 使用 uint8 数组，占用真实物理页；后续定期触页防止被回收
    ballast = np.zeros(n_bytes, dtype=np.uint8)

    def touch():
        # 触碰每隔 TOUCH_STRIDE_BYTES 的一个字节
        stride = TOUCH_STRIDE_BYTES
        for i in range(0, n_bytes, stride):
            ballast[i] = (ballast[i] + 1) & 0xFF
    return ballast, touch

ballast, touch_pages = allocate_ballast(MEM_MB)

# ---------------------
# 模拟“个体差异/数据非 IID”
# 每个 worker 固定一个 drift 向量，使其梯度带有偏移
# ---------------------
DRIFT_SCALE = float(os.getenv("DRIFT_SCALE", 0.1))
worker_drift = rng.standard_normal(MODEL_DIM, dtype=np.float32) * DRIFT_SCALE

# ---------------------
# 模拟算力波动：影响本地 step 间 sleep
# ---------------------
def compute_factor():
    return random.uniform(0.5, 1.5)

# ---------------------
# 简化的损失与梯度模拟
# - 假设目标最优解为 0 向量，则 grad ~ w + drift + 噪声
# ---------------------
def simulate_local_training(global_w, steps, lr):
    w = global_w.copy()
    total_loss = 0.0
    for _ in range(steps):
        noise = rng.standard_normal(w.shape[0], dtype=np.float32) * 0.01
        grad = w + worker_drift + noise
        w = w - lr * grad
        # 简单的 L2 loss
        total_loss += float(0.5 * np.mean(w * w))
        # 触页，维持 500MB 占用
        touch_pages()
        # 按算力因子决定训练速度
        cf = compute_factor()
        time.sleep(max(0.05, 0.15 / cf))
    avg_loss = total_loss / steps
    delta = w - global_w
    return delta, avg_loss

# ---------------------
# 拉取全局模型
# ---------------------
def pull_global():
    r = requests.get(f"{MASTER_ADDR}/global", timeout=10)
    j = r.json()
    gw = np.array(j["weights"], dtype=np.float32)
    gr = int(j["round"])
    if gw.shape[0] != MODEL_DIM:
        raise RuntimeError(f"MODEL_DIM mismatch: worker={MODEL_DIM}, master={gw.shape[0]}")
    return gw, gr

# ---------------------
# 上报本地更新
# ---------------------
def push_update(round_id, delta, num_samples, loss, cf):
    payload = {
        "worker_id": WORKER_ID,
        "round": int(round_id),
        "num_samples": int(num_samples),
        "loss": float(loss),
        "delta": delta.astype(np.float32).tolist(),
        "compute_factor": float(cf)
    }
    r = requests.post(f"{MASTER_ADDR}/update", json=payload, timeout=15)
    return r.json()

# ---------------------
# 主循环
# ---------------------
if __name__ == "__main__":
    # 预热一次触页，尽快占满内存
    touch_pages()
    print(f"[{WORKER_ID}] ballast ~{MEM_MB}MB allocated; MODEL_DIM={MODEL_DIM}")

    while True:
        try:
            global_w, global_r = pull_global()
        except Exception as e:
            print(f"[{WORKER_ID}] pull_global error: {e}")
            time.sleep(2)
            continue

        # 当前轮的本地训练
        cf = compute_factor()
        delta, loss = simulate_local_training(global_w, LOCAL_STEPS, LR) 

        # 估计本轮参与的样本数（可按数据规模与 step/batch 简单估计）
        # 也可以引入参与率：每轮仅使用部分数据
        participation = rng.uniform(0.3, 1.0)
        num_samples = int(min(DATASET_SIZE, participation * LOCAL_STEPS * BATCH_SIZE))

        try:
            resp = push_update(global_r, delta, num_samples, loss, cf)
            print(f"[{WORKER_ID}] round={global_r}, loss={loss:.4f}, samples={num_samples}, resp={resp}")
        except Exception as e:
            print(f"[{WORKER_ID}] push_update error: {e}")

        # 间隔一会进入下一次（下一轮会由 master 控制：global_round 自增后，worker 拉取到新轮权重）
        time.sleep(0.5)

