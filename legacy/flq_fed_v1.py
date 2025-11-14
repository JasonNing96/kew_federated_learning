#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, numpy as np, tensorflow as tf

# ------------------------- helpers -------------------------
def gradtovec(grad_list):
    vec = []
    for g in grad_list:
        if g is None: continue
        a = g.numpy() if hasattr(g, "numpy") else np.array(g)
        vec.append(a.reshape(-1))
    return (np.concatenate(vec, axis=0).astype(np.float32) if vec else np.zeros(0, np.float32))

def vectograd(vec, grad_template):
    out, idx = [], 0
    for g in grad_template:
        a = g.numpy() if hasattr(g, "numpy") else np.array(g)
        n = int(np.prod(a.shape))
        part = vec[idx:idx+n]; idx += n
        out.append(tf.convert_to_tensor(part.reshape(a.shape), dtype=tf.float32))
    return out

# def quantd(vec: np.ndarray, ref: np.ndarray, b: int) -> np.ndarray:
#     diff = vec - ref
#     r = float(np.max(np.abs(diff)))
#     if r == 0.0 or b <= 0: return ref.copy()
#     delta = r / (float(2**b) - 1.0)
#     q = ref - r + 2.0*delta * np.floor((vec - ref + r + delta) / (2.0*delta))
#     return q.astype(np.float32)


def quantd(vec: np.ndarray, ref: np.ndarray, b: int) -> np.ndarray:
    """
    相对均匀量化到 b 比特；b==1 时采用“符号+尺度”的稳定二值化。
    返回去量化后的向量（工程原型）。
    """
    diff = vec - ref
    if b <= 0:
        return ref.astype(np.float32)
    if b == 1:
        # 稳定二值：q = ref + alpha * sign(diff)
        alpha = float(np.mean(np.abs(diff)))
        if alpha == 0.0:
            return ref.astype(np.float32)
        sgn = np.sign(diff).astype(np.float32)
        sgn[sgn == 0.0] = 1.0
        q = ref + alpha * sgn
        return q.astype(np.float32)
    # b >= 2：均匀相对量化
    r = float(np.max(np.abs(diff)))
    if r == 0.0:
        return ref.astype(np.float32)
    delta = r / (float(2**b) - 1.0)
    q = ref - r + 2.0 * delta * np.floor((vec - ref + r + delta) / (2.0 * delta))
    return q.astype(np.float32)

# --------------------- dataset: MNIST / FMNIST ---------------------
def _load_arrays(name: str):
    name = name.lower()
    if name in ["mnist","mn"]:
        (xtr,ytr),(xte,yte)=tf.keras.datasets.mnist.load_data()
    elif name in ["fmnist","fashion_mnist","fashion"]:
        (xtr,ytr),(xte,yte)=tf.keras.datasets.fashion_mnist.load_data()
    else:
        raise ValueError(f"unknown dataset: {name}")
    xtr=(xtr.astype("float32")/255.0)[...,None]
    xte=(xte.astype("float32")/255.0)[...,None]
    ytr=ytr.astype("int64"); yte=yte.astype("int64")
    return (xtr,ytr),(xte,yte)

def make_federated_iid(dataset: str, M: int, seed: int = 1234):
    (xtr,ytr),(xte,yte)=_load_arrays(dataset)
    rng=np.random.default_rng(seed); idx=np.arange(len(xtr)); rng.shuffle(idx)
    xtr,ytr=xtr[idx],ytr[idx]
    Mi=len(xtr)//M
    dss=[]
    for m in range(M):
        xs=xtr[m*Mi:(m+1)*Mi]; ys=ytr[m*Mi:(m+1)*Mi]
        ds=tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(xs),tf.convert_to_tensor(ys))).batch(32).repeat()
        dss.append(ds)
    test_ds=tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(xte),tf.convert_to_tensor(yte))).batch(512)
    return dss,test_ds,Mi

def make_federated_noniid(dataset: str, M: int, seed: int = 1234, classes_per_client: int = 2):
    """
    Create non-IID federated data where each client has data from only a few classes
    """
    (xtr,ytr),(xte,yte)=_load_arrays(dataset)
    rng=np.random.default_rng(seed)
    
    # Get number of classes (MNIST/Fashion-MNIST both have 10 classes)
    num_classes = len(np.unique(ytr))
    
    # Sort data by class
    sort_idx = np.argsort(ytr)
    xtr_sorted, ytr_sorted = xtr[sort_idx], ytr[sort_idx]
    
    # Split data by class
    class_data = {}
    for c in range(num_classes):
        class_mask = (ytr_sorted == c)
        class_data[c] = (xtr_sorted[class_mask], ytr_sorted[class_mask])
    
    # Assign classes to clients
    client_classes = []
    for m in range(M):
        # Each client gets classes_per_client consecutive classes (with wraparound)
        start_class = (m * classes_per_client) % num_classes
        client_class_list = []
        for i in range(classes_per_client):
            client_class_list.append((start_class + i) % num_classes)
        client_classes.append(client_class_list)
    
    # Create datasets for each client
    dss = []
    total_samples_per_client = len(xtr) // M
    
    for m in range(M):
        xs_list, ys_list = [], []
        samples_per_class = total_samples_per_client // classes_per_client
        
        for class_id in client_classes[m]:
            class_x, class_y = class_data[class_id]
            # Randomly sample from this class
            class_indices = rng.choice(len(class_x), size=samples_per_class, replace=False)
            xs_list.append(class_x[class_indices])
            ys_list.append(class_y[class_indices])
        
        # Concatenate all data for this client
        xs = np.concatenate(xs_list, axis=0)
        ys = np.concatenate(ys_list, axis=0)
        
        # Shuffle client's data
        client_idx = np.arange(len(xs))
        rng.shuffle(client_idx)
        xs, ys = xs[client_idx], ys[client_idx]
        
        # Create TensorFlow dataset
        # ds = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(xs), tf.convert_to_tensor(ys))).batch(32).repeat()
        ds = (tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(xs), tf.convert_to_tensor(ys)))
        .shuffle(10000, seed=seed, reshuffle_each_iteration=True)
        .batch(32, drop_remainder=True)
        .repeat()
        .prefetch(tf.data.AUTOTUNE))
        dss.append(ds)
    
    # Test dataset remains the same for all clients
    # test_ds = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(xte), tf.convert_to_tensor(yte))).batch(512)
    test_ds = tf.data.Dataset.from_tensor_slices(
    (tf.convert_to_tensor(xte), tf.convert_to_tensor(yte))).batch(512).prefetch(tf.data.AUTOTUNE)
    
    # Calculate average samples per client
    Mi = total_samples_per_client
    
    return dss, test_ds, Mi

# --------------------------- model ---------------------------
# MLP 小网，上限在0.88 ~0.92 
def build_model(l2: float, lr: float):
    # 修正点：
    # 1) 输入假量化到 [0,1]（与你的归一化一致），不要把输出层裁剪到 [-1,1]
    # 2) 输出层不加激活，保留“原始 logits”，配合 from_logits=True
    regularizer = tf.keras.regularizers.L2(l2=l2)
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        tf.keras.layers.Lambda(lambda x: tf.quantization.fake_quant_with_min_max_vars(
            x, min=0.0, max=1.0, num_bits=8)),
        tf.keras.layers.Flatten(),
        # 可选：插一层小宽度 MLP 提升上限（例如 128 ReLU），先给最小改动：直接线性分类头
        tf.keras.layers.Dense(10, activation=None, kernel_regularizer=regularizer),
    ])
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    return model, optimizer

# 最小cnn，以便提高acc，上限在 0.98 ~ 0.99
# def build_model(l2: float, lr: float):
#     # 小型 CNN，足够把 MNIST 推到 ≥0.95；去掉输出层假量化
#     reg = tf.keras.regularizers.L2(l2=l2)
#     model = tf.keras.Sequential([
#         tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),

#         tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=reg),
#         tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=reg),
#         tf.keras.layers.MaxPooling2D(),

#         tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=reg),
#         tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=reg),
#         tf.keras.layers.MaxPooling2D(),

#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=reg),
#         tf.keras.layers.Dense(10, activation=None)  # logits
#     ])
#     optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
#     return model, optimizer


# --------------------------- FLQ train ---------------------------
def run(args):
    # 随机种
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # 数据与迭代器
    Datatr, Datate, _ = make_federated_iid(args.dataset, args.M, seed=args.seed)
    train_iters = [iter(ds.repeat()) for ds in Datatr]  # 持久迭代器

    # 模型与优化器（小 CNN 版本建议配 Adam）
    model, optimizer = build_model(l2=args.cl, lr=args.lr)

    # 元信息
    M = int(args.M)
    iters = int(args.iters)
    nv = sum(int(np.prod(v.shape)) for v in model.trainable_variables)

    # 运行模式与位宽
    mode = getattr(args, "mode", "fedavg").lower()   # "fedavg" | "bbit"
    b = int(getattr(args, "b", 8))

    # 参考向量（仅 bbit 用）
    ref = np.zeros((M, nv), dtype=np.float32)

    loss_history = np.zeros(iters, np.float32)

    for k in range(iters):
        train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        loss_round = 0.0

        agg_vec = np.zeros(nv, dtype=np.float32)
        grads_tmpl = None
        cos_stats = []  # 仅 bbit 诊断

        for m in range(M):
            x, y = next(train_iters[m])

            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                train_acc.update_state(y, logits)
                ce = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
                )
                l2 = sum([args.cl * tf.nn.l2_loss(v) for v in model.trainable_variables])
                loss = ce + l2

            grads = tape.gradient(loss, model.trainable_variables)
            grads_tmpl = grads
            g = gradtovec(grads)

            if mode == "bbit":
                q = quantd(g, ref[m], b=b)  # 去量化后的量化梯度
                # 方向诊断
                denom = (np.linalg.norm(g) * np.linalg.norm(q) + 1e-12)
                cos_stats.append(float(np.dot(g, q) / denom))
                ref[m] = q
                agg_vec += q
            else:
                agg_vec += g

            loss_round += float(loss.numpy()) / M

        # 聚合与更新
        g_hat = agg_vec / float(M)
        ccgrads = vectograd(g_hat, grads_tmpl)
        optimizer.apply_gradients(zip(ccgrads, model.trainable_variables))

        # 日志
        loss_history[k] = loss_round
        if mode == "bbit" and len(cos_stats) > 0:
            print(f"[{k+1}/{iters}] train_acc={float(train_acc.result().numpy()):.4f} "
                  f"train_loss={loss_round:.4f}  cos(g,q) mean={np.mean(cos_stats):.3f} "
                  f"min={np.min(cos_stats):.3f}")
        else:
            print(f"[{k+1}/{iters}] train_acc={float(train_acc.result().numpy()):.4f} "
                  f"train_loss={loss_round:.4f}")

    # 测试评估
    acc = tf.keras.metrics.SparseCategoricalAccuracy()
    for xi, yi in Datate:
        acc.update_state(yi, model(xi, training=False))
    print(f"Test accuracy: {acc.result().numpy():.4f}")
    return loss_history



# --------------------------- main ---------------------------
def parse_args():
    p=argparse.ArgumentParser(description="FLQ federated training (v1, clean)")
    p.add_argument("--dataset",type=str,default="mnist",choices=["mnist","fmnist","fashion","fashion_mnist"])
    p.add_argument("--M",type=int,default=10)
    p.add_argument("--iters",type=int,default=800)
    p.add_argument("--b",type=int,default=1,help="upload quantization bits")
    p.add_argument("--ck",type=float,default=0.8,help="ksi scaling")
    p.add_argument("--D",type=int,default=10,help="history length")
    p.add_argument("--C",type=int,default=100,help="lazy timeout")
    p.add_argument("--lr",type=float,default=0.02)
    p.add_argument("--cl",type=float,default=0.01,help="L2 factor")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--noniid",action="store_true",help="use non-IID data partitioning")
    p.add_argument("--mode",type=str,default="fedavg",choices=["fedavg","bbit"])
    return p.parse_args()

if __name__=="__main__":
    args=parse_args()
    _=run(args)
