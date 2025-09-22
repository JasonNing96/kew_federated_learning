#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, numpy as np, tensorflow as tf

# ------------------------- utils: pack/unpack -------------------------
def weights_to_vec(vars_list):
    return np.concatenate([v.numpy().reshape(-1) for v in vars_list]).astype(np.float32)

def vec_to_weights(vec, vars_list):
    idx = 0
    for v in vars_list:
        n = int(np.prod(v.shape))
        arr = vec[idx:idx+n].reshape(v.shape)
        v.assign(arr)
        idx += n

def gradtovec(grad_list):
    out = []
    for g in grad_list:
        if g is None: out.append(np.zeros(0, np.float32)); continue
        a = g.numpy() if hasattr(g, "numpy") else np.array(g)
        out.append(a.reshape(-1))
    return (np.concatenate(out, axis=0).astype(np.float32) if out else np.zeros(0, np.float32))

def vectograd(vec, grad_template):
    out, idx = [], 0
    for g in grad_template:
        shape = (g.numpy() if hasattr(g, "numpy") else np.array(g)).shape
        n = int(np.prod(shape))
        part = vec[idx:idx+n]; idx += n
        out.append(tf.convert_to_tensor(part.reshape(shape), dtype=tf.float32))
    return out

# ------------------------- quantization -------------------------
def quantd(vec: np.ndarray, ref: np.ndarray, b: int) -> np.ndarray:
    """Relative k-bit quantization with dequantization.
       b==1 uses sign+mean-abs scale; b>=2 uses uniform levels."""
    diff = vec - ref
    if b <= 0:
        return ref.astype(np.float32)
    if b == 1:
        alpha = float(np.mean(np.abs(diff)))
        if alpha == 0.0:
            return ref.astype(np.float32)
        sgn = np.sign(diff).astype(np.float32)
        sgn[sgn == 0.0] = 1.0
        return (ref + alpha * sgn).astype(np.float32)
    r = float(np.max(np.abs(diff)))
    if r == 0.0:
        return ref.astype(np.float32)
    step = 2.0 * r / (float(2**b) - 1.0)
    q = ref - r + step * np.floor((vec - ref + r) / step + 0.5)
    return q.astype(np.float32)

# ------------------------- datasets -------------------------
def _load_arrays(name: str):
    name = name.lower()
    if name in ["mnist", "mn"]:
        (xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()
    elif name in ["fmnist", "fashion", "fashion_mnist"]:
        (xtr, ytr), (xte, yte) = tf.keras.datasets.fashion_mnist.load_data()
    else:
        raise ValueError(f"unknown dataset: {name}")
    xtr = (xtr.astype("float32")/255.0)[..., None]
    xte = (xte.astype("float32")/255.0)[..., None]
    ytr = ytr.astype("int64"); yte = yte.astype("int64")
    return (xtr, ytr), (xte, yte)

def make_federated_iid(dataset: str, M: int, batch: int, seed: int = 1234):
    (xtr,ytr),(xte,yte)=_load_arrays(dataset)
    rng=np.random.default_rng(seed); idx=np.arange(len(xtr)); rng.shuffle(idx)
    xtr,ytr=xtr[idx],ytr[idx]
    Mi=len(xtr)//M
    dss=[]
    for m in range(M):
        xs=xtr[m*Mi:(m+1)*Mi]; ys=ytr[m*Mi:(m+1)*Mi]
        ds=(tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(xs),tf.convert_to_tensor(ys)))
             .shuffle(10000,seed=seed,reshuffle_each_iteration=True)
             .batch(batch, drop_remainder=True)
             .repeat()
             .prefetch(tf.data.AUTOTUNE))
        dss.append(ds)
    test_ds=(tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(xte),tf.convert_to_tensor(yte)))
             .batch(512).prefetch(tf.data.AUTOTUNE))
    return dss,test_ds

# ------------------------- model -------------------------
# def build_model(l2: float, lr: float):
#     # 修正点：
#     # 1) 输入假量化到 [0,1]（与你的归一化一致），不要把输出层裁剪到 [-1,1]
#     # 2) 输出层不加激活，保留“原始 logits”，配合 from_logits=True
#     regularizer = tf.keras.regularizers.L2(l2=l2)
#     model = tf.keras.Sequential([
#         tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
#         tf.keras.layers.Lambda(lambda x: tf.quantization.fake_quant_with_min_max_vars(
#             x, min=0.0, max=1.0, num_bits=8)),
#         tf.keras.layers.Flatten(),
#         # 可选：插一层小宽度 MLP 提升上限（例如 128 ReLU），先给最小改动：直接线性分类头
#         tf.keras.layers.Dense(10, activation=None, kernel_regularizer=regularizer),
#     ])
#     optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
#     return model, optimizer

def build_model(l2: float, lr: float):
    reg = tf.keras.regularizers.L2(l2=l2)
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28,28,1)),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=reg),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=reg),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dense(10, activation=None)  # logits
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    return model, opt

# ------------------------- training (with downlink quant) -------------------------
def run(args):
    tf.random.set_seed(args.seed); np.random.seed(args.seed)

    # 数据与迭代器
    Datatr, Datate = make_federated_iid(args.dataset, args.M, args.batch, seed=args.seed)
    trains = [iter(ds) for ds in Datatr]

    # 模型与优化器
    model, optimizer = build_model(l2=args.cl, lr=args.lr)

    # 形状与超参
    M = int(args.M); K = int(args.iters)
    nv = sum(int(np.prod(v.shape)) for v in model.trainable_variables)

    mode = args.mode.lower()                 # "fedavg" | "bbit" | "bin"
    b_up = 1 if mode == "bin" else int(args.b)
    b_down = int(args.b_down); b_down = 0 if b_down in [0, 32] else b_down

    # 简易资源约束（两选一）：优先用 sel_clients；否则用 bits 预算；都为 0 表示全选
    sel_clients = int(getattr(args, "sel_clients", 0))             # 每轮最多选多少客户端
    up_budget_bits = float(getattr(args, "up_budget_bits", 0.0))   # 每轮上行比特预算

    # per-client 参考（上/下行）
    ref_up   = np.zeros((M, nv), dtype=np.float32)
    ref_down = np.zeros((M, nv), dtype=np.float32)

    # 统计曲线
    loss_hist = np.zeros(K, np.float32); acc_hist = np.zeros(K, np.float32)
    cos_mean_h = np.zeros(K, np.float32); cos_min_h = np.zeros(K, np.float32)
    rmse_h = np.zeros(K, np.float32); alpha_h = np.zeros(K, np.float32)
    selcnt_h = np.zeros(K, np.int32)
    bits_up_cum = np.zeros(K, np.float64); bits_down_cum = np.zeros(K, np.float64)
    cum_up = 0.0; cum_down = 0.0

    for k in range(K):
        # 全局权重快照
        w_global = weights_to_vec(model.trainable_variables)

        train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        loss_round = 0.0

        # 候选缓存
        cand_vecs = []     # 上传向量（g 或 q）
        cand_cost = []     # 该客户端上行比特成本
        cand_gain = []     # 该客户端收益（排序依据）
        cand_idx  = []     # 客户端索引
        new_ref_up = []    # 若被选中则写回的 ref_up

        # 本轮诊断
        cos_list, rmse_list, alpha_list = [], [], []

        # ——遍历客户端，产生候选——
        for m in range(M):
            # 下行量化（Broadcast）
            if b_down > 0:
                w_down = quantd(w_global, ref_down[m], b=b_down)
                ref_down[m] = w_down
                vec_to_weights(w_down, model.trainable_variables)
                cum_down += b_down * nv
            else:
                vec_to_weights(w_global, model.trainable_variables)
                cum_down += 32 * nv  # 统计为全精度下发

            # 前向/反向
            x, y = next(trains[m])
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                train_acc.update_state(y, logits)
                ce = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
                )
                l2 = sum([args.cl * tf.nn.l2_loss(v) for v in model.trainable_variables])
                loss = ce + l2
            grads = tape.gradient(loss, model.trainable_variables)
            g = gradtovec(grads)
            loss_round += float(loss.numpy()) / M

            if mode in ["bbit", "bin"]:
                # 去量化后的量化梯度（不改既有量化逻辑）
                q = quantd(g, ref_up[m], b=b_up)

                # 诊断
                denom = (np.linalg.norm(g) * np.linalg.norm(q) + 1e-12)
                cos_list.append(float(np.dot(g, q) / denom))
                rmse_list.append(float(np.sqrt(np.mean((g - q) ** 2))))
                if b_up == 1:
                    diff = g - ref_up[m]
                    alpha_list.append(float(np.mean(np.abs(diff))))

                # 资源记账与收益（创新能量）
                cost_bits = b_up * nv
                gain = float(np.dot(q - ref_up[m], q - ref_up[m]))  # ||dL||^2

                cand_vecs.append(q); cand_cost.append(cost_bits)
                cand_gain.append(gain); cand_idx.append(m); new_ref_up.append(q)
            else:
                # FedAvg：原梯度
                cost_bits = 32 * nv
                gain = float(np.dot(g, g))  # ||g||^2
                cand_vecs.append(g); cand_cost.append(cost_bits)
                cand_gain.append(gain); cand_idx.append(m); new_ref_up.append(ref_up[m])  # 不改 ref

        # 恢复全局权重（避免被某客户端的 w_down 残留影响）
        vec_to_weights(w_global, model.trainable_variables)

        # ——选择策略：在预算内选“收益最大”的子集——
        order = np.argsort(cand_gain)[::-1]  # 按收益降序
        selected = []

        if sel_clients > 0:
            selected = list(order[:min(sel_clients, M)])
            sel_cost_bits = sum(cand_cost[i] for i in selected)
        elif up_budget_bits > 0:
            budget = up_budget_bits
            for i in order:
                if cand_cost[i] <= budget:
                    selected.append(i)
                    budget -= cand_cost[i]
            sel_cost_bits = up_budget_bits - budget
        else:
            selected = list(range(M))
            sel_cost_bits = sum(cand_cost)

        # ——聚合：仅选中者参与——
        if len(selected) > 0:
            agg_vec = np.zeros(nv, np.float32)
            for i in selected:
                agg_vec += cand_vecs[i]
                # 写回 ref_up 仅对选中客户端生效
                if mode in ["bbit", "bin"]:
                    ref_up[cand_idx[i]] = new_ref_up[i]
            g_hat = agg_vec / float(len(selected))
            # 用本轮最后一次 grads 的形状模板
            cc = vectograd(g_hat, grads)
            optimizer.apply_gradients(zip(cc, model.trainable_variables))

        # 记录统计
        loss_hist[k] = loss_round
        acc_hist[k]  = float(train_acc.result().numpy())
        selcnt_h[k]  = len(selected)
        cum_up += float(sel_cost_bits); bits_up_cum[k] = cum_up; bits_down_cum[k] = cum_down

        if len(cos_list):
            cos_mean_h[k] = float(np.mean(cos_list))
            cos_min_h[k]  = float(np.min(cos_list))
            rmse_h[k]     = float(np.mean(rmse_list))
            if len(alpha_list): alpha_h[k] = float(np.mean(alpha_list))

        # 打印
        extra = ""
        if len(cos_list):
            extra = f" | cosμ={cos_mean_h[k]:.3f} cosmin={cos_min_h[k]:.3f} rmseμ={rmse_h[k]:.4f}"
            if len(alpha_list): extra += f" αμ={alpha_h[k]:.4f}"
        print(f"[{k+1}/{K}] acc={acc_hist[k]:.4f} loss={loss_hist[k]:.4f} "
              f"sel={selcnt_h[k]}/{M}{extra} | bits↑Σ={bits_up_cum[k]:.2e} bits↓Σ={bits_down_cum[k]:.2e}")

    # 测试
    acc = tf.keras.metrics.SparseCategoricalAccuracy()
    for xi, yi in Datate:
        acc.update_state(yi, model(xi, training=False))
    print(f"Test accuracy: {acc.result().numpy():.4f}")

    return {
        "loss": loss_hist, "acc": acc_hist,
        "cos_mean": cos_mean_h, "cos_min": cos_min_h,
        "rmse": rmse_h, "alpha": alpha_h,
        "selcnt": selcnt_h,
        "bits_up_cum": bits_up_cum, "bits_down_cum": bits_down_cum
    }


# ------------------------- main -------------------------
def parse_args():
    p = argparse.ArgumentParser("FLQ v2 with downlink parameter quantization")
    p.add_argument("--dataset", type=str, default="mnist",
                   choices=["mnist","fmnist","fashion","fashion_mnist"])
    p.add_argument("--M", type=int, default=10)
    p.add_argument("--iters", type=int, default=800)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--mode", type=str, default="fedavg", choices=["fedavg","bbit","bin"])
    p.add_argument("--b", type=int, default=4, help="uplink bits; bin mode forces 1")
    p.add_argument("--b_down", type=int, default=8, help="downlink bits; 0/32 to disable")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cl", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sel_clients", type=int, default=0, help="select clients")
    p.add_argument("--up_budget_bits", type=float, default=0.0, help="uplink budget bits")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    _ = run(args)

# python flq_fed_v2.py --dataset fmnist --mode bin --b_down 8 --lr 5e-4 --iters 800
# 门限 clients 
# python flq_fed_v2.py --dataset mnist --mode bbit --b 4 --b_down 8 --iters 800 --sel_clients 3
# python flq_fed_v2.py --dataset mnist --mode bin --b_down 8 --iters 800 --up_budget_bits 1e9
