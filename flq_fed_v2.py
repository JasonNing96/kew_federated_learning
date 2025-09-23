#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, numpy as np, tensorflow as tf

# ------------------------- Save fig -----------------------------------
import pandas as pd, os

def save_excel_data(outfile: str, mode: str, history: dict,
                    bin_series: np.ndarray | None = None,
                    grad_samples: np.ndarray | None = None):
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    iters = len(history["loss"])
    df_curve = pd.DataFrame({
        "iter": np.arange(1, iters+1, dtype=np.int32),
        "loss": history["loss"].astype(np.float32),
        "acc":  history["acc"].astype(np.float32),
        "selcnt": history.get("selcnt", np.zeros(iters, np.int32)),
        "bits_up_cum": history.get("bits_up_cum", np.zeros(iters, np.float64)),
        "bits_down_cum": history.get("bits_down_cum", np.zeros(iters, np.float64)),
    })
    df_curve["cum_bits_total"] = df_curve["bits_up_cum"] + df_curve["bits_down_cum"]

    with pd.ExcelWriter(outfile) as xw:
        df_curve.to_excel(xw, sheet_name=f"curve_{mode}", index=False)
        if bin_series is not None:
            pd.DataFrame({"comm": np.arange(len(bin_series), dtype=np.int32),
                          "bit":  np.array(bin_series, dtype=np.int8)}
            ).to_excel(xw, sheet_name=f"bin_{mode}", index=False)
        if grad_samples is not None:
            pd.DataFrame({"gt": np.array(grad_samples, dtype=np.float32)}
            ).to_excel(xw, sheet_name=f"gt_{mode}", index=False)
    print(f"[save_excel_data] wrote {outfile}")
    
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

def make_federated_non_iid(dataset: str,
                           M: int,
                           batch: int,
                           alpha: float = 0.3,
                           seed: int = 1234,
                           max_tries: int = 100):
    """
    Dirichlet 非 IID 划分。
    - dataset: "mnist" | "fmnist"
    - M: 客户端数
    - batch: 训练 batch 大小
    - alpha: Dirichlet 超参（越小越非 IID：0.1/0.3 常用）
    - seed: 随机种
    - max_tries: 若有客户端样本数 < batch，则重采 Dirichlet 的最多次数
    返回: (list[tf.data.Dataset], test_ds)
    """
    (xtr, ytr), (xte, yte) = _load_arrays(dataset)
    num_classes = int(ytr.max()) + 1
    rng = np.random.default_rng(seed)

    # 预先按类分桶
    per_class_idx = [np.where(ytr == c)[0].tolist() for c in range(num_classes)]

    def _sample_partition():
        # 打乱每个类桶
        for li in per_class_idx:
            rng.shuffle(li)
        # Dirichlet 比例：shape = (num_classes, M)
        P = rng.dirichlet([alpha] * M, size=num_classes)
        client_bins = [[] for _ in range(M)]
        # 按比例为每个类分配到各客户端
        for c in range(num_classes):
            idxs = per_class_idx[c]
            n_c = len(idxs)
            if n_c == 0:
                continue
            # 先按 floor 分配，再把余数给概率大的客户端
            raw = P[c] * n_c
            cnt = np.floor(raw).astype(int)
            rem = n_c - int(cnt.sum())
            if rem > 0:
                order = np.argsort(raw - cnt)[::-1]
                for j in range(rem):
                    cnt[order[j % M]] += 1
            # 切片放入
            start = 0
            for m in range(M):
                k = int(cnt[m])
                if k > 0:
                    client_bins[m].extend(idxs[start:start + k])
                    start += k
        # 打乱每个客户端样本
        for m in range(M):
            rng.shuffle(client_bins[m])
        return client_bins

    # 反复采样直到所有客户端都有至少 batch 个样本（避免空迭代器）
    bins = _sample_partition()
    tries = 1
    while tries < max_tries and min(len(b) for b in bins) < batch:
        bins = _sample_partition()
        tries += 1

    # 构建 tf.data pipeline
    dss = []
    for m in range(M):
        inds = np.array(bins[m], dtype=np.int64)
        # 如果仍不足一个 batch，则补采全局样本到 batch（极端情况下兜底）
        if len(inds) < batch:
            extra = rng.integers(low=0, high=len(xtr), size=(batch - len(inds),), endpoint=False)
            inds = np.concatenate([inds, extra], axis=0)
        xm, ym = xtr[inds], ytr[inds]
        ds = (tf.data.Dataset.from_tensor_slices(
                (tf.convert_to_tensor(xm), tf.convert_to_tensor(ym)))
              .shuffle(min(len(inds), 10000), seed=seed, reshuffle_each_iteration=True)
              .batch(batch, drop_remainder=True)
              .repeat()
              .prefetch(tf.data.AUTOTUNE))
        dss.append(ds)

    test_ds = (tf.data.Dataset.from_tensor_slices(
                (tf.convert_to_tensor(xte), tf.convert_to_tensor(yte)))
               .batch(512)
               .prefetch(tf.data.AUTOTUNE))
    return dss, test_ds


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

def laq_quantize_stochastic(g: np.ndarray, k: int = 8) -> np.ndarray:
    # 对称 k-bit, 动态尺度 s = max|g| / L, L = 2^(k-1)-1
    L = float(2**(k-1) - 1)
    amax = float(np.max(np.abs(g))) + 1e-12
    s = amax / L
    y = g / s
    low = np.floor(y)
    p = y - low
    rnd = (np.random.rand(*y.shape) < p).astype(np.float32)
    q_int = np.clip(low + rnd, -L, L)
    return (q_int * s).astype(np.float32)

# ------------------------- training (with downlink quant) -------------------------
def run(args):
    import math
    tf.random.set_seed(args.seed); np.random.seed(args.seed)

    # ---------- data ----------
    part = getattr(args, "partition", "iid").lower()   # "iid" | "non_iid"
    alpha = float(getattr(args, "dir_alpha", 0.3))     # non-iid 的 Dirichlet α
    # Datatr, Datate = make_federated_iid(args.dataset, args.M, args.batch, seed=args.seed)
    if part == "iid":
        Datatr, Datate = make_federated_iid(args.dataset, args.M, args.batch, seed=args.seed)
    else:
        Datatr, Datate = make_federated_non_iid(args.dataset, args.M, args.batch, alpha=alpha, seed=args.seed)

    trains = [iter(ds) for ds in Datatr]   
    # ---------- model ----------
    model, optimizer = build_model(l2=args.cl, lr=args.lr)

    # ---------- meta ----------
    M = int(args.M); K = int(args.iters)
    nv = sum(int(np.prod(v.shape)) for v in model.trainable_variables)

    mode   = args.mode.lower()                          # "fedavg" | "bbit" | "bin"
    b_up   = 1 if mode == "bin" else int(args.b)        # uplink bits
    b_down = int(args.b_down); b_down = 0 if b_down in [0, 32] else b_down

    # 懒惰 + 历史强度参数（若未在 argparse 中出现，取默认）
    D         = int(getattr(args, "D", 10))             # 历史窗口长度
    ck        = float(getattr(args, "ck", 0.8))         # 历史权重缩放
    C         = int(getattr(args, "C", 50))             # 未触发的超时轮数
    warmup    = int(getattr(args, "warmup", 50))        # 暖启动强制参与
    thr_scale = float(getattr(args, "thr_scale", 1.0))  # 门限缩放
    alpha_lr  = float(args.lr)                           # 用学习率作 α（与论文口径一致）

    # 资源约束：两选一（都为 0 => 全选）
    sel_clients     = int(getattr(args, "sel_clients", 0))           # 每轮最多选多少客户端
    up_budget_bits  = float(getattr(args, "up_budget_bits", 0.0))    # 每轮上行比特预算

    # ---------- states ----------
    # 上/下行参考（相对量化中心）
    ref_up   = np.zeros((M, nv), dtype=np.float32)
    ref_down = np.zeros((M, nv), dtype=np.float32)

    # 懒惰触发需要：历史强度、误差记忆、超时钟
    theta = np.zeros(nv, dtype=np.float32)
    dtheta_hist = np.zeros((nv, D), dtype=np.float32)   # 每列存一轮的 Δθ
    # 预计算历史权重矩阵 ksi[d, k]（第 d 列的权重，随轮次衰减）
    ksi = np.zeros((D, D + 1), dtype=np.float32)
    for d in range(D):
        ksi[d, 0] = 1.0
        for k in range(1, D + 1):
            ksi[d, k] = 1.0 / float(d + 1)
    ksi *= ck

    e    = np.zeros(M, np.float32)      # 本轮量化误差 ||q-g||^2
    ehat = np.zeros(M, np.float32)      # 上次误差
    clock = np.zeros(M, np.int32)       # 超时计数

    # ---------- logs ----------
    loss_hist = np.zeros(K, np.float32); acc_hist = np.zeros(K, np.float32)
    cos_mean_h = np.zeros(K, np.float32); cos_min_h = np.zeros(K, np.float32)
    rmse_h = np.zeros(K, np.float32); alpha_h = np.zeros(K, np.float32)
    selcnt_h = np.zeros(K, np.int32)
    bits_up_cum = np.zeros(K, np.float64); bits_down_cum = np.zeros(K, np.float64)
    cum_up = 0.0; cum_down = 0.0

    for k in range(K):
        # ---- 维护历史 Δθ 与历史强度 me_k ----
        var = weights_to_vec(model.trainable_variables)
        if k > 0:
            dtheta = var - theta
            dtheta_hist = np.roll(dtheta_hist, 1, axis=1); dtheta_hist[:, 0] = dtheta
        theta = var
        # 历史强度：me_k = Σ_d ksi[d,k] * ||Δθ_{k-d}||^2
        me_k = 0.0
        col_limit = min(k, D)            # 可用的历史列数
        kk = min(k, D)
        for d in range(col_limit):
            col = dtheta_hist[:, d]
            me_k += float(ksi[d, kk] * (col @ col))

        # ---- 快照全局权重（用于下发 & 回滚）----
        w_global = var.copy()

        # ---- 本轮统计 ----
        train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        loss_round = 0.0
        grads_tpl = None

        # 候选集缓存
        cand_idx, cand_vec, cand_gain, cand_cost = [], [], [], []
        cand_q_for_ref = []  # 仅 bbit/bin：若入选则写回 ref_up
        cos_list, rmse_list, alpha_list = [], [], []

        # ---- 遍历所有客户端：生成候选（先过门限，再进入预算选择）----
        for m in range(M):
            # 下行参数量化（广播）
            if b_down > 0:
                w_down = quantd(w_global, ref_down[m], b=b_down)
                ref_down[m] = w_down
                vec_to_weights(w_down, model.trainable_variables)
                cum_down += b_down * nv
            else:
                vec_to_weights(w_global, model.trainable_variables)
                cum_down += 32 * nv

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
            grads_tpl = grads
            g = gradtovec(grads)
            loss_round += float(loss.numpy()) / M

            if mode in ["bbit", "bin"]:
                diff = g - ref_up[m]
                if b_up == 1:
                    alpha_list.append(float(np.mean(np.abs(diff))))
                q = quantd(g, ref_up[m], b=b_up)          # 去量化后的量化梯度
                # 量化误差与方向
                e[m] = float(np.dot(q - g, q - g))
                denom = (np.linalg.norm(g) * np.linalg.norm(q) + 1e-12)
                cos_list.append(float(np.dot(g, q) / denom))
                rmse_list.append(float(math.sqrt(np.mean((g - q) ** 2))))
                # 创新量
                dL = q - ref_up[m]
                gain = float(np.dot(dL, dL))
                # 门限：历史强度 + 量化误差项
                rhs = (me_k / (alpha_lr * alpha_lr * M * M)) + 3.0 * (e[m] + ehat[m])
                pass_thr = (gain >= thr_scale * rhs) or (k < warmup) or (clock[m] >= C)
                if pass_thr:
                    cand_idx.append(m)
                    cand_vec.append(q)
                    cand_q_for_ref.append(q)
                    cand_gain.append(gain)
                    cand_cost.append(b_up * nv)
                # 否则仅累积时钟，先不进入候选
            if mode in ["laq8"]:
                q = laq_quantize_stochastic(g, k=8)
                # 诊断
                denom = (np.linalg.norm(g)*np.linalg.norm(q) + 1e-12)
                cos_list.append(float(np.dot(g, q)/denom))
                rmse_list.append(float(np.sqrt(np.mean((g - q)**2))))
                # 成本与收益（与 fedavg 一致，用 ||g||^2）
                cost_bits = 8 * nv
                gain = float(np.dot(g, g))
                cand_idx.append(m); cand_vec.append(q); cand_q_for_ref.append(None)
                cand_gain.append(gain); cand_cost.append(cost_bits)
            else:
                # FedAvg：无量化门限（可选同样门限，用 ||g||^2 与 e=0）
                gain = float(np.dot(g, g))
                cand_idx.append(m)
                cand_vec.append(g)
                cand_q_for_ref.append(ref_up[m])  # 不改 ref
                cand_gain.append(gain)
                cand_cost.append(32 * nv)

        # 回滚全局权重，避免被任一 w_down 残留影响
        vec_to_weights(w_global, model.trainable_variables)

        # ---- 选择：在预算内选收益/成本高的子集 ----
        order = np.argsort(np.array(cand_gain) / (np.array(cand_cost) + 1e-12))[::-1]
        selected = []
        if sel_clients > 0:
            selected = list(order[:min(sel_clients, len(order))])
            cost_bits = sum(cand_cost[i] for i in selected)
        elif up_budget_bits > 0.0:
            budget = up_budget_bits
            for i in order:
                if cand_cost[i] <= budget:
                    selected.append(i); budget -= cand_cost[i]
            cost_bits = up_budget_bits - budget
        else:
            selected = list(order)  # 全选候选
            cost_bits = sum(cand_cost[i] for i in selected)

        # ---- 聚合与写回状态：仅对入选者 ----
        if len(selected) > 0:
            agg = np.zeros(nv, np.float32)
            for i in selected:
                m = cand_idx[i]
                agg += cand_vec[i]
                if mode in ["bbit", "bin"]:
                    ref_up[m] = cand_q_for_ref[i]   # 更新参考
                    ehat[m]  = e[m]                 # 记忆误差
                    clock[m] = 0
                else:
                    clock[m] = 0
            g_hat = agg / float(len(selected))
            cc = vectograd(g_hat, grads_tpl)
            optimizer.apply_gradients(zip(cc, model.trainable_variables))
        # 未入选者：仅时钟 +1
        not_sel = set(range(M)) - set(cand_idx[i] for i in selected)
        for m in not_sel:
            clock[m] = min(clock[m] + 1, C + 1)

        # ---- 统计与打印 ----
        cum_up += float(cost_bits)
        loss_hist[k] = loss_round
        acc_hist[k]  = float(train_acc.result().numpy())
        selcnt_h[k]  = len(selected)
        bits_up_cum[k]   = cum_up
        bits_down_cum[k] = cum_down
        if len(cos_list):
            cos_mean_h[k] = float(np.mean(cos_list))
            cos_min_h[k]  = float(np.min(cos_list))
            rmse_h[k]     = float(np.mean(rmse_list))
            if b_up == 1 and len(alpha_list): alpha_h[k] = float(np.mean(alpha_list))

        extra = ""
        if len(cos_list):
            extra = f" | cosμ={cos_mean_h[k]:.3f} cosmin={cos_min_h[k]:.3f} rmseμ={rmse_h[k]:.4f}"
            if b_up == 1 and len(alpha_list): extra += f" αμ={alpha_h[k]:.4f}"
        print(f"[{k+1}/{K}] acc={acc_hist[k]:.4f} loss={loss_hist[k]:.4f} "
              f"sel={selcnt_h[k]}/{len(cand_idx)}/{M}{extra} | bits↑Σ={bits_up_cum[k]:.2e} bits↓Σ={bits_down_cum[k]:.2e}")

    # ---------- eval ----------
    acc = tf.keras.metrics.SparseCategoricalAccuracy()
    for xi, yi in Datate:
        acc.update_state(yi, model(xi, training=False))
    print(f"Test accuracy: {acc.result().numpy():.4f}")

    return {
        "loss": loss_hist, "acc": acc_hist, "selcnt": selcnt_h,
        "cos_mean": cos_mean_h, "cos_min": cos_min_h, "rmse": rmse_h, "alpha": alpha_h,
        "bits_up_cum": bits_up_cum, "bits_down_cum": bits_down_cum
    }



# ------------------------- main -------------------------
def parse_args():
    p = argparse.ArgumentParser("FLQ v2 with downlink parameter quantization + lazy/budget")
    # data
    p.add_argument("--dataset", type=str, default="mnist",
                   choices=["mnist", "fmnist", "fashion", "fashion_mnist"])
    p.add_argument("--M", type=int, default=10)
    p.add_argument("--iters", type=int, default=800)
    p.add_argument("--batch", type=int, default=64)
    # modes
    p.add_argument("--mode", type=str, default="bbit", choices=["fedavg", "bbit", "bin","laq8"])
    p.add_argument("--b", type=int, default=4, help="uplink bits; bin mode forces 1")
    p.add_argument("--b_down", type=int, default=8, help="downlink bits; 0/32 disable")
    # optimization
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cl", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    # budget selector (set one; both 0 -> no budget limit)
    p.add_argument("--sel_clients", type=int, default=10, help="max selected clients per round")
    p.add_argument("--up_budget_bits", type=float, default=0.0, help="uplink bit budget per round")
    # lazy trigger (defaults effectively OFF to focus on budget-only comparisons)
    p.add_argument("--D", type=int, default=10, help="history window length")
    p.add_argument("--ck", type=float, default=0.8, help="history weight scale")
    p.add_argument("--C", type=int, default=1000000000, help="timeout rounds for force select")
    p.add_argument("--warmup", type=int, default=0, help="force select rounds at start")
    p.add_argument("--thr_scale", type=float, default=0.0, help="threshold scale (0 disables)")
    p.add_argument("--partition", type=str, default="iid", choices=["iid","non_iid"])
    p.add_argument("--dir_alpha", type=float, default=0.3)  # non-iid 强度，越小越非IID
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    hist = run(args)
    prefix = "results"
    outfile = f"{prefix}_{args.dataset}_{args.mode}.xlsx"
    save_excel_data(outfile, args.mode, hist,
                bin_series=None, grad_samples=None)  # bin/样本可按需填

# python flq_fed_v2.py --dataset fmnist --mode bin --b_down 8 --lr 5e-4 --iters 800
# 门限 clients 
# python flq_fed_v2.py --dataset mnist --mode bbit --b 4 --b_down 8 --iters 800 --sel_clients 3
# python flq_fed_v2.py --dataset mnist --mode bin --b_down 8 --iters 800 --up_budget_bits 1e9

# python flq_fed_v2.py --dataset mnist --mode bbit --b 4 --M 10 --iters 800 --batch 64 --lr 1e-3 --cl 5e-4 --b_down 8 --up_budget_bits 44910528
# python flq_fed_v2.py --dataset mnist --mode bbit --b 2 --M 10 --iters 800 --batch 64 --lr 1e-3 --cl 5e-4 --b_down 8 --up_budget_bits 44910528

# python flq_fed_v2.py --dataset mnist --mode bin --M 10 --iters 800 --batch 64 --lr 5e-4 --cl 5e-4 --b_down 8 --up_budget_bits 44910528 