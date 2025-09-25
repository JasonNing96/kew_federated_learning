#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, os, numpy as np, tensorflow as tf
import pandas as pd

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"已配置GPU内存增长: {len(gpus)}个GPU")
    except RuntimeError as e:
        print(f"GPU配置错误: {e}")
        
def laq_per_vector(g_vec: np.ndarray, k: int) -> np.ndarray:
    L = float(2**(k-1) - 1)
    s = (np.max(np.abs(g_vec)) + 1e-12) / L
    y = g_vec / s
    low = np.floor(y); p = y - low
    rnd = (np.random.rand(*y.shape) < p).astype(np.float32)
    q = np.clip(low + rnd, -L, L) * s
    return q.astype(np.float32)
# ------------------------- I/O -------------------------
def save_excel_data(outfile: str, mode: str, history: dict,
                    bin_series: np.ndarray | None = None,
                    grad_samples: np.ndarray | None = None):
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    iters = len(history["loss"])
    df = pd.DataFrame({
        "iter": np.arange(1, iters+1, dtype=np.int32),
        "loss": history["loss"].astype(np.float32),
        "acc":  history["acc"].astype(np.float32),
        "entropy": history["entropy"].astype(np.float32),
        "selcnt": history["selcnt"].astype(np.int32),
        "bits_up_cum": history["bits_up_cum"].astype(np.float64),
        "bits_down_cum": history["bits_down_cum"].astype(np.float64),
    })
    df["cum_bits_total"] = df["bits_up_cum"] + df["bits_down_cum"]
    with pd.ExcelWriter(outfile) as xw:
        df.to_excel(xw, sheet_name=f"curve_{mode}", index=False)
        if bin_series is not None:
            pd.DataFrame({"comm": np.arange(len(bin_series), dtype=np.int32),
                          "bit":  np.array(bin_series, dtype=np.int8)}
            ).to_excel(xw, sheet_name=f"bin_{mode}", index=False)
        if grad_samples is not None:
            pd.DataFrame({"gt": np.array(grad_samples, dtype=np.float32)}
            ).to_excel(xw, sheet_name=f"gt_{mode}", index=False)
    print(f"[save_excel_data] wrote {outfile}")

# ------------------------- tensor utils -------------------------
def weights_to_vec(vars_list):  # -> 1D float32
    return np.concatenate([v.numpy().reshape(-1) for v in vars_list]).astype(np.float32)

def vec_to_weights(vec, vars_list):
    i = 0
    for v in vars_list:
        n = int(np.prod(v.shape))
        v.assign(vec[i:i+n].reshape(v.shape))
        i += n

def gradtovec(grad_list):
    out = []
    for g in grad_list:
        if g is None: out.append(np.zeros(0, np.float32)); continue
        a = g.numpy() if hasattr(g, "numpy") else np.array(g)
        out.append(a.reshape(-1))
    return (np.concatenate(out, axis=0).astype(np.float32) if out else np.zeros(0, np.float32))

def vectograd(vec, grad_template):
    out, i = [], 0
    for g in grad_template:
        shape = (g.numpy() if hasattr(g, "numpy") else np.array(g)).shape
        n = int(np.prod(shape)); part = vec[i:i+n]; i += n
        out.append(tf.convert_to_tensor(part.reshape(shape), dtype=tf.float32))
    return out

# ------------------------- per-tensor quant helpers -------------------------
def _split_flat(vec: np.ndarray, shapes):
    out, off = [], 0
    for shp in shapes:
        n = int(np.prod(shp)); out.append(vec[off:off+n]); off += n
    return out

def _cat_flat(chunks):  # list[np.ndarray] -> np.ndarray
    return np.concatenate(chunks, axis=0) if len(chunks) else np.zeros(0, np.float32)

def _quant_tensor_stoch(x: np.ndarray, b: int) -> np.ndarray:
    # 对称 k-bit，随机舍入，无偏
    L = float(2**(b-1) - 1)
    amax = float(np.max(np.abs(x))) + 1e-12
    s = amax / L
    y = x / s
    low = np.floor(y); p = y - low
    rnd = (np.random.rand(*y.shape) < p).astype(np.float32)
    q_int = np.clip(low + rnd, -L, L)
    return (q_int * s).astype(np.float32)

def _quant_bin_tensor(diff: np.ndarray) -> np.ndarray:
    # 二值相对量化：随机 tie-break，按该张量 alpha=mean|diff|
    sgn = np.sign(diff)
    if np.any(sgn == 0):
        rnd = (np.random.rand(*sgn.shape) < 0.5).astype(np.float32)
        sgn = np.where(sgn == 0, 2.0 * rnd - 1.0, sgn)
    alpha = float(np.mean(np.abs(diff)))
    return alpha * sgn

def quant_rel_per_tensor(g_vec: np.ndarray, ref_vec: np.ndarray, b: int, shapes) -> np.ndarray:
    # 逐张量“相对域”量化：q_t = ref_t + Q(diff_t)
    if b <= 0:
        return ref_vec.astype(np.float32)
    g_chunks   = _split_flat(g_vec,  shapes)
    ref_chunks = _split_flat(ref_vec, shapes)
    out = []
    if b == 1:
        for gt, rt in zip(g_chunks, ref_chunks):
            out.append(rt + _quant_bin_tensor(gt - rt))
    else:
        for gt, rt in zip(g_chunks, ref_chunks):
            out.append(rt + _quant_tensor_stoch(gt - rt, b))
    return _cat_flat(out)

def laq_per_tensor(g_vec: np.ndarray, k: int, shapes) -> np.ndarray:
    # 逐张量 LAQ：每张量独立 max 缩放 + 随机舍入
    g_chunks = _split_flat(g_vec, shapes)
    out = [ _quant_tensor_stoch(gt, k) for gt in g_chunks ]
    return _cat_flat(out)

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
    xtr,ytr=xtr[idx],ytr[idx]; Mi=len(xtr)//M; dss=[]
    for m in range(M):
        xs=xtr[m*Mi:(m+1)*Mi]; ys=ytr[m*Mi:(m+1)*Mi]
        ds=(tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(xs),tf.convert_to_tensor(ys)))
             .shuffle(10000,seed=seed,reshuffle_each_iteration=True)
             .batch(batch, drop_remainder=True).repeat().prefetch(tf.data.AUTOTUNE))
        dss.append(ds)
    test_ds=(tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(xte),tf.convert_to_tensor(yte)))
             .batch(512).prefetch(tf.data.AUTOTUNE))
    return dss,test_ds

def make_federated_non_iid(dataset: str, M: int, batch: int, alpha: float = 0.3,
                           seed: int = 1234, max_tries: int = 100):
    (xtr, ytr), (xte, yte) = _load_arrays(dataset)
    num_classes = int(ytr.max()) + 1
    rng = np.random.default_rng(seed)
    per_class_idx = [np.where(ytr == c)[0].tolist() for c in range(num_classes)]

    def _sample_partition():
        for li in per_class_idx: rng.shuffle(li)
        P = rng.dirichlet([alpha] * M, size=num_classes)
        client_bins = [[] for _ in range(M)]
        for c in range(num_classes):
            idxs = per_class_idx[c]; n_c = len(idxs); 
            raw = P[c] * n_c; cnt = np.floor(raw).astype(int)
            rem = n_c - int(cnt.sum())
            if rem > 0:
                order = np.argsort(raw - cnt)[::-1]
                for j in range(rem): cnt[order[j % M]] += 1
            start = 0
            for m in range(M):
                k = int(cnt[m])
                if k > 0:
                    client_bins[m].extend(idxs[start:start+k]); start += k
        for m in range(M): rng.shuffle(client_bins[m])
        return client_bins

    bins = _sample_partition(); tries = 1
    while tries < max_tries and min(len(b) for b in bins) < batch:
        bins = _sample_partition(); tries += 1

    dss = []
    for m in range(M):
        inds = np.array(bins[m], dtype=np.int64)
        if len(inds) < batch:
            extra = rng.integers(low=0, high=len(xtr), size=(batch-len(inds),), endpoint=False)
            inds = np.concatenate([inds, extra], axis=0)
        xm, ym = xtr[inds], ytr[inds]
        ds = (tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(xm), tf.convert_to_tensor(ym)))
              .shuffle(min(len(inds), 10000), seed=seed, reshuffle_each_iteration=True)
              .batch(batch, drop_remainder=True).repeat().prefetch(tf.data.AUTOTUNE))
        dss.append(ds)
    test_ds = (tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(xte), tf.convert_to_tensor(yte)))
               .batch(512).prefetch(tf.data.AUTOTUNE))
    return dss, test_ds

# ------------------------- model -------------------------
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
        tf.keras.layers.Dense(10, activation=None)
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    return model, opt

# ------------------------- training -------------------------
def run(args):
    # ----- 数据 -----
    j_star = None   # Fig.3 观察的坐标
    part  = getattr(args, "partition", "iid").lower()
    alpha = float(getattr(args, "dir_alpha", 0.3))
    if part == "iid":
        Datatr, Datate = make_federated_iid(args.dataset, args.M, args.batch, seed=args.seed)
    else:
        Datatr, Datate = make_federated_non_iid(args.dataset, args.M, args.batch, alpha=alpha, seed=args.seed)
    trains = [iter(ds) for ds in Datatr]

    # ----- 模型 -----
    model, optimizer = build_model(l2=args.cl, lr=args.lr)
    shapes = [tuple(v.shape) for v in model.trainable_variables]
    nv = sum(int(np.prod(s)) for s in shapes)

    # ----- 超参与模式 -----
    M = int(args.M); K = int(args.iters)
    mode   = getattr(args, "mode", "bbit").lower()          # fedavg | bbit | bin | laq8
    b_up   = 1 if mode == "bin" else int(getattr(args, "b", 8))
    b_down = int(getattr(args, "b_down", 0))
    if b_down in (0, 32): b_down = 0                        # 0/32 -> 全精度直发

    # 懒惰相关（保持原逻辑）
    D = int(getattr(args, "D", 10)); ck = float(getattr(args, "ck", 0.8))
    C = int(getattr(args, "C", 50)); warmup = int(getattr(args, "warmup", 50))
    thr_scale = float(getattr(args, "thr_scale", 1.0))
    clip_global = float(getattr(args, "clip_global", 0.0))

    # 预算选择
    sel_clients    = int(getattr(args, "sel_clients", 0))         # >0 固定选端
    up_budget_bits = float(getattr(args, "up_budget_bits", 0.0))  # >0 按预算贪心

    # 按被选端放大步长
    scale_by_selected = bool(int(getattr(args, "scale_by_selected", 1)))
    sel_ref           = float(getattr(args, "sel_ref", 1.0))

    # ----- 状态 -----
    ref_up   = np.zeros((M, nv), np.float32)   # 上行参考
    ref_down = np.zeros((M, nv), np.float32)   # 下行参考
    ef_res   = np.zeros((M, nv), np.float32)   # 误差补偿
    theta = np.zeros(nv, np.float32)
    dtheta_hist = np.zeros((nv, D), np.float32)

    # 历史能量权重
    ksi = np.zeros((D, D+1), np.float32)
    for d in range(D):
        ksi[d, 0] = 1.0
        for k in range(1, D+1):
            ksi[d, k] = 1.0 / float(d + 1)
    ksi *= ck

    e = np.zeros(M, np.float32); ehat = np.zeros(M, np.float32); clock = np.zeros(M, np.int32)

    # ----- 统计 -----
    loss_hist = np.zeros(K, np.float32)
    acc_hist  = np.zeros(K, np.float32)
    entropy_hist = np.zeros(K, np.float32)
    selcnt_h = np.zeros(K, np.int32)
    bits_up_cum = np.zeros(K, np.float64); bits_down_cum = np.zeros(K, np.float64)
    cum_up = 0.0; cum_down = 0.0

    # Fig3 用：每轮 1 个二值位（仅 bin 模式有效）
    bin_series = []

    for k in range(K):
        # 历史能量 me_k
        var = weights_to_vec(model.trainable_variables)
        if k > 0:
            dtheta = var - theta
            dtheta_hist = np.roll(dtheta_hist, 1, axis=1); dtheta_hist[:, 0] = dtheta
        theta = var
        me_k = 0.0; col_limit = min(k, D); kk = min(k, D)
        for d in range(col_limit):
            col = dtheta_hist[:, d]; me_k += float(ksi[d, kk] * (col @ col))

        w_global = var.copy()
        train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        loss_round = 0.0; grads_tpl = None

        # 本轮候选缓存
        g_eff_buf = [None]*M; q_buf = [None]*M
        cand_idx, cand_gain, cand_cost = [], [], []

        # ---------- 本地计算与候选构建 ----------
        for m in range(M):
            # 广播（已实现低比特下发）
            if b_down > 0:
                w_down = quant_rel_per_tensor(w_global, ref_down[m], b_down, shapes)
                ref_down[m] = w_down; vec_to_weights(w_down, model.trainable_variables)
            else:
                vec_to_weights(w_global, model.trainable_variables)

            x, y = next(trains[m])
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                train_acc.update_state(y, logits)
                ce = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True))
                l2 = sum([args.cl * tf.nn.l2_loss(v) for v in model.trainable_variables])
                loss = ce + l2
            grads = tape.gradient(loss, model.trainable_variables)
            grads_tpl = grads
            g = gradtovec(grads)
            loss_round += float(loss.numpy()) / M

            if mode in ["bbit", "bin"]:
                g_eff = g + ef_res[m]
                q = quant_rel_per_tensor(g_eff, ref_up[m], b_up, shapes)
                g_eff_buf[m] = g_eff; q_buf[m] = q
                e[m] = float(np.dot(q - g_eff, q - g_eff))
                rhs = (me_k / (args.lr * args.lr * M * M)) + 3.0 * (e[m] + ehat[m])
                pass_thr = (float(np.dot(q, q)) >= thr_scale * rhs) or (k < warmup) or (clock[m] >= C)
                if pass_thr:
                    cand_idx.append(m)
                    cand_gain.append(float(np.dot(q, q)))
                    cand_cost.append(b_up * nv)
            elif mode == "laq8":
                q = laq_per_vector(g, 8)  # LAQ：整向量缩放
                q_buf[m] = q
                cand_idx.append(m); cand_gain.append(float(np.dot(q, q))); cand_cost.append(8 * nv)
            else:  # fedavg
                q = g; q_buf[m] = q
                cand_idx.append(m); cand_gain.append(float(np.dot(q, q))); cand_cost.append(32 * nv)

        # 还原全局权重
        vec_to_weights(w_global, model.trainable_variables)

        # ---------- 预算贪心选择 ----------
        order = np.argsort(np.array(cand_gain) / (np.array(cand_cost) + 1e-12))[::-1]
        selected = []
        if sel_clients > 0:
            selected = list(order[:min(sel_clients, len(order))])
        elif up_budget_bits > 0.0:
            budget = up_budget_bits
            for i in order:
                if cand_cost[i] <= budget:
                    selected.append(i); budget -= cand_cost[i]
        else:
            selected = list(order)

        # ---------- Fig3：记录本轮 1 个二值位（仅 bin） ----------
        # if mode == "bin":
        #     if len(selected) > 0:
        #         m0 = cand_idx[selected[0]]
        #         # 取该端“相对更新”的第 0 维的符号，映射到 {0,1}
        #         diff0 = (q_buf[m0] - ref_up[m0])[0]
        #         bin_series.append(1 if diff0 >= 0 else 0)
        #     else:
        #         bin_series.append(bin_series[-1] if bin_series else 0)
        
        if mode == "bin":
            if len(selected) > 0:
                # 本轮聚合前的“候选和”，用于选 j*
                agg_tmp = np.zeros(nv, np.float32)
                for idx in selected:
                    m = cand_idx[idx]; agg_tmp += q_buf[m]
                if j_star is None:
                    j_star = int(np.argmax(np.abs(agg_tmp)))  # 第一次确定后固定

                # 多数表决：看被选端在 j* 的相对更新符号
                votes = []
                for idx in selected:
                    m = cand_idx[idx]
                    diff_j = (q_buf[m][j_star] - ref_up[m][j_star])
                    votes.append(1.0 if diff_j >= 0.0 else -1.0)
                b = 1 if (np.mean(votes) >= 0.0) else 0
                bin_series.append(b)
            else:
                bin_series.append(bin_series[-1] if bin_series else 0)

        # ---------- 聚合与更新 ----------
        bits_up_this = 0.0
        if len(selected) > 0:
            agg = np.zeros(nv, np.float32)
            for idx in selected:
                m = cand_idx[idx]; q = q_buf[m]
                agg += q; bits_up_this += cand_cost[idx]
                if mode in ["bbit", "bin"]:
                    ref_up[m] = q
                    ef_res[m] = g_eff_buf[m] - q
                    ehat[m] = e[m]
                clock[m] = 0

            m_sel = float(len(selected))
            g_hat = agg / max(1.0, m_sel)
            scale_sel = (m_sel / max(1.0, sel_ref)) if scale_by_selected else 1.0
            cc = vectograd(g_hat * scale_sel, grads_tpl)

            if clip_global > 0:
                gnorm = np.sqrt(sum(tf.reduce_sum(gi**2) for gi in cc))
                if gnorm > clip_global:
                    s = clip_global / (gnorm + 1e-12)
                    cc = [gi * s for gi in cc]
            optimizer.apply_gradients(zip(cc, model.trainable_variables))

        # 未选端
        picked = set(cand_idx[i] for i in selected)
        for m in range(M):
            if m in picked: continue
            if mode in ["bbit", "bin"] and g_eff_buf[m] is not None:
                ef_res[m] = g_eff_buf[m]
            clock[m] = min(clock[m] + 1, C + 1)

        # ----- 比特统计 -----
        selcnt = len(selected)
        bits_down_this = (b_down if b_down > 0 else 32) * nv * selcnt
        cum_up += bits_up_this; cum_down += bits_down_this

        # ----- 测试 -----
        test_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        ent_sum = 0.0; ntest = 0
        for xi, yi in Datate:
            logits = model(xi, training=False)
            test_acc.update_state(yi, logits)
            ent = tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(
                yi, logits, from_logits=True)).numpy()
            ent_sum += float(ent); ntest += int(yi.shape[0])
        entropy = ent_sum / max(1, ntest)

        # ----- 记录与打印 -----
        loss_hist[k]    = loss_round
        acc_hist[k]     = float(test_acc.result().numpy())
        entropy_hist[k] = float(entropy)
        selcnt_h[k]     = selcnt
        bits_up_cum[k]   = cum_up
        bits_down_cum[k] = cum_down

        print(f"[{k+1}/{K}] acc={acc_hist[k]:.4f} entropy={entropy_hist[k]:.4f} "
              f"sel={selcnt}/{len(cand_idx)}/{M} | bits↑Σ={cum_up:.2e} bits↓Σ={cum_down:.2e}")

    return {
        "loss": loss_hist, "acc": acc_hist, "entropy": entropy_hist,
        "selcnt": selcnt_h, "bits_up_cum": bits_up_cum, "bits_down_cum": bits_down_cum,
        "bin_series": np.array(bin_series, dtype=np.int8) if mode == "bin" else None
    }



# ------------------------- main -------------------------
def parse_args():
    p = argparse.ArgumentParser("FLQ v3 per-tensor quant + downlink quant + budget")
    # data
    p.add_argument("--dataset", type=str, default="mnist",
                   choices=["mnist", "fmnist", "fashion", "fashion_mnist"])
    p.add_argument("--M", type=int, default=10)
    p.add_argument("--iters", type=int, default=800)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--partition", type=str, default="non_iid", choices=["iid","non_iid"])
    p.add_argument("--dir_alpha", type=float, default=0.1)
    # modes
    p.add_argument("--mode", type=str, default="bbit", choices=["fedavg","bbit","bin","laq8"])
    p.add_argument("--b", type=int, default=8, help="uplink bits; bin forces 1")
    p.add_argument("--b_down", type=int, default=8, help="downlink bits; 0/32 = FP32")
    # opt
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cl", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clip_global", type=float, default=0.0)
    # budget
    p.add_argument("--sel_clients", type=int, default=0)
    p.add_argument("--up_budget_bits", type=float, default=17000000.0)
    # lazy
    p.add_argument("--D", type=int, default=10)
    p.add_argument("--ck", type=float, default=0.8)
    p.add_argument("--C", type=int, default=1000000000)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--thr_scale", type=float, default=0.0)
    p.add_argument("--scale_by_selected", type=int, default=1,
               help="按被选端数量放大步长")
    p.add_argument("--sel_ref", type=float, default=1.0,
               help="参考端数，默认为 1（等价于 LAQ 的规模）")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    tf.random.set_seed(args.seed); np.random.seed(args.seed)
    hist = run(args)
    outfile = f"results_{args.dataset}_{args.mode}.xlsx"
    save_excel_data(outfile, args.mode, hist, bin_series=hist.get("bin_series"))
