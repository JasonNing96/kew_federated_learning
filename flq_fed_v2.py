# flq_fed_v2.py
from __future__ import annotations
import argparse, time, numpy as np, tensorflow as tf, os, random

# optional Excel export
try:
    import pandas as pd
except Exception:
    pd = None

# TF1 eager compat
if tf.__version__.startswith("1."):
    tf.compat.v1.enable_eager_execution()

# ---------------- utils: vectorize / de-vectorize ----------------
def gradtovec(grad_list):
    vs = []
    for g in grad_list:
        a = g.numpy()
        vs.append(a.reshape(-1))
    return np.concatenate(vs).astype(np.float32)

def vectograd(vec, tmpl_list):
    out, p = [], 0
    for t in tmpl_list:
        shp = t.shape.as_list() if hasattr(t.shape, "as_list") else list(t.shape)
        cnt = int(np.prod(shp))
        out.append(vec[p:p+cnt].reshape(shp).astype(np.float32))
        p += cnt
    return out

# ---------------- quantizers ----------------
def quantd(vec, ref, b):
    """Relative uniform quantization (low-bit, b>=2)."""
    diff = vec - ref
    r = np.max(np.abs(diff)) + 1e-12
    L = np.floor(2 ** b) - 1
    delta = r / (L + 1e-12)
    q = ref - r + 2 * delta * np.floor((diff + r + delta) / (2 * delta))
    return q.astype(np.float32)

def bin_relative_quant(gvec, ref):
    """Relative binary quantization with alpha = mean|diff|."""
    diff = gvec - ref
    alpha = np.mean(np.abs(diff)).astype(np.float32)
    vec_q = ref + alpha * np.sign(diff).astype(np.float32)
    qerr  = np.sum((vec_q - gvec)**2).astype(np.float32)
    return vec_q, qerr

def laq_quantize_stochastic(g, k=8):
    """LAQ: symmetric k-bit with stochastic rounding (unbiased)."""
    L = 2**(k-1) - 1
    s = (np.max(np.abs(g)) + 1e-12) / L
    y = g / s
    low = np.floor(y)
    p = y - low
    rnd = (np.random.rand(*y.shape) < p).astype(np.float32)
    q_int = np.clip(low + rnd, -L, L)
    return (q_int * s).astype(np.float32)

# ---------------- data & model ----------------
def load_dataset(name="mnist"):
    if name == "fmnist":
        (xtr, ytr), (xte, yte) = tf.keras.datasets.fashion_mnist.load_data()
    else:
        (xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()
    xtr = (xtr / 255.0).astype("float32")[..., np.newaxis]
    xte = (xte / 255.0).astype("float32")[..., np.newaxis]
    return (xtr, ytr), (xte, yte)

def split_iid(x, y, M, batch):
    n = len(x); per = n // M
    dss, iters = [], []
    for m in range(M):
        ds = tf.data.Dataset.from_tensor_slices(
            (tf.convert_to_tensor(x[m*per:(m+1)*per]),
             tf.convert_to_tensor(y[m*per:(m+1)*per]))
        ).shuffle(2048).repeat().batch(batch)
        dss.append(ds); iters.append(iter(ds))
    return dss, iters

def split_dirichlet(x, y, M, alpha=0.3, batch=128):
    y = np.array(y); C = int(np.max(y) + 1)
    idx_by_c = [np.where(y == c)[0] for c in range(C)]
    for arr in idx_by_c:
        np.random.shuffle(arr)
    parts = [[] for _ in range(M)]
    for idx in idx_by_c:
        props = np.random.dirichlet([alpha] * M)
        cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        chunks = np.split(idx, cuts)
        for m in range(M): parts[m].extend(chunks[m])
    dss, iters = [], []
    for m in range(M):
        sel = np.array(sorted(parts[m]))
        ds = tf.data.Dataset.from_tensor_slices(
            (tf.convert_to_tensor(x[sel]), tf.convert_to_tensor(y[sel]))
        ).shuffle(2048).repeat().batch(batch)
        dss.append(ds); iters.append(iter(ds))
    return dss, iters

def build_model(l2=0.0005, lr=0.02, model_kind="cnn"):
    reg = tf.keras.regularizers.L2(l2)
    if model_kind == "cnn":
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28,28,1)),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=reg),
            tf.keras.layers.Dense(10)  # logits
        ])
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28,28,1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, kernel_regularizer=reg)
        ])
    opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    return model, opt

# ---------------- one run ----------------
def run_one(mode, args):
    # seeds
    np.random.seed(args.seed); random.seed(args.seed); tf.random.set_seed(args.seed)

    (xtr, ytr), (xte, yte) = load_dataset(args.dataset)
    if args.partition == "iid":
        dss, iters = split_iid(xtr, ytr, args.M, args.batch)
    else:
        dss, iters = split_dirichlet(xtr, ytr, args.M, alpha=args.dir_alpha, batch=args.batch)
    test_ds = tf.data.Dataset.from_tensor_slices(
        (tf.convert_to_tensor(xte), tf.convert_to_tensor(yte))
    ).batch(512)

    model, optimizer = build_model(l2=args.cl, lr=args.lr, model_kind=args.model)

    nv = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    M, D, C, alpha = args.M, args.D, args.C, args.lr

    # 懒惰缓存
    clock = np.zeros(M, dtype=np.int32)
    e = np.zeros(M, dtype=np.float32)
    ehat = np.zeros(M, dtype=np.float32)
    theta = np.zeros(nv, dtype=np.float32)
    dtheta_hist = np.zeros((nv, D), dtype=np.float32)
    mgr = np.zeros((M, nv), dtype=np.float32)          # 每客户端的“已上报参考”
    dL = np.zeros((M, nv), dtype=np.float32)            # 本轮触发的参考增量
    ef_residual = np.zeros((M, nv), dtype=np.float32)   # 误差补偿
    g_hat = np.zeros(nv, dtype=np.float32)              # 服务器端全局量化梯度估计（关键）

    Loss = np.zeros(args.iters, dtype=np.float32)
    CommUp = np.zeros(args.iters, dtype=np.float64)
    BitsUp = np.zeros(args.iters, dtype=np.float64)
    BitsDown = np.zeros(args.iters, dtype=np.float64)
    acc_eval = np.zeros(args.iters, dtype=np.float32)

    bin_series = []
    grad_sample_pool = []

    for k in range(args.iters):
        # 维护 dtheta 历史
        var_vec = gradtovec(model.trainable_variables)
        if k > 0:
            dtheta = var_vec - theta
            dtheta_hist = np.roll(dtheta_hist, 1, axis=1)
            dtheta_hist[:, 0] = dtheta
        theta = var_vec

        sel_mask = np.zeros(M, dtype=bool)

        for m in range(M):
            # 本地多步
            acc_grad = None
            for _ in range(args.local_steps):
                images, labels = next(iters[m])
                with tf.GradientTape() as tape:
                    logits = model(images, training=True)
                    ce = tf.reduce_mean(
                        tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))
                    l2loss = sum([args.cl * tf.nn.l2_loss(v) for v in model.trainable_variables])
                    loss_value = ce + l2loss
                grads = tape.gradient(loss_value, model.trainable_variables)
                if acc_grad is None:
                    acc_grad = [g.numpy() for g in grads]
                else:
                    for i, g in enumerate(grads): acc_grad[i] += g.numpy()
            grads = [tf.convert_to_tensor(g/float(args.local_steps)) for g in acc_grad]
            gvec = gradtovec(grads)

            if m == 0:
                bin_series.append(1 if gvec[0] >= 0 else 0)
            if k % max(1, args.eval_every) == 0 and len(grad_sample_pool) < 2000:
                grad_sample_pool.extend(gvec[:min(50, len(gvec))])

            # 动态门限项 me_m
            weights = 1.0 / (np.arange(D) + 1.0)
            me_m = float(np.sum((dtheta_hist ** 2).sum(axis=0) * weights))

            if mode == "flq_bin":
                g_eff = gvec + ef_residual[m]
                vec_q, e_m = bin_relative_quant(g_eff, mgr[m])
                e[m] = float(e_m)
                dL[m] = vec_q - mgr[m]
                th = (me_m / (alpha * alpha * M * M)) + 3.0 * (e[m] + ehat[m])
                force = (k < args.warmup) or (clock[m] >= C)
                if force or (float(dL[m] @ dL[m]) >= args.thr_scale * th):
                    sel_mask[m] = True
                    mgr[m] = vec_q; ehat[m] = e[m]; clock[m] = 0
                    ef_residual[m] = g_eff - vec_q
                else:
                    clock[m] += 1
                    ef_residual[m] = g_eff

            elif mode == "flq_lowbit":
                g_eff = gvec + ef_residual[m]
                vec_q = quantd(g_eff, mgr[m], b=args.b_up)   # 典型 8-bit
                diff = vec_q - g_eff
                e[m] = float(diff @ diff)
                dL[m] = vec_q - mgr[m]
                th = (me_m / (alpha * alpha * M * M)) + 3.0 * (e[m] + ehat[m])
                force = (k < args.warmup) or (clock[m] >= C)
                if force or (float(dL[m] @ dL[m]) >= args.thr_scale * th):
                    sel_mask[m] = True
                    mgr[m] = vec_q; ehat[m] = e[m]; clock[m] = 0
                    ef_residual[m] = g_eff - vec_q
                else:
                    clock[m] += 1
                    ef_residual[m] = g_eff

            elif mode == "laq8":
                dL[m] = laq_quantize_stochastic(gvec, k=8)
                sel_mask[m] = True

            else:  # qgd
                dL[m] = gvec
                sel_mask[m] = True

        # ——聚合与更新（关键修正）——
        sel_cnt = float(sel_mask.sum())

        if mode in ["flq_bin", "flq_lowbit"]:
            # 仅把触发客户端的参考变动量累计到全局估计；无人触发则保持上轮估计
            if sel_cnt > 0:
                g_hat += dL[sel_mask].sum(axis=0) / M    # 等价：g_hat = mean_m(mgr[m])
            ccgrads = vectograd(g_hat, grads)
        else:
            # QGD/LAQ8：用当轮量化后的梯度均值
            g_hat = dL.mean(axis=0)
            ccgrads = vectograd(g_hat, grads)

        # 可选全局裁剪
        if args.clip_global > 0:
            gnorm = np.sqrt(sum((g**2).sum() for g in ccgrads))
            if gnorm > args.clip_global:
                scale = args.clip_global / (gnorm + 1e-12)
                ccgrads = [g * scale for g in ccgrads]

        optimizer.apply_gradients(zip(ccgrads, model.trainable_variables))

        # 下/上行比特统计
        if mode in ["flq_bin", "flq_lowbit"]:
            bit_down = 8 * nv * M
        elif mode == "laq8":
            bit_down = (args.down_laq8 if args.down_laq8 in [8, 32] else 32) * nv * M
        else:
            bit_down = 32 * nv * M

        if mode == "flq_bin":
            bit_up = args.b_up * nv * sel_cnt
        elif mode == "flq_lowbit":
            bit_up = args.b_up * nv * sel_cnt
        elif mode == "laq8":
            bit_up = 8 * nv * M
        else:
            bit_up = 32 * nv * M

        Loss[k] = float(loss_value.numpy())
        CommUp[k] = (0 if k == 0 else CommUp[k-1]) + sel_cnt
        BitsUp[k] = (0 if k == 0 else BitsUp[k-1]) + bit_up
        BitsDown[k] = (0 if k == 0 else BitsDown[k-1]) + bit_down

        # 评估与日志
        if (k + 1) % max(1, args.eval_every) == 0:
            acc = tf.keras.metrics.SparseCategoricalAccuracy()
            for xi, yi in test_ds:
                acc.update_state(yi, model(xi, training=False))
            acc_eval[k] = acc.result().numpy()
            trig_rate = sel_cnt / M
            print(f"[{k+1}/{args.iters}] mode={mode} acc={acc_eval[k]:.4f} loss={Loss[k]:.4f} sel={int(sel_cnt)} trig={trig_rate:.2f} up_bits={BitsUp[k]:.2e}")

        # 余弦学习率
        if args.lr_min < args.lr:
            t = (k+1)/float(args.iters)
            lr_new = args.lr_min + 0.5*(args.lr - args.lr_min)*(1+np.cos(np.pi*t))
            optimizer.learning_rate.assign(lr_new)

    # 导出 Excel
    if pd is not None:
        df_curve = pd.DataFrame({
            "iter": np.arange(1, args.iters+1),
            "loss": Loss,
            "acc": acc_eval,
            "cum_uploads": CommUp,
            "cum_bits_up": BitsUp,
            "cum_bits_down": BitsDown,
            "cum_bits_total": BitsUp + BitsDown
        })
        fn = f"./{args.out_prefix}_{args.dataset}_{mode}.xlsx"
        with pd.ExcelWriter(fn) as xw:
            df_curve.to_excel(xw, sheet_name=f"curve_{mode}", index=False)
            pd.DataFrame({"comm": np.arange(len(bin_series)),
                          "bit": np.array(bin_series, dtype=int)}
                        ).to_excel(xw, sheet_name=f"bin_{mode}", index=False)
            if len(grad_sample_pool) > 0:
                pd.DataFrame({"gt": np.array(grad_sample_pool, dtype=np.float32)}
                            ).to_excel(xw, sheet_name=f"gt_{mode}", index=False)
        print(f"Excel saved: {fn}")


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist","fmnist"], default="mnist")
    ap.add_argument("--mode", choices=["flq_bin","flq_lowbit","laq8","qgd"], default="flq_bin")
    ap.add_argument("--partition", choices=["iid","dir"], default="iid")
    ap.add_argument("--dir_alpha", type=float, default=0.3)
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--local_steps", type=int, default=2)
    ap.add_argument("--b_up", type=int, default=1, help="uplink bits for FLQ; bin=1, lowbit=8 etc.")
    ap.add_argument("--C", type=int, default=10)
    ap.add_argument("--D", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=50, help="前 warmup 轮强制上传")
    ap.add_argument("--thr_scale", type=float, default=0.5, help="门限缩放系数(<1 放松，>1 收紧)")
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--lr_min", type=float, default=0.005)
    ap.add_argument("--cl", type=float, default=0.0005)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--down_laq8", type=int, default=32)
    ap.add_argument("--clip_global", type=float, default=5.0)
    ap.add_argument("--model", choices=["linear","cnn"], default="cnn")
    ap.add_argument("--out_prefix", type=str, default="results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tic = time.time()
    run_one(args.mode, args)
    print(f"Runtime {time.time()-tic:.2f}s")

if __name__ == "__main__":
    main()
