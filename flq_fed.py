# flq_fed.py
from __future__ import annotations
import argparse, time, numpy as np, tensorflow as tf
import matplotlib.pyplot as plt

# 可选：pandas 用于导出 Excel（若无则自动跳过导出）
try:
    import pandas as pd
except Exception:
    pd = None

# --- TF1 兼容：启用 eager ---
if tf.__version__.startswith("1."):
    tf.compat.v1.enable_eager_execution()

# ---------- 复用自 FLQ.py：向量化与量化 ----------
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
        out.append(vec[p:p+cnt].reshape(shp).astype(np.float32)); p += cnt
    return out

def quantd(vec, ref, b):
    """相对均匀量化：对 vec 相对 ref 做 b-bit 量化（与 FLQ.py 同型）"""
    diff = vec - ref
    r = np.max(np.abs(diff)) + 1e-12
    L = np.floor(2 ** b) - 1
    delta = r / (L + 1e-12)
    q = ref - r + 2 * delta * np.floor((diff + r + delta) / (2 * delta))
    return q.astype(np.float32)

def laq_quantize_stochastic(g, k=8):
    """LAQ：对称均匀量化 + 随机舍入（无偏）"""
    L = 2**(k-1) - 1
    s = (np.max(np.abs(g)) + 1e-12) / L
    y = g / s
    low = np.floor(y)
    p = y - low
    rnd = (np.random.rand(*y.shape) < p).astype(np.float32)
    q_int = np.clip(low + rnd, -L, L)
    return (q_int * s).astype(np.float32)

# ---------- 数据与模型 ----------
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
        ).shuffle(1024).repeat().batch(batch)
        dss.append(ds); iters.append(iter(ds))
    return dss, iters

def split_dirichlet(x, y, M, alpha=0.3, batch=128):
    y = np.array(y); C = int(np.max(y) + 1)
    idx_by_c = [np.where(y == c)[0] for c in range(C)]
    for arr in idx_by_c: np.random.shuffle(arr)
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
        ).shuffle(1024).repeat().batch(batch)
        dss.append(ds); iters.append(iter(ds))
    return dss, iters

def build_model(l2=0.01, lr=0.02):
    reg = tf.keras.regularizers.L2(l2)
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28,28,1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, kernel_regularizer=reg)  # logits
    ])
    opt = tf.keras.optimizers.SGD(learning_rate=lr)
    return model, opt

# ---------- 三张图的绘制函数 ----------
def plot_fig2(loss_dict):
    """loss_dict: {mode: (iters_array, loss_array)}"""
    plt.figure()
    for k, (xs, ys) in loss_dict.items():
        plt.plot(xs, ys, label=k.upper())
    plt.xlabel("Iterations"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()

def plot_fig3_binary_series(bits01, title="Binary values over communications"):
    """bits01: 长度T的0/1序列（取某一组件的符号）"""
    xs = np.arange(len(bits01))
    plt.figure()
    plt.scatter(xs, bits01, s=12)
    plt.axhline(0.5, color="r", linewidth=1)
    plt.xlabel("Communication"); plt.ylabel("Binary values"); plt.title(title); plt.tight_layout()

def plot_fig4_quant_scatter(gt, k_list=(4,8)):
    """gt: 一维梯度样本数组。绘制不同k的量化近似与RMSE"""
    xs = np.linspace(0, len(gt)-1, len(gt))
    for k in k_list:
        q = laq_quantize_stochastic(gt.copy(), k=k)  # 用对称均匀近似展示
        rmse = np.sqrt(np.mean((q-gt)**2))
        plt.figure()
        plt.scatter(np.abs(gt), q, s=5)
        plt.plot(np.abs(gt), gt, 'r', linewidth=1)
        plt.xlabel("||x - y||_2 (proxy)"); plt.ylabel("Approximations")
        plt.title(f"FLQ with k = {k}, RMSE = {rmse:.4f}")
        plt.tight_layout()
        
        


# ---------- 训练主体 ----------
def run_one(mode, args):
    (xtr, ytr), (xte, yte) = load_dataset(args.dataset)
    if args.partition == "iid":
        dss, iters = split_iid(xtr, ytr, args.M, args.batch)
    else:
        dss, iters = split_dirichlet(xtr, ytr, args.M, alpha=args.dir_alpha, batch=args.batch)
    test_ds = tf.data.Dataset.from_tensor_slices(
        (tf.convert_to_tensor(xte), tf.convert_to_tensor(yte))
    ).batch(512)

    model, optimizer = build_model(l2=args.cl, lr=args.lr)

    nv = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    M, D, C, alpha = args.M, args.D, args.C, args.lr

    # 懒惰缓冲
    clock = np.zeros(M, dtype=np.int32)
    e = np.zeros(M, dtype=np.float32)
    ehat = np.zeros(M, dtype=np.float32)
    theta = np.zeros(nv, dtype=np.float32)
    dtheta_hist = np.zeros((nv, D), dtype=np.float32)
    mgr = np.zeros((M, nv), dtype=np.float32)
    dL = np.zeros((M, nv), dtype=np.float32)

    Loss = np.zeros(args.iters, dtype=np.float32)
    CommUp = np.zeros(args.iters, dtype=np.float64)   # 累计上行“次数”
    BitsUp = np.zeros(args.iters, dtype=np.float64)   # 累计上行比特
    BitsDown = np.zeros(args.iters, dtype=np.float64) # 累计下行比特
    acc_eval = np.zeros(args.iters, dtype=np.float32)

    # 记录用于 Fig.3/4 的样本
    bin_series = []
    grad_sample_pool = []

    for k in range(args.iters):
        # 维护 dtheta
        var_vec = gradtovec(model.trainable_variables)
        if k > 0:
            dtheta = var_vec - theta
            dtheta_hist = np.roll(dtheta_hist, 1, axis=1)
            dtheta_hist[:, 0] = dtheta
        theta = var_vec

        sel_mask = np.zeros(M, dtype=bool)
        for m in range(M):
            images, labels = next(iters[m])
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                ce = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))
                l2loss = sum([args.cl * tf.nn.l2_loss(v) for v in model.trainable_variables])
                loss_value = ce + l2loss
            grads = tape.gradient(loss_value, model.trainable_variables)
            gvec = gradtovec(grads)

            # 采样一维用于 Fig.3 序列（取 worker0 的第一个分量符号）
            if m == 0:
                bin_series.append(1 if gvec[0] >= 0 else 0)
            # 收集少量梯度样本用于 Fig.4
            if k % max(1, args.eval_every) == 0 and len(grad_sample_pool) < 2000:
                grad_sample_pool.extend(gvec[:min(50, len(gvec))])

            if mode == "flq_bin":
                vec_q = quantd(gvec, mgr[m], args.b_up)      # 上行按 b_up=1 或者小位宽
                diff = vec_q - gvec
                e[m] = float(diff @ diff)

                # me[m]
                weights = 1.0 / (np.arange(D) + 1.0)
                me_m = float(np.sum((dtheta_hist ** 2).sum(axis=0) * weights))

                dL[m] = vec_q - mgr[m]
                th = (me_m / (alpha * alpha * M * M)) + 3.0 * (e[m] + ehat[m])

                if (float(dL[m] @ dL[m]) >= th) or (clock[m] >= C):
                    sel_mask[m] = True
                    mgr[m] = vec_q; ehat[m] = e[m]; clock[m] = 0
                else:
                    clock[m] += 1

            elif mode == "flq_lowbit":
                # 双向低比特，无懒惰
                dL[m] = laq_quantize_stochastic(gvec, k=args.b_up)  # 这里沿用8-bit
                sel_mask[m] = True

            elif mode == "laq8":
                dL[m] = laq_quantize_stochastic(gvec, k=8)
                sel_mask[m] = True

            else:  # qgd
                dL[m] = gvec
                sel_mask[m] = True

        # 聚合更新（等权；需要 FedAvg 可改为样本数加权）
        dsa = dL[sel_mask].sum(axis=0) if np.any(sel_mask) else np.zeros(nv, dtype=np.float32)
        ccgrads = vectograd(dsa, grads)
        optimizer.apply_gradients(zip(ccgrads, model.trainable_variables))

        # 统计比特：下行按 bit_down * nv * M；上行按 bit_up * nv * sel_count
        sel_cnt = float(sel_mask.sum())
        bit_down = (8 if mode in ["flq_bin", "flq_lowbit"] else (8 if mode=="laq8" and args.down_laq8==8 else 32)) * nv * M
        if mode == "flq_bin":
            bit_up = args.b_up * nv * sel_cnt                   # 典型 b_up=1
        elif mode in ["flq_lowbit", "laq8"]:
            bit_up = (8 if mode!="laq8" else 8) * nv * M        # 每轮全体上传
        else:
            bit_up = 32 * nv * M

        Loss[k] = float(loss_value.numpy())
        CommUp[k] = (0 if k == 0 else CommUp[k-1]) + sel_cnt
        BitsUp[k] = (0 if k == 0 else BitsUp[k-1]) + bit_up
        BitsDown[k] = (0 if k == 0 else BitsDown[k-1]) + bit_down

        if (k + 1) % max(1, args.eval_every) == 0:
            acc = tf.keras.metrics.SparseCategoricalAccuracy()
            for xi, yi in test_ds:
                acc.update_state(yi, model(xi, training=False))
            acc_eval[k] = acc.result().numpy()
            print(f"[{k+1}/{args.iters}] mode={mode} loss={Loss[k]:.4f} acc={acc_eval[k]:.4f} up_comm={CommUp[k]:.0f} up_bits={BitsUp[k]:.2e} down_bits={BitsDown[k]:.2e}")

    # --- 导出 Excel（可选） ---
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
        fn = f"./flq_results_{args.dataset}.xlsx"
        with pd.ExcelWriter(fn) as xw:
            df_curve.to_excel(xw, sheet_name=f"curve_{mode}", index=False)
            # Fig.3 原始二值序列
            pd.DataFrame({"comm": np.arange(len(bin_series)),
                        "bit": np.array(bin_series, dtype=int)}
                        ).to_excel(xw, sheet_name=f"bin_{mode}", index=False)
            # Fig.4 梯度采样
            if len(grad_sample_pool) > 0:
                pd.DataFrame({"gt": np.array(grad_sample_pool, dtype=np.float32)}
                            ).to_excel(xw, sheet_name=f"gt_{mode}", index=False)
            # Table I 采样点（如需）
            for it in [1000, 1500, 3000]:
                if it <= args.iters:
                    pd.DataFrame([{
                        "Method": mode, "Iteration": it,
                        "Broadcast(bits)": BitsDown[it-1],
                        "Upload(bits)": BitsUp[it-1],
                        "Accuracy(%)": float(acc_eval[it-1]*100.0)
                    }]).to_excel(xw, sheet_name=f"table_{mode}_{it}", index=False)
    # if pd is not None:
    #     df_curve = pd.DataFrame({
    #         "iter": np.arange(1, args.iters+1),
    #         "loss": Loss,
    #         "acc_eval": acc_eval,
    #         "cum_uploads": CommUp,
    #         "cum_bits_up": BitsUp,
    #         "cum_bits_down": BitsDown,
    #         "cum_bits_total": BitsUp + BitsDown
    #     })
    #     fn = f"./flq_results_{args.dataset}.xlsx"
    #     with pd.ExcelWriter(fn, engine="openpyxl" if "openpyxl" in (pd.__dict__.get("__all__", []) or []) else None, mode="w") as xw:
    #         df_curve.to_excel(xw, sheet_name=f"curve_{mode}", index=False)
    #         # Table1-like 采样：用户可改 checkpoints
    #         for it in [1000, 1500, 3000]:
    #             if it <= args.iters:
    #                 df_tbl = pd.DataFrame([{
    #                     "Method": mode, "Iteration": it,
    #                     "Broadcast(bits)": BitsDown[it-1],
    #                     "Upload(bits)": BitsUp[it-1],
    #                     "Accuracy(%)": float(acc_eval[it-1]*100.0)
    #                 }])
    #                 df_tbl.to_excel(xw, sheet_name=f"table_{mode}_{it}", index=False)
    #     print(f"Excel saved: {fn}")

    # --- 三张图 ---
    plot_fig2({mode: (np.arange(1, args.iters+1), Loss)})
    plot_fig3_binary_series(np.array(bin_series[:args.iters]), title="Results of gradient quantification in communication")
    gt = np.array(grad_sample_pool[:2000], dtype=np.float32)
    if gt.size > 0:
        plot_fig4_quant_scatter(gt, k_list=(4,8))

    return {
        "loss": Loss, "iters": np.arange(1, args.iters+1),
        "bits_up": BitsUp, "bits_down": BitsDown, "uploads": CommUp
    }

# ---------- 主入口 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist","fmnist"], default="mnist")
    ap.add_argument("--mode", choices=["flq_bin","flq_lowbit","laq8","qgd"], default="flq_bin")
    ap.add_argument("--partition", choices=["iid","dir"], default="iid")
    ap.add_argument("--dir_alpha", type=float, default=0.3, help="Dirichlet alpha for Non-IID")
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--b_up", type=int, default=1, help="uplink bitwidth for FLQ(binary)")
    ap.add_argument("--C", type=int, default=100)      # 强制通信周期
    ap.add_argument("--D", type=int, default=10)       # 阈值历史窗口
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--cl", type=float, default=0.01)  # L2
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--down_laq8", type=int, default=32, help="LAQ下行位宽，默认32以匹配论文Table口径")
    args = ap.parse_args()

    tic = time.time()
    run_one(args.mode, args)
    print(f"Runtime {time.time()-tic:.2f}s")
    plt.show()

if __name__ == "__main__":
    main()
