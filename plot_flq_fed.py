import argparse, pandas as pd, matplotlib.pyplot as plt, numpy as np, os

# 模式别名
MODE_ALIASES = {
    "qgd": "fedavg", "fedavg": "fedavg",
    "flq_lowbit": "bbit", "bbit": "bbit",
    "flq_bin": "bin", "bin": "bin",
    "laq8": "laq8"
}
MODES_DEFAULT = ["fedavg", "bbit", "bin"]

def draw_fig1(excel_map, use_bits=False, which_bits="cum_bits_total",
              y="loss", title="Convergence"):
    plt.figure()
    for name, path in excel_map.items():
        df = pd.read_excel(path, sheet_name=f"curve_{name}")
        xs = df[which_bits] if use_bits else df["iter"]
        plt.plot(xs, df[y], label=name.upper())
    plt.xlabel("Cumulative bits" if use_bits else "Iterations")
    plt.ylabel(y.capitalize()); plt.title(title); plt.legend(); plt.tight_layout()

def draw_binary_series(excel_path, mode):
    try:
        df = pd.read_excel(excel_path, sheet_name=f"bin_{mode}")
    except Exception:
        return
    plt.figure()
    plt.scatter(df["comm"], df["bit"], s=10)
    plt.axhline(0.5, linewidth=1)
    plt.xlabel("Communication"); plt.ylabel("Binary values")
    plt.title("Results of gradient quantification in communication")
    plt.tight_layout()

def laq_quantize_stochastic(g, k=8):
    L = 2**(k-1) - 1
    s = (np.max(np.abs(g)) + 1e-12) / L
    y = g / s; low = np.floor(y); p = y - low
    rnd = (np.random.rand(*y.shape) < p).astype(np.float32)
    q_int = np.clip(low + rnd, -L, L)
    return (q_int * s).astype(np.float32)

def draw_kbit_rmse(excel_path, mode, k_list=(4,8)):
    try:
        df = pd.read_excel(excel_path, sheet_name=f"gt_{mode}")
    except Exception:
        return
    gt = df["gt"].values.astype("float32")
    for k in k_list:
        q = laq_quantize_stochastic(gt, k=k)
        rmse = float(np.sqrt(np.mean((q-gt)**2)))
        plt.figure()
        plt.scatter(np.abs(gt), q, s=6)
        plt.plot(np.abs(gt), gt)
        plt.xlabel("||x - y||"); plt.ylabel("Approximations")
        plt.title(f"FLQ with k={k}, RMSE={rmse:.4f}")
        plt.tight_layout()

# # ===== 论文 Fig.2：Iterations vs Loss[entropy]，QGD/LAQ/FLQ(Ours) =====
# def draw_paper_fig2_iter(excel_map, max_iter=800):
#     # 顺序与标签固定：QGD(=fedavg), LAQ(=laq8), FLQ(Ours)(=bin 上行 + 8-bit 下发)
#     order = [("fedavg", "QGD", "-"),
#              ("laq8",   "LAQ", "--"),
#              ("bin",    "FLQ(Ours)", "-")]
#     plt.figure()
#     for key, label, ls in order:
#         if key not in excel_map:  # 未提供该曲线则跳过
#             continue
#         df = pd.read_excel(excel_map[key], sheet_name=f"curve_{key}")
#         df = df.iloc[:max_iter]
#         plt.plot(df["iter"], df["loss"], linestyle=ls, label=label)
#     plt.xlabel("Iterations#")
#     plt.ylabel("Loss [entropy]")
#     plt.xlim(0, max_iter)
#     plt.legend()
#     plt.tight_layout()

# ===== 论文 Fig.2：Iterations vs Loss[entropy]，QGD / LAQ / FLQ(Ours) =====
def draw_paper_fig2_iter(excel_map, max_iter=800):
    import pandas as pd
    import matplotlib.pyplot as plt

    # 选择 FLQ 曲线：优先 bbit，无则退回 bin
    flq_key = "bbit" if "bbit" in excel_map else ("bin" if "bin" in excel_map else None)

    order = [("fedavg", "QGD", "-"),
             ("laq8",   "LAQ",  "--")]
    if flq_key:
        order.append((flq_key, "FLQ(Ours)", "-"))

    plt.figure()
    for key, label, ls in order:
        if key not in excel_map:
            continue
        df = pd.read_excel(excel_map[key], sheet_name=f"curve_{key}")
        df = df.iloc[:max_iter]
        ycol = "entropy" if "entropy" in df.columns else "loss"
        plt.plot(df["iter"], df[ycol], linestyle=ls, label=label)

    plt.xlabel("Iterations#")
    plt.ylabel("Loss [entropy]")
    plt.xlim(0, max_iter)
    plt.legend()
    plt.tight_layout()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel_dir", type=str, default=".")
    ap.add_argument("--dataset", choices=["mnist","fmnist"], default="mnist")
    ap.add_argument("--modes", nargs="+", default=MODES_DEFAULT)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--prefix", type=str, default="results")
    ap.add_argument("--max_iter", type=int, default=800)  # 论文上限
    args = ap.parse_args()

    # 收集 Excel
    excel_map = {}
    for m in args.modes:
        mode_key = MODE_ALIASES.get(m.lower(), m.lower())
        path = os.path.join(args.excel_dir, f"{args.prefix}_{args.dataset}_{mode_key}.xlsx")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        excel_map[mode_key] = path

    # ——论文 Fig.2——
    draw_paper_fig2_iter(excel_map, max_iter=args.max_iter)
    if args.save: plt.savefig(f"Fig2_{args.dataset}.png", dpi=300)

    # 其余可选图：保留
    # draw_fig1(excel_map, use_bits=False, title="Comparison of convergence (iterations)")
    # if args.save: plt.savefig(f"Fig1_iter_{args.dataset}.png", dpi=200)
    # draw_fig1(excel_map, use_bits=True, title="Comparison of convergence (bits)")
    # if args.save: plt.savefig(f"Fig1_bits_{args.dataset}.png", dpi=200)

    if "bin" in excel_map:
        draw_binary_series(excel_map["bin"], "bin")
        if args.save: plt.savefig(f"Fig_binary_series_{args.dataset}.png", dpi=200)
        draw_kbit_rmse(excel_map["bin"], "bin", k_list=(4,8))
        if args.save:
            plt.savefig(f"Fig_k4_{args.dataset}.png", dpi=200)
            plt.savefig(f"Fig_k8_{args.dataset}.png", dpi=200)

    if not args.save:
        plt.show()

if __name__ == "__main__":
    main()
