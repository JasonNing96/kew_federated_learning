# plot_flq_fed.py
import argparse, pandas as pd, matplotlib.pyplot as plt, numpy as np, os

MODES_DEFAULT = ["qgd","laq8","flq_lowbit","flq_bin"]

def draw_fig1(excel_map, use_bits=False, which_bits="cum_bits_total",
              y="loss", title="Convergence"):
    plt.figure()
    for name, path in excel_map.items():
        df = pd.read_excel(path, sheet_name=f"curve_{name}")
        xs = df[which_bits] if use_bits else df["iter"]
        plt.plot(xs, df[y], label=name.upper())
    plt.xlabel("Cumulative bits" if use_bits else "Iterations")
    plt.ylabel(y.capitalize()); plt.title(title); plt.legend(); plt.tight_layout()

def draw_fig2(excel_path, mode):
    df = pd.read_excel(excel_path, sheet_name=f"bin_{mode}")
    plt.figure()
    plt.scatter(df["comm"], df["bit"], s=10)
    plt.axhline(0.5, color="r", linewidth=1)
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

def draw_fig3(excel_path, mode, k_list=(4,8)):
    df = pd.read_excel(excel_path, sheet_name=f"gt_{mode}")
    gt = df["gt"].values.astype("float32")
    for k in k_list:
        q = laq_quantize_stochastic(gt, k=k)
        rmse = float(np.sqrt(np.mean((q-gt)**2)))
        plt.figure()
        plt.scatter(np.abs(gt), q, s=6)
        plt.plot(np.abs(gt), gt, 'r', linewidth=1)
        plt.xlabel("||x - y||"); plt.ylabel("Approximations")
        plt.title(f"FLQ with k={k}, RMSE={rmse:.4f}")
        plt.tight_layout()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel_dir", type=str, default=".")
    ap.add_argument("--dataset", choices=["mnist","fmnist"], default="mnist")
    ap.add_argument("--modes", nargs="+", default=MODES_DEFAULT)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--prefix", type=str, default="results")
    args = ap.parse_args()

    excel_map = {}
    for m in args.modes:
        path = os.path.join(args.excel_dir, f"{args.prefix}_{args.dataset}_{m}.xlsx")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        excel_map[m] = path

    # Fig1：按迭代与按累计比特各一张
    draw_fig1(excel_map, use_bits=False, title="Comparison of convergence (iterations)")
    if args.save: plt.savefig(f"Fig1_iter_{args.dataset}.png", dpi=200)
    draw_fig1(excel_map, use_bits=True, title="Comparison of convergence (bits)")
    if args.save: plt.savefig(f"Fig1_bits_{args.dataset}.png", dpi=200)

    # Fig2：二值序列（取 flq_bin）
    if "flq_bin" in excel_map:
        draw_fig2(excel_map["flq_bin"], "flq_bin")
        if args.save: plt.savefig(f"Fig2_binary_{args.dataset}.png", dpi=200)

    # Fig3：k=4/8 量化散点与 RMSE（用 flq_bin 的梯度样本）
    if "flq_bin" in excel_map:
        draw_fig3(excel_map["flq_bin"], "flq_bin", k_list=(4,8))
        if args.save:
            plt.savefig(f"Fig3_k4_{args.dataset}.png", dpi=200)
            plt.savefig(f"Fig3_k8_{args.dataset}.png", dpi=200)

    if not args.save:
        plt.show()

if __name__ == "__main__":
    main()
