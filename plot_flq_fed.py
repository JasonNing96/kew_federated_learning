import pandas as pd
import matplotlib.pyplot as plt

def draw_fig1_convergence(excel_path, modes, x="iter", y="loss",
                          use_bits=False, which_bits="cum_bits_total",
                          title="Comparison of convergence"):
    plt.figure()
    for m in modes:
        df = pd.read_excel(excel_path, sheet_name=f"curve_{m}")
        xs = df[which_bits] if use_bits else df[x]
        plt.plot(xs, df[y], label=m.upper())
    plt.xlabel("Cumulative bits" if use_bits else "Iterations")
    plt.ylabel(y.capitalize()); plt.title(title); plt.legend(); plt.tight_layout()

def draw_fig2_binary_series(excel_path, mode, title="Results of gradient quantification"):
    df = pd.read_excel(excel_path, sheet_name=f"bin_{mode}")
    plt.figure()
    plt.scatter(df["comm"], df["bit"], s=10)
    plt.axhline(0.5, color="r", linewidth=1)
    plt.xlabel("Communication"); plt.ylabel("Binary values"); plt.title(title); plt.tight_layout()

def draw_fig3_quant_scatter(excel_path, mode, k_list=(4,8), title_prefix="Quantification"):
    df = pd.read_excel(excel_path, sheet_name=f"gt_{mode}")
    gt = df["gt"].values.astype("float32")
    # 简单双边对称量化（与 LAQ 一致）
    def laq_quantize_stochastic(g, k=8):
        import numpy as np
        L = 2**(k-1) - 1
        s = (np.max(np.abs(g)) + 1e-12) / L
        y = g / s
        low = np.floor(y)
        p = y - low
        rnd = (np.random.rand(*y.shape) < p).astype(np.float32)
        q_int = np.clip(low + rnd, -L, L)
        return (q_int * s).astype(np.float32)
    for k in k_list:
        q = laq_quantize_stochastic(gt, k=k)
        import numpy as np
        rmse = float(np.sqrt(np.mean((q-gt)**2)))
        plt.figure()
        plt.scatter(np.abs(gt), q, s=6)
        plt.plot(np.abs(gt), gt, 'r', linewidth=1)
        plt.xlabel("||x - y||_2 (proxy)"); plt.ylabel("Approximations")
        plt.title(f"{title_prefix} with k={k}, RMSE={rmse:.4f}")
        plt.tight_layout()

if __name__ == "__main__":
    draw_fig1_convergence("flq_fed_mnist.xlsx", ["sign1", "int8", "4bit"], use_bits=True, which_bits="cum_bits_total")
    draw_fig2_binary_series("flq_fed_mnist.xlsx", "sign1")
    draw_fig3_quant_scatter("flq_fed_mnist.xlsx", "sign1")
    plt.show()  