from flask import Flask, request, jsonify
import threading
import os
import time
import numpy as np

app = Flask(__name__)

# ---------------------
# 可配置参数
# ---------------------
MODEL_DIM = int(os.getenv("MODEL_DIM", 100_000))       # 模型参数量，float32 => ~0.4MB/10万
AGG_MIN_CLIENTS = int(os.getenv("AGG_MIN_CLIENTS", 2)) # 每轮聚合至少等待的客户端数
SEED = int(os.getenv("SEED", 42))

rng = np.random.default_rng(SEED)

# ---------------------
# 全局状态
# ---------------------
status_lock = threading.Lock()

# 全局模型与元信息
global_round = 0
global_weights = rng.standard_normal(MODEL_DIM, dtype=np.float32)

# 每轮待聚合的客户端上报缓冲
pending_updates = {}

# 最近一次各 worker 状态（用于 /status 与 /metrics）
workers_status = {}

# ---------------------
# API
# ---------------------

@app.route("/update", methods=["POST"])
def update():
    global global_round, global_weights   # ✅ 必须放在函数开头

    """
    Worker 上报本地训练结果：
    {
        "worker_id": "string",
        "round": int,
        "num_samples": int,
        "loss": float,
        "delta": [float32...],
        "compute_factor": float
    }
    """
    data = request.get_json(force=True)
    worker_ip = request.remote_addr
    worker_id = data.get("worker_id") or worker_ip

    delta = np.array(data.get("delta", []), dtype=np.float32)
    num_samples = int(data.get("num_samples", 0))
    this_round = int(data.get("round", 0))
    loss = float(data.get("loss", 0.0))
    compute_factor = float(data.get("compute_factor", 0.0))
    ts = time.time()

    if delta.shape[0] != global_weights.shape[0]:
        return jsonify({"status": "error", "msg": f"delta size mismatch: {delta.shape[0]} vs {global_weights.shape[0]}"}), 400

    aggregated = False
    current_global_round = None

    with status_lock:
        # 记录 worker 最新状态
        workers_status[worker_id] = {
            "ip": worker_ip,
            "round": this_round,
            "num_samples": num_samples,
            "loss": loss,
            "compute_factor": compute_factor,
            "last_update_ts": ts
        }

        # 累积到待聚合队列
        bucket = pending_updates.setdefault(this_round, [])
        bucket.append({
            "worker": worker_id,
            "num_samples": num_samples,
            "loss": loss,
            "delta": delta
        })

        # 达到最小客户端数且该轮仍是当前全局轮
        if (this_round == global_round) and (len(bucket) >= AGG_MIN_CLIENTS):
            # FedAvg: w_{t+1} = w_t + sum_k ( (n_k / N_total) * delta_k )
            N_total = sum(u["num_samples"] for u in bucket if u["num_samples"] > 0)
            if N_total > 0:
                agg_delta = np.zeros_like(global_weights)
                for u in bucket:
                    if u["num_samples"] > 0:
                        agg_delta += (u["num_samples"] / N_total) * u["delta"]
                global_weights = global_weights + agg_delta
            else:
                # 如果没有样本数，就简单平均
                agg_delta = np.mean([u["delta"] for u in bucket], axis=0)
                global_weights = global_weights + agg_delta

            global_round += 1
            aggregated = True
            pending_updates.pop(this_round, None)  # 清理该轮缓存

        current_global_round = global_round

    return jsonify({
        "status": "ok",
        "aggregated": aggregated,
        "global_round": current_global_round
    })


@app.route("/global", methods=["GET"])
def get_global():
    """
    Worker 拉取当前全局模型
    """
    with status_lock:
        w = global_weights.tolist()
        r = global_round
    return jsonify({"round": r, "weights": w})


@app.route("/status", methods=["GET"])
def status():
    with status_lock:
        return jsonify({
            "global_round": global_round,
            "num_pending_rounds": len(pending_updates),
            "workers": workers_status
        })


@app.route("/metrics", methods=["GET"])
def metrics():
    """
    Prometheus metrics
    """
    lines = []
    with status_lock:
        # 全局
        lines.append("# HELP fl_global_round Current global round")
        lines.append("# TYPE fl_global_round gauge")
        lines.append(f"fl_global_round {global_round}")

        # 各 worker
        lines.append("# HELP fl_worker_loss Last reported loss")
        lines.append("# TYPE fl_worker_loss gauge")
        lines.append("# HELP fl_worker_num_samples Last reported num samples")
        lines.append("# TYPE fl_worker_num_samples gauge")
        lines.append("# HELP fl_worker_last_update Timestamp of last update")
        lines.append("# TYPE fl_worker_last_update gauge")

        for wid, info in workers_status.items():
            lines.append(f'fl_worker_loss{{worker="{wid}"}} {info.get("loss", 0.0)}')
            lines.append(f'fl_worker_num_samples{{worker="{wid}"}} {info.get("num_samples", 0)}')
            lines.append(f'fl_worker_last_update{{worker="{wid}"}} {int(info.get("last_update_ts", 0))}')

    return "\n".join(lines), 200, {"Content-Type": "text/plain; charset=utf-8"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
 
