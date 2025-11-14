"""
联邦学习参数服务器 - 基于FastAPI + FedAvg
功能：
1. 提供全局模型权重下载 (/global)
2. 接收客户端更新并聚合 (/update)
3. 支持多轮联邦训练
"""
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse
import torch
import io
import threading
from ultralytics import YOLO
from datetime import datetime
import os

# ============= 配置参数 =============
CLIENTS_PER_ROUND = 3  # 每轮参与的客户端数
ROUNDS = 2              # 快速测试用
MODEL_PATH = "yolov8n.pt"
DATA_YAML = "client1/oil.yaml"  # 数据配置（用于初始化正确的模型架构）
SAVE_DIR = "server_checkpoints"

# ============= 初始化 =============
app = FastAPI(title="Federated Learning Server")
lock = threading.Lock()

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)

# 加载初始模型（使用正确的类别数）
print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 初始化参数服务器...")
print(f"[{datetime.now().strftime('%H:%M:%S')}] 📦 加载基础模型: {MODEL_PATH}")
print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 数据配置: {DATA_YAML}")

# 读取类别数
import yaml
with open(DATA_YAML, 'r', encoding='utf-8') as f:
    data_config = yaml.safe_load(f)
nc = data_config.get('nc', 80)
print(f"[{datetime.now().strftime('%H:%M:%S')}] 🏷️  类别数: {nc}")

# 方法：直接用数据配置覆盖模型，让YOLO自动重建输出层
model = YOLO(MODEL_PATH)
# 触发模型重建（通过访问model属性并指定task和nc）
from ultralytics.nn.tasks import DetectionModel
model.model = DetectionModel(model.model.yaml, ch=3, nc=nc)

# 尝试加载预训练权重（忽略输出层不匹配）
import torch
pretrained = torch.load(MODEL_PATH, map_location='cpu')
if 'model' in pretrained:
    pretrained_sd = pretrained['model'].state_dict()
else:
    pretrained_sd = pretrained

# 加载兼容的权重（跳过输出层）
model_sd = model.model.state_dict()
compatible_sd = {}
skipped_count = 0
for k, v in pretrained_sd.items():
    if k in model_sd and model_sd[k].shape == v.shape:
        compatible_sd[k] = v
    else:
        skipped_count += 1

model.model.load_state_dict(compatible_sd, strict=False)
print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 加载预训练权重: {len(compatible_sd)}/{len(model_sd)} 层 (跳过{skipped_count}个输出层)")

global_sd = {k: v.cpu().clone() for k, v in model.model.state_dict().items()}

# 全局状态
round_id = 0
buf = []        # 缓存客户端上传的state_dict
buf_n = []      # 缓存客户端样本数
training_done = False

print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 服务器就绪，等待客户端连接...")
print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 配置: {CLIENTS_PER_ROUND}客户端/轮 × {ROUNDS}轮")
print("=" * 60)


def aggregate():
    """
    FedAvg聚合算法：加权平均所有客户端的权重
    权重 = local_weight * (n_samples / total_samples)
    """
    global global_sd, buf, buf_n, round_id, training_done

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🔄 开始第 {round_id + 1} 轮聚合...")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📥 收集到 {len(buf)} 个客户端更新")

    keys = list(global_sd.keys())
    total_samples = sum(buf_n)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 总样本数: {total_samples}")

    # 初始化聚合后的权重
    avg_sd = {k: torch.zeros_like(global_sd[k]) for k in keys}

    # 加权平均
    with torch.no_grad():
        for i, (client_sd, n_samples) in enumerate(zip(buf, buf_n)):
            weight = n_samples / total_samples
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   Client {i+1}: 样本数={n_samples}, 权重={weight:.3f}")
            for k in keys:
                client_tensor = client_sd[k].cpu()

                # 跳过整数类型参数（如索引、计数器等），不进行加权平均
                if client_tensor.dtype in [torch.int32, torch.int64, torch.long]:
                    if avg_sd[k].sum() == 0:  # 只在第一次复制
                        avg_sd[k] = client_tensor.clone()
                    continue

                # 统一浮点类型
                if avg_sd[k].dtype != client_tensor.dtype:
                    client_tensor = client_tensor.to(avg_sd[k].dtype)

                avg_sd[k] += client_tensor * weight

    # 更新全局模型
    global_sd = {k: avg_sd[k].clone().cpu() for k in keys}

    # 保存checkpoint
    round_id += 1
    checkpoint_path = os.path.join(SAVE_DIR, f"global_round_{round_id}.pt")
    torch.save(global_sd, checkpoint_path)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 💾 保存checkpoint: {checkpoint_path}")

    # 清空缓存
    buf.clear()
    buf_n.clear()

    # 检查是否完成所有轮次
    if round_id >= ROUNDS:
        training_done = True
        print(f"\n{'='*60}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎉 所有 {ROUNDS} 轮联邦训练完成！")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📦 最终模型: {checkpoint_path}")
        print(f"{'='*60}\n")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 第 {round_id} 轮聚合完成，等待下一轮...")
        print("=" * 60)


@app.get("/")
def root():
    """服务器状态查询"""
    return JSONResponse({
        "status": "running",
        "current_round": round_id,
        "total_rounds": ROUNDS,
        "clients_per_round": CLIENTS_PER_ROUND,
        "buffered_updates": len(buf),
        "training_done": training_done
    })


@app.get("/global")
def get_global():
    """
    客户端拉取全局模型权重
    Returns:
        - Body: 序列化的state_dict
        - Header X-Round: 当前轮次
        - Header X-Done: 是否训练完成
    """
    with lock:
        bio = io.BytesIO()
        torch.save(global_sd, bio)
        bio.seek(0)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📤 客户端拉取全局模型 (Round {round_id})")

        return Response(
            bio.getvalue(),
            media_type="application/octet-stream",
            headers={
                "X-Round": str(round_id),
                "X-Done": str(training_done)
            }
        )


@app.post("/update")
async def update(n: int = Form(...), file: UploadFile = File(...)):
    """
    客户端上传本地训练后的权重
    Args:
        n: 客户端训练样本数
        file: 序列化的state_dict文件
    Returns:
        - round: 当前轮次
        - done: 是否完成所有训练
        - buffered: 当前已收集的更新数
    """
    global training_done

    # 加载客户端权重
    content = await file.read()
    client_sd = torch.load(io.BytesIO(content), map_location="cpu")

    aggregated = False
    with lock:
        # 缓存客户端更新
        buf.append(client_sd)
        buf_n.append(max(1, int(n)))

        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📥 收到客户端更新: 样本数={n}, 已收集={len(buf)}/{CLIENTS_PER_ROUND}")

        # 达到聚合条件：收集足够的客户端 且 未超过总轮数
        if len(buf) >= CLIENTS_PER_ROUND and round_id < ROUNDS:
            aggregate()
            aggregated = True

    return JSONResponse({
        "round": round_id,
        "done": training_done,
        "buffered": len(buf),
        "aggregated": aggregated
    })


@app.get("/status")
def get_status():
    """详细状态查询（用于监控）"""
    return JSONResponse({
        "current_round": round_id,
        "total_rounds": ROUNDS,
        "training_done": training_done,
        "buffered_updates": len(buf),
        "clients_per_round": CLIENTS_PER_ROUND,
        "waiting_for": CLIENTS_PER_ROUND - len(buf) if not training_done else 0
    })


if __name__ == "__main__":
    import uvicorn
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🌐 启动服务器: http://0.0.0.0:8080")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📡 支持内网多设备访问，使用内网IP:8080连接")
    uvicorn.run(app, host="0.0.0.0", port=8080)
