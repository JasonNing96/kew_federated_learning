"""
联邦学习参数服务器 - 基于FastAPI + FedAvg
功能：
1. 提供全局模型权重下载 (/global)
2. 接收客户端更新并聚合 (/update)
3. 支持多轮联邦训练
4. 支持FLQ量化聚合（可选）
"""
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse
import torch
import io
import threading
from ultralytics import YOLO
from datetime import datetime
import os
import sys
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from flq_modules.config import FLQConfig
from flq_modules.utils import ParamMetadata, model_to_vector, vector_to_model
from flq_modules.aggregation import QuantizedAggregator, fedavg_aggregate
from flq_modules.quantization import compute_compression_ratio

# ============= 加载配置 =============
try:
    config = FLQConfig()
    print(f"✅ 加载配置: {config}")
except Exception as e:
    print(f"⚠️  配置加载失败，使用默认配置: {e}")
    config = None

# ============= 配置参数 =============
CLIENTS_PER_ROUND = config.clients_per_round if config else 3
ROUNDS = config.rounds if config else 2
MODEL_PATH = config.model_name if config else "yolov8n.pt"
DATA_YAML = "client1/oil.yaml"  # 数据配置（用于初始化正确的模型架构）
SAVE_DIR = config.save_dir if config else "server_checkpoints"

# 量化参数
QUANT_ENABLED = config.quantization_enabled if config else False
QUANT_BITS = config.quantization_bits if config else 32
USE_ERROR_FEEDBACK = config.use_error_feedback if config else True

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

# 统计信息
stats = {
    "round_times": [],      # 每轮耗时
    "upload_bits": [],      # 上行比特数
    "download_bits": [],    # 下行比特数
    "accuracies": [],       # 每轮准确率
    "round_start_time": None
}

# 初始化量化聚合器（如果启用）
quantized_aggregator = None
param_metadata = None
if QUANT_ENABLED:
    param_metadata = ParamMetadata(model.model)
    quantized_aggregator = QuantizedAggregator(
        bits=QUANT_BITS,
        shapes=param_metadata.shapes,
        use_error_feedback=USE_ERROR_FEEDBACK
    )
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔧 量化聚合器已启用: {QUANT_BITS}-bit")

print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 服务器就绪，等待客户端连接...")
print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 配置: {CLIENTS_PER_ROUND}客户端/轮 × {ROUNDS}轮")
if QUANT_ENABLED:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🗜️  量化: {QUANT_BITS}-bit (压缩率{compute_compression_ratio(32, QUANT_BITS):.1f}x)")
else:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📦 量化: 关闭 (FP32基线)")
print("=" * 60)


def print_final_summary():
    """打印最终训练总结"""
    print(f"\n{'='*70}")
    print(f"{'🎉 联邦学习训练完成':^70}")
    print(f"{'='*70}")
    print(f"  🔢 总轮数      : {round_id}")
    print(f"  ⏱️  总耗时      : {sum(stats['round_times']):.2f} 秒")
    print(f"  📤 总上行      : {sum(stats['upload_bits'])/1e9:.2f} Gbits")
    print(f"  📥 总下行      : {sum(stats['download_bits'])/1e9:.2f} Gbits")
    print(f"  📊 总通信      : {(sum(stats['upload_bits'])+sum(stats['download_bits']))/1e9:.2f} Gbits")
    if stats['accuracies']:
        print(f"  🎯 最终mAP50   : {stats['accuracies'][-1]:.4f}")
        print(f"  📈 最佳mAP50   : {max(stats['accuracies']):.4f}")
    print(f"  📦 最终模型    : server_checkpoints/global_round_{round_id}.pt")
    print(f"{'='*70}\n")

def aggregate():
    """
    FedAvg聚合算法：加权平均所有客户端的权重
    权重 = local_weight * (n_samples / total_samples)
    """
    global global_sd, buf, buf_n, round_id, training_done, stats

    round_start = time.time()
    
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

    # === 统计通信开销 ===
    # 计算单个state_dict的比特数
    param_count = sum(v.numel() for v in global_sd.values())
    
    # 上行：客户端使用量化比特数（如果启用）
    upload_bits_per_client = param_count * (QUANT_BITS if QUANT_ENABLED else 32)
    upload_bits = upload_bits_per_client * len(buf_n)
    
    # 下行：服务器始终下发FP32模型
    download_bits_per_client = param_count * 32
    download_bits = download_bits_per_client * len(buf_n)
    
    stats["upload_bits"].append(upload_bits)
    stats["download_bits"].append(download_bits)
    
    # === 验证模型准确率（暂时跳过，训练完成后统一验证） ===
    # TODO: 实时验证有PyTorch兼容性问题，后续优化
    mAP50 = 0.0
    mAP50_95 = 0.0
    stats["accuracies"].append(0.0)
    
    # === 计算耗时 ===
    round_time = time.time() - round_start
    stats["round_times"].append(round_time)
    
    # === 打印统计信息 ===
    print(f"\n{'='*70}")
    print(f"{'📊 Round ' + str(round_id) + ' 统计信息':^70}")
    print(f"{'='*70}")
    print(f"  ⏱️  Tround    : {round_time:.2f} 秒")
    print(f"  📤 BitUp     : {upload_bits/1e9:.3f} Gbits ({upload_bits/1e6:.1f} MB)")
    print(f"  📥 BitDown   : {download_bits/1e9:.3f} Gbits ({download_bits/1e6:.1f} MB)")
    print(f"  📊 BitTotal  : {(upload_bits+download_bits)/1e9:.3f} Gbits")
    # print(f"  🎯 mAP50     : {mAP50:.4f}")  # 实时验证已禁用
    print(f"  👥 Clients   : {len(buf_n)}/{CLIENTS_PER_ROUND}")
    print(f"  📦 Params    : {param_count:,} ({param_count*4/1e6:.1f} MB)")
    
    # 压缩率（相对FP32）
    if QUANT_ENABLED:
        compression_ratio = compute_compression_ratio(32, QUANT_BITS)
        print(f"  🗜️  Compress  : {compression_ratio:.2f}x ({QUANT_BITS}-bit quantization)")
    else:
        print(f"  🗜️  Compress  : 1.00x (FP32 baseline)")
    print(f"{'='*70}\n")

    # 清空缓存
    buf.clear()
    buf_n.clear()

    # 检查是否完成所有轮次
    if round_id >= ROUNDS:
        training_done = True
        print_final_summary()
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 第 {round_id} 轮完成，等待下一轮...")
        print("=" * 70)


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
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🌐 启动服务器: http://0.0.0.0:8087")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📡 支持内网多设备访问，使用内网IP:8087连接")
    uvicorn.run(app, host="0.0.0.0", port=8087)
