#!/usr/bin/env python3
# GPU-only stable trainer for old Xeon (no AVX2) + CUDA 11.8

import os
# ===== 环境开关：在导入 torch/cv2 前设置 =====
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"      # 避免 MKL 走高指令
os.environ["ATEN_CPU_CAPABILITY"] = "default"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from ultralytics import YOLO
import torch

# 可选：压制 OpenCV 的矢量化
try:
    import cv2
    cv2.setUseOptimized(False)
    cv2.setNumThreads(0)
    try: cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
except Exception:
    cv2 = None

def main():
    # 使用相对路径，基于脚本所在目录
    ROOT = os.path.dirname(os.path.abspath(__file__))
    data_yaml = os.path.join(ROOT, "oil-detection-2-2", "data.yaml")
    assert os.path.exists(data_yaml), f"data.yaml 不存在: {data_yaml}"

    # 基础 CUDA 检查与安全设置
    assert torch.cuda.is_available(), "CUDA 不可用"
    dev = torch.device("cuda:0")
    print("CUDA device:", torch.cuda.get_device_name(0))
    torch.backends.mkldnn.enabled = False     # 防止 CPU 回退触发 oneDNN
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False    # 禁用benchmark避免指令集问题
    torch.use_deterministic_algorithms(True, warn_only=True)

    # 预热一次张量到 GPU，提前暴露环境问题
    _ = torch.sigmoid(torch.ones(4, device=dev)).sum().item()

    model = YOLO("yolov8s.pt")

    train_args = dict(
        data=data_yaml,
        epochs=5,  # 快速测试用
        batch=16,                # 4060 Ti 可从 16 起，显存不足再降
        imgsz=640,
        project=os.path.join(ROOT, "runs"),
        name="oil_spill_gpu",
        patience=20,
        save_period=10,
        exist_ok=True,
        plots=True,
        val=True,
        device=0,                # 使用 GPU:0
        workers=0,               # 先用 0，确认稳定后再提到 2~4
        amp=True,               # 启用AMP
        verbose=True,
    )

    print("\n训练参数：")
    for k, v in train_args.items():
        print(f"  {k}: {v}")

    print("\n开始训练…")
    results = model.train(**train_args)
    print("\n✅ 训练完成")
    return results

if __name__ == "__main__":
    main()

