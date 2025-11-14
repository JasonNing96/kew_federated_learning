#!/usr/bin/env python3
"""
本地训练测试 - 验证YOLO训练流程（不需要服务器）
目的：确认训练本身能正常完成
"""
import os
import time
import yaml
from datetime import datetime
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def test_training(use_gpu=True, fast_mode=True):
    """
    测试YOLO训练
    Args:
        use_gpu: 是否使用GPU（如果可用）
        fast_mode: 快速模式（小图像+小批次）
    """
    log("🔍 开始本地训练测试...")
    
    # 配置
    CLIENT_ID = "test"
    DATA_YAML = f"client1/oil.yaml"  # 使用client1的数据
    BASE_MODEL = "yolov8n.pt"
    
    if not os.path.exists(DATA_YAML):
        log(f"❌ 数据文件不存在: {DATA_YAML}")
        log("💡 请先运行: python split_dataset.py")
        return False
    
    # 读取数据配置
    with open(DATA_YAML, 'r') as f:
        nc = yaml.safe_load(f).get('nc', 80)
    
    log(f"📊 数据配置: {DATA_YAML} (类别数: {nc})")
    
    # 初始化模型
    log("📦 初始化模型...")
    model = YOLO(BASE_MODEL)
    model.model = DetectionModel(model.model.yaml, ch=3, nc=nc)
    log(f"✅ 模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练配置
    if fast_mode:
        epochs = 1
        imgsz = 320      # 小图像
        batch = 4        # 小批次
        log("⚡ 快速模式: 320px, batch=4")
    else:
        epochs = 1
        imgsz = 640
        batch = 8
        log("🐢 标准模式: 640px, batch=8")
    
    # 设备选择
    import torch
    if use_gpu and torch.cuda.is_available():
        device = 'cuda:0'
        log(f"🎮 使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        log("💻 使用CPU（可能较慢）")
    
    # 开始训练
    log(f"\n{'='*60}")
    log(f"🎯 开始训练: {epochs} epoch, {imgsz}px, batch={batch}, device={device}")
    log(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        results = model.train(
            data=DATA_YAML,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project='test_runs',
            name='local_training_test',
            exist_ok=True,
            verbose=True,        # 显示详细输出
            patience=999,        # 禁用早停
            plots=False,         # 禁用绘图（节省时间）
            save=False,          # 不保存模型（节省空间）
            val=True             # 启用验证
        )
        
        elapsed = time.time() - start_time
        
        log(f"\n{'='*60}")
        log(f"✅ 训练完成！耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
        log(f"{'='*60}")
        
        # 显示结果
        if hasattr(results, 'results_dict'):
            rd = results.results_dict
            log("\n📊 训练结果:")
            log(f"   mAP50: {rd.get('metrics/mAP50(B)', 0):.4f}")
            log(f"   mAP50-95: {rd.get('metrics/mAP50-95(B)', 0):.4f}")
            log(f"   box_loss: {rd.get('train/box_loss', 0):.4f}")
            log(f"   cls_loss: {rd.get('train/cls_loss', 0):.4f}")
        
        log(f"\n💡 预估完整训练时间:")
        log(f"   Fast模式(320px): ~{elapsed:.0f}秒/epoch")
        log(f"   Standard模式(640px): ~{elapsed*4:.0f}秒/epoch (~{elapsed*4/60:.1f}分钟/epoch)")
        
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        log(f"\n❌ 训练失败（耗时: {elapsed:.1f}秒）")
        log(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("\n" + "="*60)
    print("  YOLO本地训练测试")
    print("="*60 + "\n")
    
    # 测试1: GPU快速模式
    import torch
    if torch.cuda.is_available():
        log("测试1: GPU + 快速模式")
        success = test_training(use_gpu=True, fast_mode=True)
        if not success:
            return 1
    else:
        log("⚠️  GPU不可用，使用CPU模式")
        log("测试1: CPU + 快速模式")
        success = test_training(use_gpu=False, fast_mode=True)
        if not success:
            return 1
    
    log("\n" + "="*60)
    log("🎉 测试通过！训练流程正常。")
    log("="*60)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

