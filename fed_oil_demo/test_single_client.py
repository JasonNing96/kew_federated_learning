#!/usr/bin/env python3
"""
单客户端训练测试 - 诊断联邦学习问题
用于验证客户端能否完整走完：拉取→训练→上传流程
"""
import sys
import time
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def main():
    log("🔍 开始单客户端诊断测试...")
    
    # 1. 检查服务器连接
    log("步骤1: 检查服务器连接")
    try:
        import requests
        response = requests.get("http://127.0.0.1:8080/", timeout=5)
        log(f"✅ 服务器响应: {response.json()}")
    except Exception as e:
        log(f"❌ 服务器连接失败: {e}")
        log("💡 请先启动服务器: ./start_server.sh")
        return
    
    # 2. 测试模型拉取
    log("\n步骤2: 测试拉取全局模型")
    try:
        from client import pull_global_model, YOLO, BASE_MODEL, DATA_YAML
        import yaml
        
        # 初始化模型
        model = YOLO(BASE_MODEL)
        with open(DATA_YAML, 'r') as f:
            nc = yaml.safe_load(f).get('nc', 80)
        
        from ultralytics.nn.tasks import DetectionModel
        model.model = DetectionModel(model.model.yaml, ch=3, nc=nc)
        
        round_id, is_done = pull_global_model(model)
        log(f"✅ 拉取成功: Round {round_id}, Done={is_done}")
        log(f"   模型参数数量: {sum(p.numel() for p in model.parameters())}")
    except Exception as e:
        log(f"❌ 拉取失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 测试本地训练（快速模式）
    log("\n步骤3: 测试本地训练（快速模式）")
    log("   配置: epochs=1, batch=4, imgsz=320, device=cpu")
    
    try:
        start_time = time.time()
        
        results = model.train(
            data=DATA_YAML,
            epochs=1,
            imgsz=320,      # 减小图像尺寸
            batch=4,        # 减小批次
            device='cpu',
            project='test_runs',
            name='single_client_test',
            exist_ok=True,
            verbose=False,
            patience=999,   # 禁用早停
            plots=False     # 禁用绘图
        )
        
        elapsed = time.time() - start_time
        log(f"✅ 训练完成，耗时: {elapsed:.1f}秒")
        log(f"   结果: {results.results_dict}")
        
    except Exception as e:
        log(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 测试上传更新
    log("\n步骤4: 测试上传更新")
    try:
        from client import push_local_update
        response = push_local_update(model, n_samples=100)
        log(f"✅ 上传成功: {response}")
    except Exception as e:
        log(f"❌ 上传失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    log("\n" + "="*60)
    log("🎉 所有测试通过！客户端可以正常工作。")
    log("💡 如果多客户端仍有问题，可能是资源竞争或超时。")
    log("   建议: 增加 client.py 中的训练超时时间")
    log("="*60)

if __name__ == "__main__":
    # 设置客户端ID
    import os
    os.environ["CLIENT_ID"] = "test"
    
    main()

