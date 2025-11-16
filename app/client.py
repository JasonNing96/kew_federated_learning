"""
FLQ-Fed 联邦学习客户端
简化版 - 线性流程，易于调试
"""
import os
import time
import torch
import yaml
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional
from ultralytics import YOLO

from .config import Config


# ==================== 全局配置 ====================

PROJECT_ROOT = Path(__file__).parent.parent
CLIENT_ID = None
SERVER_URL = None
DATA_YAML = None
OUTPUT_DIR = None


# ==================== 工具函数 ====================

def _ts():
    """时间戳"""
    return datetime.now().strftime('%H:%M:%S')


def _log(msg: str):
    """日志输出"""
    print(f"[{_ts()}] {msg}")


# ==================== 核心流程 ====================

def pull_global_model(model: YOLO) -> tuple:
    """
    从服务器拉取全局模型
    
    Returns:
        current_round: 当前轮次
        is_done: 是否完成训练
    """
    try:
        response = requests.get(f"{SERVER_URL}/global", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # 反序列化并加载模型
        state_dict = {k: torch.tensor(v) for k, v in data['state_dict'].items()}
        model.model.load_state_dict(state_dict, strict=False)
        
        current_round = data['round']
        is_done = data['done']
        
        _log(f"📥 拉取全局模型成功 (Round {current_round})")
        return current_round, is_done
        
    except Exception as e:
        _log(f"❌ 拉取模型失败: {e}")
        raise


def train_local(model: YOLO, round_id: int, config: Config):
    """
    本地训练
    
    Args:
        model: YOLO模型
        round_id: 当前轮次
        config: 配置对象
    """
    _log(f"🎯 开始本地训练 Round {round_id}...")
    
    try:
        results = model.train(
            data=DATA_YAML,
            epochs=config.local_epochs,
            batch=config.batch_size,
            imgsz=640,
            device=config.device,
            workers=config.workers,
            project=OUTPUT_DIR,
            name=f"round_{round_id}",
            exist_ok=True,
            verbose=config.verbose,
            val=config.enable_val,
            plots=config.enable_plots
        )
        
        # 提取关键指标
        metrics = results.results_dict if hasattr(results, 'results_dict') else {}
        map50 = metrics.get('metrics/mAP50(B)', 0.0)
        
        _log(f"✅ 本地训练完成 (mAP50: {map50:.3f})")
        return results
        
    except Exception as e:
        _log(f"❌ 训练失败: {e}")
        raise


def push_update(model: YOLO, n_samples: int, round_id: int):
    """
    上传本地更新到服务器
    
    Args:
        model: 训练后的模型
        n_samples: 本地样本数
        round_id: 当前轮次
    """
    try:
        # 序列化 state_dict
        state_dict = model.model.state_dict()
        serialized = {k: v.cpu().tolist() for k, v in state_dict.items()}
        
        # 发送更新
        payload = {
            "client_id": CLIENT_ID,
            "state_dict": serialized,
            "n_samples": n_samples,
            "round_id": round_id
        }
        
        _log(f"📤 上传本地更新...")
        response = requests.post(f"{SERVER_URL}/update", json=payload, timeout=60)
        if not response.ok:
            _log(f"⚠️  上传失败详情: {response.text}")
        response.raise_for_status()
        
        result = response.json()
        _log(f"✅ 上传成功 (Round {result['round']}, 缓冲={result['buffered']})")
        
        return result
        
    except Exception as e:
        _log(f"❌ 上传失败: {e}")
        raise


def count_samples(data_yaml_path: str) -> int:
    """统计本地训练样本数"""
    with open(data_yaml_path) as f:
        cfg = yaml.safe_load(f)
    
    # 统计训练集图片数
    train_path = Path(cfg['path']) / cfg['train']
    if train_path.exists():
        return len(list(train_path.glob('*.jpg'))) + len(list(train_path.glob('*.png')))
    return 0


# ==================== 主流程 ====================

def start_client(client_id: int, server_url: str = None, config_path: Optional[str] = None):
    """
    启动联邦学习客户端
    
    Args:
        client_id: 客户端ID (1, 2, 3, ...)
        server_url: 服务器地址（可选）
        config_path: 配置文件路径（可选）
    """
    global CLIENT_ID, SERVER_URL, DATA_YAML, OUTPUT_DIR
    
    print("="*70)
    print(f"🚀 FLQ客户端 #{client_id}")
    print("="*70)
    
    # 加载配置
    config = Config(config_path)
    
    # 设置全局变量
    CLIENT_ID = client_id
    SERVER_URL = server_url or f"http://{config.server_host}:{config.server_port}"
    DATA_YAML = str(PROJECT_ROOT / "data" / f"client{client_id}" / "oil.yaml")
    OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / f"client{client_id}" / "runs")
    
    _log(f"🌐 服务器: {SERVER_URL}")
    _log(f"📁 数据: {DATA_YAML}")
    _log(f"📂 输出: {OUTPUT_DIR}")
    _log(f"🖥️  设备: {config.device}")
    
    # 检查数据文件
    if not os.path.exists(DATA_YAML):
        _log(f"❌ 数据配置不存在: {DATA_YAML}")
        _log(f"💡 提示: 请先运行 python scripts/split_dataset.py")
        return
    
    # 统计样本数
    n_samples = count_samples(DATA_YAML)
    _log(f"📊 本地样本数: {n_samples}\n")
    
    # 初始化模型
    _log("📦 初始化模型...")
    model_path = PROJECT_ROOT / config.model_name
    model = YOLO(str(model_path))
    
    # 读取类别数
    with open(DATA_YAML) as f:
        data_cfg = yaml.safe_load(f)
    nc = data_cfg.get('nc', 80)
    
    from ultralytics.nn.tasks import DetectionModel
    model.model = DetectionModel(model.model.yaml, ch=3, nc=nc)
    _log(f"✅ 模型初始化完成 (nc={nc})\n")
    
    # 主循环
    last_round = -1
    
    while True:
        try:
            # 1. 拉取全局模型
            current_round, is_done = pull_global_model(model)
            
            # 检查是否完成
            if is_done:
                print("\n" + "="*70)
                _log("🎉 所有联邦训练轮次已完成！")
                _log(f"📁 训练结果: {OUTPUT_DIR}")
                print("="*70)
                break
            
            # 检查是否已训练过当前轮次
            if current_round == last_round:
                _log("⏳ 等待服务器聚合...")
                time.sleep(5)
                continue
            
            # 2. 本地训练
            train_local(model, current_round, config)
            
            # 3. 上传更新
            response = push_update(model, n_samples, current_round)
            
            # 更新轮次记录
            last_round = current_round
            
            # 检查服务器返回的完成标志
            if response.get("done", False):
                print("\n" + "="*70)
                _log("🎉 联邦训练完成（服务器通知）")
                print("="*70)
                break
            
            _log(f"✅ Round {current_round} 完成\n")
        
        except KeyboardInterrupt:
            _log("⚠️  用户中断")
            break
        
        except Exception as e:
            _log(f"❌ 错误: {e}")
            _log("🔄 5秒后重试...")
            time.sleep(5)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python -m app.client <client_id>")
        sys.exit(1)
    
    start_client(int(sys.argv[1]))

