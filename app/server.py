"""
FLQ-Fed 联邦学习服务器
简化版 - 集中所有服务器逻辑
"""
import os
import torch
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn

from .config import Config
from .model_utils import (
    state_dict_to_vector, vector_to_state_dict,
    fedavg_aggregate, compute_model_size, compute_compression_ratio
)


# ==================== 数据模型 ====================

class UpdateRequest(BaseModel):
    """客户端上传更新的请求"""
    client_id: int
    state_dict: Dict[str, Any]  # 序列化的state_dict（支持任意嵌套）
    n_samples: int
    round_id: int


class StatusResponse(BaseModel):
    """服务器状态响应"""
    current_round: int
    total_rounds: int
    training_done: bool
    buffered_updates: int
    clients_per_round: int
    waiting_for: int


# ==================== 服务器状态 ====================

class ServerState:
    """服务器全局状态"""
    
    def __init__(self, config: Config, initial_model):
        self.config = config
        self.model = initial_model
        self.global_state = initial_model.model.state_dict()
        
        # 训练状态
        self.current_round = 0
        self.training_done = False
        
        # 缓冲区
        self.update_buffer = []
        self.sample_counts = []
        
        # 统计信息
        self.round_start_time = None
        self.total_params, self.model_size_mb = compute_model_size(self.global_state, 32)
        
        print(f"[{self._ts()}] 📦 模型参数: {self.total_params:,} ({self.model_size_mb:.1f} MB)")
        print(f"[{self._ts()}] 🎯 训练目标: {config.rounds} 轮 × {config.clients_per_round} 客户端")
    
    def _ts(self):
        """时间戳"""
        return datetime.now().strftime('%H:%M:%S')
    
    def add_update(self, state_dict: Dict, n_samples: int):
        """添加客户端更新到缓冲区"""
        self.update_buffer.append(state_dict)
        self.sample_counts.append(n_samples)
        
        waiting = self.config.clients_per_round - len(self.update_buffer)
        print(f"[{self._ts()}] 📥 收到客户端更新 ({len(self.update_buffer)}/{self.config.clients_per_round})")
        
        if len(self.update_buffer) >= self.config.clients_per_round:
            self._aggregate_and_advance()
    
    def _aggregate_and_advance(self):
        """聚合更新并推进到下一轮"""
        print(f"\n{'='*70}")
        print(f"[{self._ts()}] 🔄 聚合 Round {self.current_round}")
        print(f"{'='*70}")
        
        # FedAvg 聚合
        self.global_state = fedavg_aggregate(self.update_buffer, self.sample_counts)
        
        # 保存checkpoint
        save_dir = Path(self.config.server_save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = save_dir / f"global_round_{self.current_round + 1}.pt"
        torch.save(self.global_state, checkpoint_path)
        
        # 统计信息
        round_time = (datetime.now() - self.round_start_time).total_seconds() if self.round_start_time else 0
        compress_ratio = compute_compression_ratio(32, self.config.quant_bits if self.config.quant_enabled else 32)
        
        print(f"⏱️  轮次时间: {round_time:.1f}s")
        print(f"💾 Checkpoint: {checkpoint_path}")
        print(f"🗜️  压缩率: {compress_ratio:.2f}x")
        print(f"{'='*70}\n")
        
        # 清空缓冲区并推进
        self.update_buffer.clear()
        self.sample_counts.clear()
        self.current_round += 1
        self.round_start_time = datetime.now()
        
        # 检查是否完成
        if self.current_round >= self.config.rounds:
            self.training_done = True
            print(f"🎉 所有训练轮次已完成！")
    
    def get_global_model(self) -> tuple:
        """获取全局模型"""
        return self.global_state, self.current_round, self.training_done


# ==================== FastAPI 应用 ====================

def create_app(config: Config, initial_model) -> FastAPI:
    """创建 FastAPI 应用"""
    
    app = FastAPI(title="FLQ-Fed Server")
    state = ServerState(config, initial_model)
    
    @app.get("/")
    def root():
        return {"message": "FLQ-Fed Server", "version": "2.0-simplified"}
    
    @app.get("/status", response_model=StatusResponse)
    def get_status():
        """获取服务器状态"""
        return StatusResponse(
            current_round=state.current_round,
            total_rounds=state.config.rounds,
            training_done=state.training_done,
            buffered_updates=len(state.update_buffer),
            clients_per_round=state.config.clients_per_round,
            waiting_for=state.config.clients_per_round - len(state.update_buffer)
        )
    
    @app.get("/global")
    def get_global():
        """客户端拉取全局模型"""
        global_state, round_id, done = state.get_global_model()
        
        # 序列化 state_dict
        serialized = {k: v.cpu().tolist() for k, v in global_state.items()}
        
        return {
            "state_dict": serialized,
            "round": round_id,
            "done": done
        }
    
    @app.post("/update")
    def receive_update(request: UpdateRequest):
        """接收客户端更新"""
        if state.training_done:
            return {"success": True, "message": "训练已完成", "done": True}
        
        # 反序列化 state_dict
        state_dict = {
            k: torch.tensor(v) for k, v in request.state_dict.items()
        }
        
        # 添加更新
        state.add_update(state_dict, request.n_samples)
        
        return {
            "success": True,
            "round": state.current_round,
            "done": state.training_done,
            "buffered": len(state.update_buffer)
        }
    
    return app


# ==================== 启动函数 ====================

def start_server(config_path: Optional[str] = None):
    """
    启动联邦学习服务器
    
    Args:
        config_path: 配置文件路径（可选）
    """
    print("="*70)
    print("🚀 FLQ-Fed 联邦学习服务器")
    print("="*70)
    
    # 加载配置
    config = Config(config_path)
    print(f"✅ 配置: {config}\n")
    
    # 初始化模型
    from ultralytics import YOLO
    project_root = Path(__file__).parent.parent
    model_path = project_root / config.model_name
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📦 加载模型: {model_path}")
    model = YOLO(str(model_path))
    
    # 初始化类别数（从数据配置读取）
    data_yaml = project_root / "data" / "client1" / "oil.yaml"
    if data_yaml.exists():
        import yaml
        with open(data_yaml) as f:
            data_cfg = yaml.safe_load(f)
        nc = data_cfg.get('nc', 80)
        
        from ultralytics.nn.tasks import DetectionModel
        model.model = DetectionModel(model.model.yaml, ch=3, nc=nc)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 模型初始化完成 (nc={nc})\n")
    
    # 创建 FastAPI 应用
    app = create_app(config, model)
    
    # 启动服务器
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🌐 启动服务器: http://{config.server_host}:{config.server_port}")
    print(f"{'='*70}\n")
    
    uvicorn.run(
        app,
        host=config.server_host,
        port=config.server_port,
        log_level="warning"
    )


if __name__ == "__main__":
    start_server()

