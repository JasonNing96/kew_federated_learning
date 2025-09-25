#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLQ 联邦学习 Master 节点
基于 flq_fed_v3.py 算法，实现 FLQ 聚合策略
"""

import os
import time
import json
import argparse
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
import threading
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import pandas as pd

# ======================= Jetson GPU 配置 =======================
def setup_jetson_gpu():
    """配置 Jetson GPU 设置"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[GPU] Master 已配置 GPU: {len(gpus)}个GPU")
        except RuntimeError as e:
            print(f"[GPU] 配置错误: {e}")

# ======================= FLQ 工具函数 =======================
def weights_to_vec(vars_list):
    """模型权重转为向量"""
    return np.concatenate([v.numpy().reshape(-1) for v in vars_list]).astype(np.float32)

def vec_to_weights(vec, vars_list):
    """向量转为模型权重"""
    i = 0
    for v in vars_list:
        n = int(np.prod(v.shape))
        v.assign(vec[i:i+n].reshape(v.shape))
        i += n

def vectograd(vec, grad_template):
    """向量转为梯度格式"""
    out, i = [], 0
    for g in grad_template:
        shape = (g.numpy() if hasattr(g, "numpy") else np.array(g)).shape
        n = int(np.prod(shape))
        part = vec[i:i+n]
        i += n
        out.append(tf.convert_to_tensor(part.reshape(shape), dtype=tf.float32))
    return out

def build_model(l2: float = 5e-4, lr: float = 1e-3):
    """构建全局模型"""
    reg = tf.keras.regularizers.L2(l2=l2)
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=reg),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=reg),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dense(10, activation=None)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    return model, optimizer

def load_test_dataset(dataset: str):
    """加载测试数据集"""
    dataset = dataset.lower()
    if dataset in ["mnist", "mn"]:
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset in ["fmnist", "fashion", "fashion_mnist"]:
        (_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    else:
        raise ValueError(f"未知数据集: {dataset}")
    
    x_test = (x_test.astype("float32") / 255.0)[..., None]
    y_test = y_test.astype("int64")
    
    test_ds = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
               .batch(512)
               .prefetch(tf.data.AUTOTUNE))
    
    return test_ds

# ======================= FLQ Master 类 =======================
class FLQMaster:
    def __init__(self, config: Dict):
        self.config = config
        setup_jetson_gpu()
        
        # 基本参数
        self.total_workers = config.get('total_workers', 4)
        self.min_workers = config.get('min_workers', 2)
        self.max_rounds = config.get('max_rounds', 1000)
        self.dataset = config.get('dataset', 'fmnist')
        
        # FLQ 参数
        self.mode = config.get('mode', 'laq8')
        self.scale_by_selected = config.get('scale_by_selected', True)
        self.sel_ref = config.get('sel_ref', 1.0)
        self.clip_global = config.get('clip_global', 0.0)
        
        # 预算选择参数
        self.sel_clients = config.get('sel_clients', 0)  # 固定选择客户端数
        self.up_budget_bits = config.get('up_budget_bits', 0.0)  # 上行预算
        
        # 初始化模型
        self.model, self.optimizer = build_model(
            l2=config.get('l2', 5e-4),
            lr=config.get('lr', 1e-3)
        )
        
        # 模型相关
        self.shapes = [tuple(v.shape) for v in self.model.trainable_variables]
        self.nv = sum(int(np.prod(s)) for s in self.shapes)
        self.grads_template = None
        
        # 测试数据
        self.test_ds = load_test_dataset(self.dataset)
        
        # 联邦学习状态
        self.current_round = 0
        self.worker_updates = {}  # 存储当前轮次的更新
        self.worker_status = {}   # 工作节点状态
        self.round_start_time = None
        
        # 统计信息
        self.history = {
            'round': [],
            'accuracy': [],
            'loss': [],
            'selected_workers': [],
            'total_bits_up': [],
            'round_time': []
        }
        
        # 线程锁
        self.lock = threading.Lock()
        
        print(f"[Master] 初始化完成，模型参数量: {self.nv:,}")
        print(f"[Master] 等待 {self.min_workers}/{self.total_workers} 个工作节点...")

    def get_global_weights(self) -> np.ndarray:
        """获取全局模型权重向量"""
        return weights_to_vec(self.model.trainable_variables)

    def evaluate_model(self) -> Tuple[float, float]:
        """评估全局模型性能"""
        test_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        test_loss = tf.keras.metrics.Mean()
        
        for x_batch, y_batch in self.test_ds:
            logits = self.model(x_batch, training=False)
            test_acc.update_state(y_batch, logits)
            
            ce_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                y_batch, logits, from_logits=True))
            test_loss.update_state(ce_loss)
        
        return float(test_acc.result().numpy()), float(test_loss.result().numpy())

    def aggregate_updates(self, updates: List[Dict]) -> bool:
        """FLQ 聚合更新"""
        if not updates:
            return False
        
        # 解析更新数据
        candidates = []
        for update in updates:
            worker_id = update['worker_id']
            local_update = np.array(update['update'], dtype=np.float32)
            num_samples = update['num_samples']
            bits_up = update.get('bits_up', 32 * self.nv)
            
            # 计算增益和成本
            gain = float(np.dot(local_update, local_update))
            cost = bits_up
            
            candidates.append({
                'worker_id': worker_id,
                'update': local_update,
                'num_samples': num_samples,
                'gain': gain,
                'cost': cost,
                'ratio': gain / (cost + 1e-12)
            })
        
        # 预算贪心选择
        candidates.sort(key=lambda x: x['ratio'], reverse=True)
        selected = []
        
        if self.sel_clients > 0:
            # 固定选择数量
            selected = candidates[:min(self.sel_clients, len(candidates))]
        elif self.up_budget_bits > 0.0:
            # 预算约束选择
            budget = self.up_budget_bits
            for cand in candidates:
                if cand['cost'] <= budget:
                    selected.append(cand)
                    budget -= cand['cost']
        else:
            # 选择所有
            selected = candidates
        
        if not selected:
            return False
        
        # 聚合选中的更新
        total_samples = sum(s['num_samples'] for s in selected)
        if total_samples > 0:
            # 按样本数加权平均
            agg_update = np.zeros(self.nv, dtype=np.float32)
            for s in selected:
                weight = s['num_samples'] / total_samples
                agg_update += weight * s['update']
        else:
            # 简单平均
            agg_update = np.mean([s['update'] for s in selected], axis=0)
        
        # 按被选端数放大步长
        if self.scale_by_selected:
            m_sel = float(len(selected))
            scale_factor = m_sel / max(1.0, self.sel_ref)
            agg_update *= scale_factor
        
        # 全局梯度裁剪
        if self.clip_global > 0:
            if self.grads_template is None:
                # 创建模板（第一次聚合时）
                self.grads_template = [tf.zeros_like(v) for v in self.model.trainable_variables]
            
            grad_tensors = vectograd(agg_update, self.grads_template)
            gnorm = np.sqrt(sum(tf.reduce_sum(gi**2) for gi in grad_tensors))
            
            if gnorm > self.clip_global:
                clip_scale = self.clip_global / (gnorm + 1e-12)
                grad_tensors = [gi * clip_scale for gi in grad_tensors]
                # 重新转换为向量
                agg_update = np.concatenate([tf.reshape(g, [-1]).numpy() for g in grad_tensors])
        else:
            # 直接应用更新
            grad_tensors = vectograd(agg_update, [tf.zeros_like(v) for v in self.model.trainable_variables])
        
        # 应用聚合的梯度
        self.optimizer.apply_gradients(zip(grad_tensors, self.model.trainable_variables))
        
        # 记录统计信息
        total_bits = sum(s['cost'] for s in selected)
        selected_workers = [s['worker_id'] for s in selected]
        
        print(f"[Master] Round {self.current_round}: 选择 {len(selected)}/{len(candidates)} 个工作节点")
        print(f"[Master] 选中工作节点: {selected_workers}")
        print(f"[Master] 总上行比特数: {total_bits:.2e}")
        
        return True

    def start_new_round(self):
        """开始新一轮"""
        with self.lock:
            if self.current_round >= self.max_rounds:
                return False
            
            self.current_round += 1
            self.worker_updates.clear()
            self.round_start_time = time.time()
            
            print(f"\n[Master] ========== Round {self.current_round} 开始 ==========")
            return True

    def can_aggregate(self) -> bool:
        """检查是否可以进行聚合"""
        with self.lock:
            return len(self.worker_updates) >= self.min_workers

    def save_results(self, filepath: str):
        """保存训练结果"""
        if not self.history['round']:
            return
        
        df = pd.DataFrame(self.history)
        df.to_excel(filepath, index=False)
        print(f"[Master] 结果已保存到: {filepath}")

# ======================= Flask API =======================
app = Flask(__name__)
master: Optional[FLQMaster] = None

@app.route('/get_global_model', methods=['GET'])
def get_global_model():
    """获取全局模型"""
    if master is None:
        return jsonify({'error': 'Master not initialized'}), 500
    
    global_weights = master.get_global_weights()
    return jsonify({
        'round': master.current_round,
        'weights': global_weights.tolist()
    })

@app.route('/upload_update', methods=['POST'])
def upload_update():
    """接收工作节点更新"""
    if master is None:
        return jsonify({'error': 'Master not initialized'}), 500
    
    data = request.get_json()
    worker_id = data['worker_id']
    round_num = data['round']
    
    # 检查轮次
    if round_num != master.current_round:
        return jsonify({'error': f'Round mismatch: expected {master.current_round}, got {round_num}'}), 400
    
    # 存储更新
    with master.lock:
        master.worker_updates[worker_id] = data
        master.worker_status[worker_id] = {
            'last_update': time.time(),
            'round': round_num,
            'loss': data.get('loss', 0.0)
        }
    
    # 检查是否可以聚合
    should_aggregate = master.can_aggregate()
    
    if should_aggregate:
        # 执行聚合
        updates = list(master.worker_updates.values())
        success = master.aggregate_updates(updates)
        
        if success:
            # 评估模型
            accuracy, loss = master.evaluate_model()
            round_time = time.time() - master.round_start_time
            
            # 记录历史
            master.history['round'].append(master.current_round)
            master.history['accuracy'].append(accuracy)
            master.history['loss'].append(loss)
            master.history['selected_workers'].append(len(updates))
            master.history['total_bits_up'].append(sum(u.get('bits_up', 0) for u in updates))
            master.history['round_time'].append(round_time)
            
            print(f"[Master] Round {master.current_round} 完成: Acc={accuracy:.4f}, Loss={loss:.4f}, Time={round_time:.2f}s")
            
            # 开始下一轮
            master.start_new_round()
    
    return jsonify({'status': 'success', 'aggregated': should_aggregate})

@app.route('/status', methods=['GET'])
def get_status():
    """获取系统状态"""
    if master is None:
        return jsonify({'error': 'Master not initialized'}), 500
    
    with master.lock:
        return jsonify({
            'current_round': master.current_round,
            'max_rounds': master.max_rounds,
            'connected_workers': len(master.worker_status),
            'pending_updates': len(master.worker_updates),
            'worker_status': master.worker_status
        })

@app.route('/stop', methods=['POST'])
def stop_training():
    """停止训练并保存结果"""
    if master is None:
        return jsonify({'error': 'Master not initialized'}), 500
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"flq_results_{master.dataset}_{master.mode}_{timestamp}.xlsx"
    master.save_results(filename)
    
    return jsonify({'status': 'Training stopped', 'results_saved': filename})

def main():
    parser = argparse.ArgumentParser(description='FLQ Master Server')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器地址')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # 创建 master
    global master
    master = FLQMaster(config)
    
    # 开始第一轮
    master.start_new_round()
    
    print(f"[Master] 服务器启动: {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)

if __name__ == "__main__":
    main()