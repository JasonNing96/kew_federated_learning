#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLQ 联邦学习 Worker 节点
基于 flq_fed_v3.py 算法，针对 Jetson Nano 优化
"""

import os
import time
import json
import socket
import argparse
import requests
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
import threading
import logging

# ======================= Jetson Nano GPU 优化 =======================
def setup_jetson_gpu():
    """配置 Jetson Nano GPU 设置"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 启用内存增长，避免预分配所有显存
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Jetson Nano 显存较小，设置显存限制
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]  # 2GB
            )
            print(f"[GPU] 已配置 Jetson GPU: {len(gpus)}个GPU，内存增长已启用")
        except RuntimeError as e:
            print(f"[GPU] 配置错误: {e}")
    else:
        print("[GPU] 未检测到GPU，将使用CPU")

# ======================= FLQ 核心算法函数 =======================
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

def gradtovec(grad_list):
    """梯度列表转为向量"""
    out = []
    for g in grad_list:
        if g is None: 
            out.append(np.zeros(0, np.float32))
            continue
        a = g.numpy() if hasattr(g, "numpy") else np.array(g)
        out.append(a.reshape(-1))
    return (np.concatenate(out, axis=0).astype(np.float32) if out else np.zeros(0, np.float32))

def laq_per_vector(g_vec: np.ndarray, k: int) -> np.ndarray:
    """LAQ 逐向量量化"""
    L = float(2**(k-1) - 1)
    s = (np.max(np.abs(g_vec)) + 1e-12) / L
    y = g_vec / s
    low = np.floor(y)
    p = y - low
    rnd = (np.random.rand(*y.shape) < p).astype(np.float32)
    q = np.clip(low + rnd, -L, L) * s
    return q.astype(np.float32)

def _split_flat(vec: np.ndarray, shapes):
    """按形状分割扁平向量"""
    out, off = [], 0
    for shp in shapes:
        n = int(np.prod(shp))
        out.append(vec[off:off+n])
        off += n
    return out

def _cat_flat(chunks):
    """连接扁平块"""
    return np.concatenate(chunks, axis=0) if len(chunks) else np.zeros(0, np.float32)

def _quant_tensor_stoch(x: np.ndarray, b: int) -> np.ndarray:
    """随机量化"""
    L = float(2**(b-1) - 1)
    amax = float(np.max(np.abs(x))) + 1e-12
    s = amax / L
    y = x / s
    low = np.floor(y)
    p = y - low
    rnd = (np.random.rand(*y.shape) < p).astype(np.float32)
    q_int = np.clip(low + rnd, -L, L)
    return (q_int * s).astype(np.float32)

def quant_rel_per_tensor(g_vec: np.ndarray, ref_vec: np.ndarray, b: int, shapes) -> np.ndarray:
    """逐张量相对量化"""
    if b <= 0:
        return ref_vec.astype(np.float32)
    
    g_chunks = _split_flat(g_vec, shapes)
    ref_chunks = _split_flat(ref_vec, shapes)
    out = []
    
    for gt, rt in zip(g_chunks, ref_chunks):
        diff = gt - rt
        out.append(rt + _quant_tensor_stoch(diff, b))
    
    return _cat_flat(out)

# ======================= 数据集加载 =======================
def load_dataset(dataset: str, worker_id: int, total_workers: int, batch_size: int, 
                alpha: float = 0.1, seed: int = 42):
    """为 worker 加载分片数据集"""
    dataset = dataset.lower()
    if dataset in ["mnist", "mn"]:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset in ["fmnist", "fashion", "fashion_mnist"]:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    else:
        raise ValueError(f"未知数据集: {dataset}")
    
    # 数据预处理
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test = (x_test.astype("float32") / 255.0)[..., None]
    y_train = y_train.astype("int64")
    y_test = y_test.astype("int64")
    
    # 为当前 worker 分片数据 (简单均匀分片)
    n_samples = len(x_train) // total_workers
    start_idx = worker_id * n_samples
    end_idx = (worker_id + 1) * n_samples if worker_id < total_workers - 1 else len(x_train)
    
    x_local = x_train[start_idx:end_idx]
    y_local = y_train[start_idx:end_idx]
    
    # 创建数据集
    train_ds = (tf.data.Dataset.from_tensor_slices((x_local, y_local))
                .shuffle(10000, seed=seed, reshuffle_each_iteration=True)
                .batch(batch_size, drop_remainder=True)
                .repeat()
                .prefetch(tf.data.AUTOTUNE))
    
    test_ds = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
               .batch(512)
               .prefetch(tf.data.AUTOTUNE))
    
    return train_ds, test_ds, len(x_local)

# ======================= 模型构建 =======================
def build_model(l2: float = 5e-4, lr: float = 1e-3):
    """构建 CNN 模型"""
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

# ======================= FLQ Worker 类 =======================
class FLQWorker:
    def __init__(self, config: Dict):
        self.config = config
        self.worker_id = config['worker_id']
        self.master_url = config['master_url']
        
        # 网络和训练参数
        self.local_epochs = config.get('local_epochs', 1)
        self.batch_size = config.get('batch_size', 32)
        self.dataset = config.get('dataset', 'fmnist')
        self.total_workers = config.get('total_workers', 4)
        
        # FLQ 参数
        self.mode = config.get('mode', 'laq8')  # fedavg, bbit, bin, laq8
        self.b_up = config.get('b_up', 8)       # 上行量化位数
        self.b_down = config.get('b_down', 8)   # 下行量化位数
        self.clip_global = config.get('clip_global', 0.0)
        
        # 懒惰参数
        self.D = config.get('D', 10)
        self.ck = config.get('ck', 0.8)
        self.C = config.get('C', 50)
        self.warmup = config.get('warmup', 10)
        self.thr_scale = config.get('thr_scale', 1.0)
        
        # 初始化模型和数据
        setup_jetson_gpu()
        self.model, self.optimizer = build_model(
            l2=config.get('l2', 5e-4), 
            lr=config.get('lr', 1e-3)
        )
        
        self.train_ds, self.test_ds, self.n_local_samples = load_dataset(
            self.dataset, self.worker_id, self.total_workers, self.batch_size
        )
        self.train_iter = iter(self.train_ds)
        
        # FLQ 状态
        self.shapes = [tuple(v.shape) for v in self.model.trainable_variables]
        self.nv = sum(int(np.prod(s)) for s in self.shapes)
        self.ref_up = np.zeros(self.nv, np.float32)
        self.ref_down = np.zeros(self.nv, np.float32)
        self.ef_res = np.zeros(self.nv, np.float32)
        
        # 懒惰相关状态
        self.theta = np.zeros(self.nv, np.float32)
        self.dtheta_hist = np.zeros((self.nv, self.D), np.float32)
        self.e = 0.0
        self.ehat = 0.0
        self.clock = 0
        
        # 历史能量权重
        self.ksi = np.zeros((self.D, self.D+1), np.float32)
        for d in range(self.D):
            self.ksi[d, 0] = 1.0
            for k in range(1, self.D+1):
                self.ksi[d, k] = 1.0 / float(d + 1)
        self.ksi *= self.ck
        
        # 通信相关
        self.current_round = 0
        self.session = requests.Session()
        self.session.timeout = 30
        
        print(f"[Worker {self.worker_id}] 初始化完成，模型参数量: {self.nv:,}, 本地样本数: {self.n_local_samples}")

    def communicate_with_master(self, endpoint: str, data: Dict = None, method: str = 'GET') -> Optional[Dict]:
        """与 master 通信"""
        url = f"{self.master_url}/{endpoint}"
        try:
            if method == 'GET':
                response = self.session.get(url)
            elif method == 'POST':
                response = self.session.post(url, json=data)
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[Worker {self.worker_id}] 通信错误 {endpoint}: {e}")
            return None

    def local_training(self, global_round: int) -> Tuple[np.ndarray, float, bool]:
        """FLQ 本地训练"""
        # 计算历史能量
        var = weights_to_vec(self.model.trainable_variables)
        if global_round > 0:
            dtheta = var - self.theta
            self.dtheta_hist = np.roll(self.dtheta_hist, 1, axis=1)
            self.dtheta_hist[:, 0] = dtheta
        self.theta = var
        
        me_k = 0.0
        col_limit = min(global_round, self.D)
        kk = min(global_round, self.D)
        for d in range(col_limit):
            col = self.dtheta_hist[:, d]
            me_k += float(self.ksi[d, kk] * (col @ col))
        
        # 本地训练
        total_loss = 0.0
        for epoch in range(self.local_epochs):
            x, y = next(self.train_iter)
            with tf.GradientTape() as tape:
                logits = self.model(x, training=True)
                ce = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True))
                l2 = sum([self.config.get('l2', 5e-4) * tf.nn.l2_loss(v) for v in self.model.trainable_variables])
                loss = ce + l2
            
            grads = tape.gradient(loss, self.model.trainable_variables)
            g = gradtovec(grads)
            total_loss += float(loss.numpy())
            
            # FLQ 量化逻辑
            if self.mode in ["bbit", "bin"]:
                g_eff = g + self.ef_res
                q = quant_rel_per_tensor(g_eff, self.ref_up, self.b_up, self.shapes)
                self.e = float(np.dot(q - g_eff, q - g_eff))
                
                # 懒惰阈值检查
                rhs = (me_k / (self.config.get('lr', 1e-3) ** 2 * self.total_workers ** 2)) + 3.0 * (self.e + self.ehat)
                pass_thr = (float(np.dot(q, q)) >= self.thr_scale * rhs) or (global_round < self.warmup) or (self.clock >= self.C)
                
                if pass_thr:
                    self.ref_up = q
                    self.ef_res = g_eff - q
                    self.ehat = self.e
                    self.clock = 0
                    return q, total_loss / self.local_epochs, True
                else:
                    self.ef_res = g_eff
                    self.clock = min(self.clock + 1, self.C + 1)
                    return np.zeros_like(q), total_loss / self.local_epochs, False
                    
            elif self.mode == "laq8":
                q = laq_per_vector(g, 8)
                return q, total_loss / self.local_epochs, True
            else:  # fedavg
                return g, total_loss / self.local_epochs, True

    def run(self):
        """运行 worker 主循环"""
        print(f"[Worker {self.worker_id}] 开始运行...")
        
        while True:
            try:
                # 1. 获取全局模型
                global_info = self.communicate_with_master('get_global_model')
                if not global_info:
                    time.sleep(5)
                    continue
                
                global_round = global_info['round']
                global_weights = np.array(global_info['weights'], dtype=np.float32)
                
                # 2. 下行量化（如果启用）
                if self.b_down > 0 and self.b_down < 32:
                    global_weights = quant_rel_per_tensor(global_weights, self.ref_down, self.b_down, self.shapes)
                    self.ref_down = global_weights
                
                # 3. 更新本地模型
                vec_to_weights(global_weights, self.model.trainable_variables)
                
                # 4. 本地训练
                local_update, avg_loss, should_upload = self.local_training(global_round)
                
                # 5. 上传更新（如果通过懒惰阈值）
                if should_upload:
                    upload_data = {
                        'worker_id': self.worker_id,
                        'round': global_round,
                        'update': local_update.tolist(),
                        'num_samples': self.n_local_samples,
                        'loss': float(avg_loss),
                        'mode': self.mode,
                        'bits_up': self.b_up * self.nv if self.mode != 'fedavg' else 32 * self.nv
                    }
                    
                    result = self.communicate_with_master('upload_update', upload_data, 'POST')
                    status = "上传成功" if result else "上传失败"
                    print(f"[Worker {self.worker_id}] Round {global_round}: Loss={avg_loss:.4f}, {status}")
                else:
                    print(f"[Worker {self.worker_id}] Round {global_round}: 未通过懒惰阈值，跳过上传")
                
                self.current_round = global_round
                time.sleep(1)  # 短暂等待避免过度频繁请求
                
            except KeyboardInterrupt:
                print(f"[Worker {self.worker_id}] 收到中断信号，退出...")
                break
            except Exception as e:
                print(f"[Worker {self.worker_id}] 运行错误: {e}")
                time.sleep(5)

def main():
    parser = argparse.ArgumentParser(description='FLQ Worker')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--worker_id', type=int, required=True, help='Worker ID')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    config['worker_id'] = args.worker_id
    
    # 创建并运行 worker
    worker = FLQWorker(config)
    worker.run()

if __name__ == "__main__":
    main()
