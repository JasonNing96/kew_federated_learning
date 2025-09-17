#!/usr/bin/env python3
"""
简化版FLQ联邦学习测试脚本
基于master.py和worker.py，整合为单文件运行
使用真实MNIST数据集测试FLQ量化算法效果
"""

import numpy as np
import time
import random
import json
import os
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---------------------
# 全局配置参数
# ---------------------
NUM_WORKERS = 10       # 联邦学习客户端数量（减少加速测试）
NUM_ROUNDS = 200        # 联邦学习轮次（减少加速测试）
LOCAL_EPOCHS = 5      # 每个客户端的本地训练轮数（减少加速测试）
BATCH_SIZE = 64       # 本地训练批次大小
LR = 0.01            # 学习率
SEED = 42            # 随机种子

# FLQ量化配置
FLQ_MODE = "sign1"    # 量化模式: "off", "sign1", "int8", "4bit"

# MNIST数据集配置
TRAIN_SIZE = 60000    # MNIST训练集大小
TEST_SIZE = 10000     # MNIST测试集大小

print(f"📊 FLQ联邦学习测试配置:")
print(f"  客户端数: {NUM_WORKERS}")
print(f"  训练轮次: {NUM_ROUNDS}")
print(f"  量化模式: {FLQ_MODE}")
print(f"  本地轮数: {LOCAL_EPOCHS}")
print(f"  批次大小: {BATCH_SIZE}")

# 设置随机种子
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  使用设备: {device}")

# ---------------------
# PyTorch MNIST模型定义
# ---------------------
class MNISTNet(nn.Module):
    """简单的MNIST分类网络"""
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def get_model_parameters_count(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters())

def model_to_vector(model):
    """将模型参数转换为一维向量"""
    vec = []
    for param in model.parameters():
        vec.append(param.data.cpu().numpy().flatten())
    return np.concatenate(vec)

def vector_to_model(vector, model):
    """将一维向量转换回模型参数"""
    start_idx = 0
    for param in model.parameters():
        param_length = param.numel()
        param_data = vector[start_idx:start_idx + param_length]
        param.data = torch.from_numpy(param_data.reshape(param.shape)).float().to(param.device)
        start_idx += param_length

# ---------------------
# MNIST数据集加载和分割
# ---------------------
def load_mnist_data():
    """加载MNIST数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset

def split_dataset_for_workers_iid(dataset, num_workers):
    """
    为多个worker分割数据集（完全IID数据分布）
    每个worker都包含所有类别，且类别分布相同
    """
    num_items = len(dataset)
    
    # 获取标签
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    else:
        labels = [dataset[i][1] for i in range(len(dataset))]
    
    labels = np.array(labels)
    num_classes = len(np.unique(labels))
    
    # 按类别组织数据索引
    class_indices = {}
    for cls in range(num_classes):
        class_indices[cls] = np.where(labels == cls)[0]
    
    worker_datasets = []
    
    print("📊 使用IID（独立同分布）数据分割方式")
    
    for worker_id in range(num_workers):
        worker_indices = []
        
        # 每个worker从每个类别中获得相同比例的数据
        for cls in range(num_classes):
            cls_data = class_indices[cls]
            # 将该类别的数据平均分配给所有workers
            start_idx = worker_id * len(cls_data) // num_workers
            end_idx = (worker_id + 1) * len(cls_data) // num_workers
            worker_indices.extend(cls_data[start_idx:end_idx])
        
        # 随机打乱索引，保证数据的随机性
        np.random.shuffle(worker_indices)
        worker_datasets.append(Subset(dataset, worker_indices))
        
        # 统计每个worker的类别分布
        worker_labels = [labels[i] for i in worker_indices]
        unique, counts = np.unique(worker_labels, return_counts=True)
        print(f"Worker {worker_id} IID分布 ({len(worker_indices)}样本): {dict(zip(unique, counts))}")
    
    return worker_datasets

def split_dataset_for_workers(dataset, num_workers, alpha=0.5):
    """
    为多个worker分割数据集（模拟Non-IID数据分布）
    alpha: 控制数据分布的不均匀程度，0=完全Non-IID，1=完全IID
    """
    num_items = len(dataset)
    items_per_worker = num_items // num_workers
    
    # 获取标签
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    else:
        labels = [dataset[i][1] for i in range(len(dataset))]
    
    labels = np.array(labels)
    num_classes = len(np.unique(labels))
    
    worker_datasets = []
    
    for worker_id in range(num_workers):
        if alpha == 1.0:
            # 完全IID：随机分配
            indices = np.random.choice(num_items, items_per_worker, replace=False)
        else:
            # Non-IID：每个worker主要包含特定类别的数据
            primary_classes = np.random.choice(num_classes, max(1, int(alpha * num_classes)), replace=False)
            
            # 获取主要类别的索引
            primary_indices = []
            for cls in primary_classes:
                cls_indices = np.where(labels == cls)[0]
                primary_indices.extend(cls_indices[:items_per_worker // len(primary_classes)])
            
            # 随机添加其他类别的少量数据
            remaining_count = items_per_worker - len(primary_indices)
            if remaining_count > 0:
                other_indices = np.setdiff1d(range(num_items), primary_indices)
                additional_indices = np.random.choice(other_indices, remaining_count, replace=False)
                primary_indices.extend(additional_indices)
            
            indices = primary_indices[:items_per_worker]
        
        worker_dataset = Subset(dataset, indices)
        worker_datasets.append(worker_dataset)
        
        # 统计每个worker的类别分布
        worker_labels = [labels[i] for i in indices]
        unique, counts = np.unique(worker_labels, return_counts=True)
        print(f"Worker {worker_id} 数据分布: {dict(zip(unique, counts))}")
    
    return worker_datasets

# ---------------------
# FLQ量化算法（从flq_quantization.py简化）
# ---------------------
def flq_sign_quantization(gradients):
    """FLQ符号量化（1位）"""
    signs = np.sign(gradients)
    scale_factor = np.mean(np.abs(gradients))
    quantized = signs * scale_factor
    logical_bits = len(gradients)  # 每参数1位
    return quantized, scale_factor, logical_bits

def flq_8bit_quantization(gradients):
    """FLQ 8位量化"""
    max_abs = np.max(np.abs(gradients))
    if max_abs == 0:
        return gradients, 1.0, 8 * len(gradients)
    
    scale_factor = max_abs / 127.0
    quantized_int8 = np.clip(np.round(gradients / scale_factor), -128, 127)
    quantized = quantized_int8 * scale_factor
    logical_bits = 8 * len(gradients)
    return quantized, scale_factor, logical_bits

def flq_4bit_quantization(gradients):
    """FLQ 4位量化"""
    max_abs = np.max(np.abs(gradients))
    if max_abs == 0:
        return gradients, 1.0, 4 * len(gradients)
    
    scale_factor = max_abs / 7.0
    quantized_int4 = np.clip(np.round(gradients / scale_factor), -8, 7)
    quantized = quantized_int4 * scale_factor
    logical_bits = 4 * len(gradients)
    return quantized, scale_factor, logical_bits

def apply_flq_quantization(gradients, mode):
    """应用FLQ量化"""
    if mode == "off":
        return gradients, 32 * len(gradients)
    elif mode == "sign1":
        quantized, scale, logical_bits = flq_sign_quantization(gradients)
        return quantized, logical_bits
    elif mode == "int8":
        quantized, scale, logical_bits = flq_8bit_quantization(gradients)
        return quantized, logical_bits
    elif mode == "4bit":
        quantized, scale, logical_bits = flq_4bit_quantization(gradients)
        return quantized, logical_bits
    else:
        raise ValueError(f"Unknown FLQ mode: {mode}")

# ---------------------
# 参数服务器类（基于master.py）
# ---------------------
class FederatedMaster:
    """联邦学习参数服务器"""
    
    def __init__(self, min_clients: int = 10):
        # 创建全局模型来获取参数维度
        global_model = MNISTNet()
        self.model_dim = get_model_parameters_count(global_model)
        self.global_weights = model_to_vector(global_model)
        
        self.min_clients = min_clients
        self.global_round = 0
        self.pending_updates = {}
        self.workers_status = {}
        
        print(f"🏛️ 参数服务器初始化完成")
        print(f"  模型参数量: {self.model_dim}")
        print(f"  最小客户端数: {min_clients}")
    
    def get_global_model(self) -> Tuple[np.ndarray, int]:
        """获取全局模型"""
        return self.global_weights.copy(), self.global_round
    
    def receive_update(self, worker_id: str, round_id: int, delta: np.ndarray, 
                      num_samples: int, loss: float) -> Dict[str, Any]:
        """接收客户端更新"""
        # 验证维度
        if delta.shape[0] != self.model_dim:
            return {"status": "error", "msg": f"Dimension mismatch: {delta.shape[0]} vs {self.model_dim}"}
        
        # 记录worker状态
        self.workers_status[worker_id] = {
            "round": round_id,
            "num_samples": num_samples,
            "loss": loss,
            "timestamp": time.time()
        }
        
        # 累积更新
        if round_id not in self.pending_updates:
            self.pending_updates[round_id] = []
        
        self.pending_updates[round_id].append({
            "worker_id": worker_id,
            "delta": delta,
            "num_samples": num_samples,
            "loss": loss
        })
        
        # 检查是否可以聚合
        aggregated = False
        if (round_id == self.global_round and 
            len(self.pending_updates[round_id]) >= self.min_clients):
            aggregated = self._aggregate_updates(round_id)
        
        return {
            "status": "ok",
            "aggregated": aggregated,
            "global_round": self.global_round
        }
    
    def _aggregate_updates(self, round_id: int) -> bool:
        """FedAvg聚合算法"""
        updates = self.pending_updates[round_id]
        
        # 计算总样本数
        total_samples = sum(u["num_samples"] for u in updates)
        
        if total_samples > 0:
            # 加权平均聚合
            agg_delta = np.zeros_like(self.global_weights)
            for update in updates:
                weight = update["num_samples"] / total_samples
                agg_delta += weight * update["delta"]
        else:
            # 简单平均
            agg_delta = np.mean([u["delta"] for u in updates], axis=0)
        
        # 更新全局模型
        self.global_weights = self.global_weights + agg_delta
        self.global_round += 1
        
        # 清理已聚合的更新
        del self.pending_updates[round_id]
        
        print(f"🔄 第{round_id}轮聚合完成，参与客户端: {len(updates)}")
        avg_loss = np.mean([u["loss"] for u in updates])
        print(f"  平均损失: {avg_loss:.4f}")
        
        return True

# ---------------------
# 联邦学习客户端类（基于worker.py）
# ---------------------
class FederatedWorker:
    """联邦学习客户端"""
    
    def __init__(self, worker_id: str, dataset, test_dataset):
        self.worker_id = worker_id
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.dataset_size = len(dataset)
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            drop_last=True
        )
        
        # 创建本地模型
        self.model = MNISTNet().to(device)
        self.criterion = nn.NLLLoss()
        
        print(f"👤 客户端 {worker_id} 初始化完成")
        print(f"  本地数据量: {self.dataset_size}")
        print(f"  模型参数量: {get_model_parameters_count(self.model)}")
    
    def local_training(self, global_weights: np.ndarray, epochs: int, lr: float) -> Tuple[np.ndarray, float, float]:
        """真实的本地MNIST训练"""
        # 1. 设置全局权重到本地模型
        vector_to_model(global_weights, self.model)
        initial_weights = global_weights.copy()
        
        # 2. 创建优化器
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        
        # 3. 本地训练
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        correct = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # 统计
                epoch_loss += loss.item()
                epoch_samples += len(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            
            total_loss += epoch_loss
            total_samples += epoch_samples
            
            if epoch == 0:  # 只在第一个epoch打印详细信息
                print(f"  [{self.worker_id}] Epoch {epoch+1}/{epochs}: "
                      f"Loss={epoch_loss/len(self.train_loader):.4f}")
        
        # 4. 计算最终指标
        avg_loss = total_loss / (epochs * len(self.train_loader))
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        
        # 5. 计算权重更新
        final_weights = model_to_vector(self.model)
        delta = final_weights - initial_weights
        
        return delta, avg_loss, accuracy
    
    def evaluate_model(self, global_weights: np.ndarray = None) -> Tuple[float, float]:
        """评估模型在测试集上的性能"""
        if global_weights is not None:
            vector_to_model(global_weights, self.model)
        
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        # 创建测试数据加载器（只使用前1000个样本加速测试）
        test_subset = Subset(self.test_dataset, range(min(1000, len(self.test_dataset))))
        test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(data)
        
        avg_loss = test_loss / len(test_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def participate_round(self, master: FederatedMaster, flq_mode: str) -> Dict[str, Any]:
        """参与一轮联邦学习"""
        # 1. 从服务器获取全局模型
        global_weights, current_round = master.get_global_model()
        
        # 2. 本地训练
        delta, train_loss, train_accuracy = self.local_training(global_weights, LOCAL_EPOCHS, LR)
        
        # 3. FLQ量化
        quantized_delta, logical_bits = apply_flq_quantization(delta, flq_mode)
        
        # 4. 计算通信开销
        original_bits = 32 * len(delta)  # 原始32位浮点数
        compression_ratio = original_bits / logical_bits if logical_bits > 0 else 1.0
        
        # 5. 使用实际参与的样本数量
        num_samples = self.dataset_size
        
        # 6. 上报给服务器
        resp = master.receive_update(
            worker_id=self.worker_id,
            round_id=current_round,
            delta=quantized_delta,
            num_samples=num_samples,
            loss=train_loss
        )
        
        return {
            "round": current_round,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "num_samples": num_samples,
            "logical_bits": logical_bits,
            "compression_ratio": compression_ratio,
            "response": resp
        }

# ---------------------
# 实验结果记录和分析
# ---------------------
class ExperimentLogger:
    """实验日志记录器 - 记录训练结果到JSON文件用于绘图分析"""
    
    def __init__(self, experiment_name: str = "flq_experiment"):
        self.experiment_name = experiment_name
        self.results = []
        self.communication_costs = []
        self.accuracies = []
        
        # 根据论文图片设计的数据结构
        self.experiment_data = {
            "experiment_info": {
                "name": experiment_name,
                "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
                "config": {
                    "num_workers": NUM_WORKERS,
                    "num_rounds": NUM_ROUNDS,
                    "local_epochs": LOCAL_EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LR,
                    "flq_mode": FLQ_MODE
                }
            },
            "convergence_data": {
                "rounds": [],
                "global_test_loss": [],
                "global_test_accuracy": [],
                "average_train_loss": [],
                "average_train_accuracy": [],
                "communication_bits": [],
                "compression_ratios": []
            },
            "communication_data": {
                "total_upload_bits": 0,
                "total_broadcast_bits": 0,
                "per_round_bits": [],
                "quantization_levels": []
            },
            "quantization_analysis": {
                "mode": FLQ_MODE,
                "rmse_values": [],
                "approximation_quality": [],
                "binary_values": []
            },
            "resource_optimization": {
                "final_accuracy": 0.0,
                "total_iterations": 0,
                "total_communication_bits": 0,
                "convergence_round": -1
            }
        }
    
    def log_round(self, round_id: int, worker_results: List[Dict], global_loss: float, global_accuracy: float = 0.0):
        """记录一轮实验结果"""
        # 聚合本轮统计
        total_bits = sum(r["logical_bits"] for r in worker_results)
        avg_compression = np.mean([r["compression_ratio"] for r in worker_results])
        avg_train_loss = np.mean([r["train_loss"] for r in worker_results])
        avg_train_accuracy = np.mean([r["train_accuracy"] for r in worker_results])
        
        round_result = {
            "round": round_id,
            "global_loss": global_loss,
            "global_accuracy": global_accuracy,
            "avg_train_loss": avg_train_loss,
            "avg_train_accuracy": avg_train_accuracy,
            "total_communication_bits": total_bits,
            "avg_compression_ratio": avg_compression,
            "num_workers": len(worker_results)
        }
        
        self.results.append(round_result)
        self.communication_costs.append(total_bits)
        self.accuracies.append(global_accuracy)
        
        # 更新JSON数据结构 - 用于论文图表绘制
        self._update_json_data(round_id, worker_results, global_loss, global_accuracy, 
                              total_bits, avg_compression, avg_train_loss, avg_train_accuracy)
        
        print(f"📈 第{round_id}轮结果:")
        print(f"  全局测试损失: {global_loss:.4f}")
        print(f"  全局测试精度: {global_accuracy:.3f}")
        print(f"  平均训练损失: {avg_train_loss:.4f}")
        print(f"  平均训练精度: {avg_train_accuracy:.3f}")
        print(f"  通信开销: {total_bits:,} bits")
        print(f"  平均压缩比: {avg_compression:.1f}:1")
    
    def _update_json_data(self, round_id: int, worker_results: List[Dict], global_loss: float, 
                         global_accuracy: float, total_bits: int, avg_compression: float,
                         avg_train_loss: float, avg_train_accuracy: float):
        """更新JSON数据结构"""
        # 收敛数据 (对应论文图2)
        conv_data = self.experiment_data["convergence_data"]
        conv_data["rounds"].append(round_id)
        conv_data["global_test_loss"].append(float(global_loss))
        conv_data["global_test_accuracy"].append(float(global_accuracy))
        conv_data["average_train_loss"].append(float(avg_train_loss))
        conv_data["average_train_accuracy"].append(float(avg_train_accuracy))
        conv_data["communication_bits"].append(int(total_bits))
        conv_data["compression_ratios"].append(float(avg_compression))
        
        # 通信数据 (对应论文图3和表1)
        comm_data = self.experiment_data["communication_data"]
        comm_data["per_round_bits"].append(int(total_bits))
        comm_data["total_upload_bits"] += total_bits  # 客户端上传
        comm_data["total_broadcast_bits"] += total_bits // len(worker_results)  # 服务器广播
        
        # 量化分析 (对应论文图4)
        quant_data = self.experiment_data["quantization_analysis"]
        if FLQ_MODE == "sign1":
            # 计算二进制值 (0或1)
            binary_vals = [1.0 if r["compression_ratio"] > 1.0 else 0.0 for r in worker_results]
            quant_data["binary_values"].extend(binary_vals)
        
        # 计算RMSE (近似误差)
        if len(worker_results) > 0:
            # 简化的RMSE计算：基于压缩比的近似误差
            rmse = 1.0 / avg_compression if avg_compression > 0 else 1.0
            quant_data["rmse_values"].append(float(rmse))
            quant_data["approximation_quality"].append(float(global_accuracy))
        
        # 资源优化汇总 (对应论文表1)
        resource_data = self.experiment_data["resource_optimization"]
        resource_data["final_accuracy"] = float(global_accuracy)
        resource_data["total_iterations"] = round_id + 1
        resource_data["total_communication_bits"] = comm_data["total_upload_bits"]
        
        # 检测收敛点（损失变化小于阈值）
        if (len(conv_data["global_test_loss"]) > 5 and 
            resource_data["convergence_round"] == -1):
            recent_losses = conv_data["global_test_loss"][-5:]
            if max(recent_losses) - min(recent_losses) < 0.01:  # 收敛阈值
                resource_data["convergence_round"] = round_id
    
    def plot_results(self, flq_mode: str):
        """绘制实验结果"""
        if not self.results:
            print("⚠️ 没有实验数据可绘制")
            return
        
        rounds = [r["round"] for r in self.results]
        global_losses = [r["global_loss"] for r in self.results]
        global_accuracies = [r["global_accuracy"] for r in self.results]
        comm_costs = [r["total_communication_bits"] / 1e6 for r in self.results]  # 转换为MB
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        ax1.plot(rounds, global_losses, 'b-o', linewidth=2, markersize=4)
        ax1.set_xlabel('联邦学习轮次')
        ax1.set_ylabel('全局测试损失')
        ax1.set_title(f'FLQ-{flq_mode} 损失收敛曲线')
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(rounds, global_accuracies, 'g-o', linewidth=2, markersize=4)
        ax2.set_xlabel('联邦学习轮次')
        ax2.set_ylabel('全局测试准确率')
        ax2.set_title(f'FLQ-{flq_mode} 准确率曲线')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # 通信开销
        ax3.plot(rounds, comm_costs, 'r-s', linewidth=2, markersize=4)
        ax3.set_xlabel('联邦学习轮次')
        ax3.set_ylabel('通信开销 (Mbits)')
        ax3.set_title(f'FLQ-{flq_mode} 通信开销')
        ax3.grid(True, alpha=0.3)
        
        # 准确率-通信开销效率图
        ax4.plot(comm_costs, global_accuracies, 'm-d', linewidth=2, markersize=4)
        ax4.set_xlabel('累计通信开销 (Mbits)')
        ax4.set_ylabel('全局测试准确率')
        ax4.set_title(f'FLQ-{flq_mode} 通信效率')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f'flq_{flq_mode}_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 实验结果图已保存为 flq_{flq_mode}_results.png")
    
    def save_experiment_data(self, filename: str = None):
        """保存实验数据到JSON文件"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            flq_mode = self.experiment_data["experiment_info"]["config"]["flq_mode"]
            filename = f"experiment_results_{flq_mode}_{timestamp}.json"
        
        # 确保results目录存在
        os.makedirs("results", exist_ok=True)
        filepath = os.path.join("results", filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.experiment_data, f, indent=2, ensure_ascii=False)
            print(f"📄 实验数据已保存到: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ 保存实验数据失败: {e}")
            return None
    
    def load_experiment_data(self, filepath: str):
        """从JSON文件加载实验数据"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.experiment_data = json.load(f)
            print(f"📄 实验数据已从 {filepath} 加载")
            return True
        except Exception as e:
            print(f"❌ 加载实验数据失败: {e}")
            return False
    
    def get_summary_for_comparison(self):
        """获取用于对比的实验摘要数据"""
        resource_data = self.experiment_data["resource_optimization"]
        conv_data = self.experiment_data["convergence_data"]
        comm_data = self.experiment_data["communication_data"]
        
        return {
            "mode": self.experiment_data["quantization_analysis"]["mode"],
            "final_accuracy": resource_data["final_accuracy"],
            "final_loss": conv_data["global_test_loss"][-1] if conv_data["global_test_loss"] else 0.0,
            "total_communication_bits": resource_data["total_communication_bits"],
            "convergence_round": resource_data["convergence_round"],
            "total_iterations": resource_data["total_iterations"],
            "avg_compression_ratio": np.mean(conv_data["compression_ratios"]) if conv_data["compression_ratios"] else 1.0
        }

# ---------------------
# 主实验函数
# ---------------------
def run_federated_flq_experiment(flq_mode: str = "sign1"):
    """运行FLQ联邦学习实验"""
    print(f"\n🚀 开始FLQ-{flq_mode}联邦学习实验")
    print("=" * 50)
    
    # 1. 加载和分割MNIST数据集
    print("📥 加载MNIST数据集...")
    train_dataset, test_dataset = load_mnist_data()
    worker_datasets = split_dataset_for_workers(train_dataset, NUM_WORKERS, alpha=0.5)
    
    # 2. 初始化参数服务器
    master = FederatedMaster(min_clients=10)
    
    # 3. 初始化客户端
    workers = []
    for i in range(NUM_WORKERS):
        worker = FederatedWorker(f"worker_{i}", worker_datasets[i], test_dataset)
        workers.append(worker)
    
    # 4. 初始化实验记录器
    experiment_name = f"flq_{flq_mode}_federated_learning"
    logger = ExperimentLogger(experiment_name)
    
    # 5. 开始联邦学习训练
    for round_id in range(NUM_ROUNDS):
        print(f"\n🔄 第{round_id}轮联邦学习开始...")
        
        # 所有客户端参与训练
        worker_results = []
        for worker in workers:
            result = worker.participate_round(master, flq_mode)
            worker_results.append(result)
            
            print(f"  {worker.worker_id}: train_loss={result['train_loss']:.4f}, "
                  f"train_acc={result['train_accuracy']:.3f}, "
                  f"compression={result['compression_ratio']:.1f}:1")
        
        # 评估全局模型性能
        global_weights, _ = master.get_global_model()
        
        # 使用第一个worker来评估全局模型（所有worker共享测试集）
        test_loss, test_accuracy = workers[0].evaluate_model(global_weights)
        
        print(f"🎯 全局模型评估: test_loss={test_loss:.4f}, test_acc={test_accuracy:.3f}")
        
        # 记录实验结果
        logger.log_round(round_id, worker_results, test_loss, test_accuracy)
        
        # 短暂间隔
        time.sleep(0.1)
    
    print(f"\n✅ FLQ-{flq_mode}联邦学习实验完成！")
    
    # 6. 生成实验报告
    final_loss = logger.results[-1]["global_loss"]
    final_accuracy = logger.results[-1]["global_accuracy"]
    total_comm = sum(logger.communication_costs) / 1e6  # 转换为Mbits
    avg_compression = np.mean([r["avg_compression_ratio"] for r in logger.results])
    
    print(f"\n📊 实验总结:")
    print(f"  最终全局损失: {final_loss:.4f}")
    print(f"  最终全局准确率: {final_accuracy:.3f}")
    print(f"  总通信开销: {total_comm:.2f} Mbits")
    print(f"  平均压缩比: {avg_compression:.1f}:1")
    
    # 6. 保存实验数据到JSON文件
    json_filepath = logger.save_experiment_data()
    if json_filepath:
        print(f"✅ 训练数据已保存，可用于绘制论文图表")
    
    # 7. 绘制论文图表
    try:
        from plot_experiment import PlotExperiment
        plotter = PlotExperiment()
        # 加载刚保存的实验数据
        if json_filepath:
            json_filename = os.path.basename(json_filepath)
            plotter.load_experiment_data(json_filename)
            # 绘制单个实验的图表
            plotter.plot_gradient_quantization(flq_mode)
            print("✅ 论文图表已生成")
    except Exception as e:
        print(f"⚠️ 绘图功能可选，跳过: {e}")
    
    # 8. 传统绘制结果（可选）
    # logger.plot_results(flq_mode)
    
    return logger

# ---------------------
# 对比实验函数
# ---------------------
def compare_flq_modes():
    """对比不同FLQ量化模式"""
    modes = ["off", "sign1", "int8", "4bit"]
    results = {}
    
    print(f"\n🔬 开始FLQ量化模式对比实验")
    print("=" * 60)
    
    for mode in modes:
        print(f"\n{'='*20} FLQ-{mode} {'='*20}")
        logger = run_federated_flq_experiment(mode)
        results[mode] = logger
    
    # 汇总对比结果
    print(f"\n📈 FLQ量化模式对比总结:")
    print("-" * 60)
    print(f"{'模式':<10} {'最终损失':<12} {'最终精度':<12} {'总通信(Mbits)':<15} {'平均压缩比':<12}")
    print("-" * 60)
    
    comparison_data = {
        "comparison_timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
        "modes_compared": list(modes),
        "summary": []
    }
    
    for mode, logger in results.items():
        summary = logger.get_summary_for_comparison()
        comparison_data["summary"].append(summary)
        
        print(f"{mode:<10} {summary['final_loss']:<12.4f} {summary['final_accuracy']:<12.3f} "
              f"{summary['total_communication_bits']/1e6:<15.2f} {summary['avg_compression_ratio']:<12.1f}")
    
    # 保存对比结果到JSON
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    comparison_filepath = os.path.join("results", f"flq_modes_comparison_{timestamp}.json")
    os.makedirs("results", exist_ok=True)
    
    try:
        with open(comparison_filepath, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
        print(f"\n📄 对比结果已保存到: {comparison_filepath}")
    except Exception as e:
        print(f"❌ 保存对比结果失败: {e}")
    
    # 生成完整的对比图表
    try:
        from plot_experiment import PlotExperiment
        plotter = PlotExperiment()
        plotter.load_all_experiments()
        plotter.load_comparison_data()
        plotter.plot_all_figures()
        print("✅ 完整的论文对比图表已生成")
    except Exception as e:
        print(f"⚠️ 对比图表生成可选，跳过: {e}")
    
    return results

# ---------------------
# 主程序入口
# ---------------------
if __name__ == "__main__":
    print("🎯 FLQ联邦学习简化测试脚本")
    print("基于论文《Federated Optimal Framework with Low-bitwidth Quantization》")
    print("整合master.py和worker.py功能，单文件运行")
    
    # 选择实验模式
    print(f"\n请选择实验模式:")
    print("1. 单一FLQ模式测试")
    print("2. 多种FLQ模式对比")
    
    # choice = input("请输入选择 (1 或 2): ").strip()
    choice = "2"
    
    if choice == "1":
        # 单一模式测试
        print(f"\n当前FLQ模式: {FLQ_MODE}")
        logger = run_federated_flq_experiment(FLQ_MODE)
    
    elif choice == "2":
        # 多模式对比测试
        results = compare_flq_modes()
    
    else:
        print("输入无效，使用默认模式运行...")
        logger = run_federated_flq_experiment(FLQ_MODE)
    
    print(f"\n🎉 所有实验完成！结果已保存和可视化。")
