#!/usr/bin/env python3
"""
FLQ联邦学习实现脚本
基于legacy/FLQ.py的量化和懒惰聚合机制
支持MNIST和Fashion-MNIST数据集的真实联邦学习过程
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import json
import copy
import os
from typing import List, Tuple, Dict, Any
from sklearn.preprocessing import QuantileTransformer

# ---------------------
# 全局配置参数
# ---------------------
# 联邦学习基础配置
NUM_WORKERS = 10           # 客户端数量 M
NUM_ROUNDS = 800          # 训练轮次 Iter
BATCH_SIZE = 64           # 批次大小
LEARNING_RATE = 0.02      # 学习率 alpha
SEED = 1234               # 随机种子

# FLQ量化配置
QUANTIZATION_BITS = 4     # 量化位数 b
REGULARIZATION_COEF = 0.01  # L2正则化系数 cl

# FLQ懒惰聚合参数
LAZY_D = 10               # 历史窗口大小 D
FORCE_COMM_PERIOD = 100   # 强制通信周期 C
WEIGHT_DECAY = 0.8        # 权重衰减系数 ck
BETA_PARAM = 0.001        # Beta参数

# 数据集配置
DATASET_TYPE = "mnist"    # "mnist" 或 "fashion_mnist"

print(f"🎯 FLQ联邦学习配置:")
print(f"  客户端数: {NUM_WORKERS}")
print(f"  训练轮次: {NUM_ROUNDS}")
print(f"  量化位数: {QUANTIZATION_BITS}")
print(f"  数据集: {DATASET_TYPE}")

# 设置随机种子
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 设备配置
print(f"  使用设备: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")

# ---------------------
# 核心工具函数（来自FLQ.py）
# ---------------------
def gradtovec(grad):
    """将梯度列表转换为一维向量"""
    vec = np.array([])
    le = len(grad)
    for i in range(0, le):
        a = grad[i]
        b = a.numpy()

        if len(a.shape) == 2:
            da = int(a.shape[0])
            db = int(a.shape[1])
            b = b.reshape(da * db)
        vec = np.concatenate((vec, b), axis=0)
    return vec

def vectograd(vec, grad):
    """将一维向量转换回梯度格式"""
    le = len(grad)
    for i in range(0, le):
        a = grad[i]
        b = a.numpy()
        if len(a.shape) == 2:
            da = int(a.shape[0])
            db = int(a.shape[1])
            c = vec[0:da * db]
            c = c.reshape(da, db)
            lev = len(vec)
            vec = vec[da * db:lev]
        else:
            da = int(a.shape[0])
            c = vec[0:da]
            lev = len(vec)
            vec = vec[da:lev]
        grad[i] = 0 * grad[i] + c
    return grad

def quantd(vec, v2, b):
    """FLQ量化函数（来自原始代码）"""
    n = len(vec)
    r = max(abs(vec - v2))
    if r == 0:
        return vec.copy()
    delta = r / (np.floor(2 ** b) - 1)
    quantv = v2 - r + 2 * delta * np.floor((vec - v2 + r + delta) / (2 * delta))
    return quantv

# ---------------------
# 数据集加载和分割
# ---------------------
def load_dataset(dataset_type="mnist"):
    """加载指定数据集"""
    if dataset_type == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset_type == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")
    
    # 数据预处理
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    print(f"📥 已加载{dataset_type}数据集:")
    print(f"  训练集: {x_train.shape}, 测试集: {x_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)

def split_data_for_workers(x_train, y_train, num_workers):
    """为联邦学习客户端分割数据"""
    num_samples = len(x_train)
    samples_per_worker = num_samples // num_workers
    
    worker_data = []
    for m in range(num_workers):
        start_idx = m * samples_per_worker
        end_idx = (m + 1) * samples_per_worker
        
        worker_x = x_train[start_idx:end_idx]
        worker_y = y_train[start_idx:end_idx]
        
        # 创建TensorFlow数据集
        dataset = tf.data.Dataset.from_tensor_slices(
            (tf.cast(worker_x[..., tf.newaxis], tf.float32),
             tf.cast(worker_y, tf.int64))
        )
        dataset = dataset.batch(samples_per_worker)
        worker_data.append(dataset)
        
        # 统计每个客户端的类别分布
        unique, counts = np.unique(worker_y, return_counts=True)
        print(f"  客户端{m}: {len(worker_y)}样本, 类别分布 {dict(zip(unique, counts))}")
    
    return worker_data

# ---------------------
# 模型定义（基于FLQ.py的量化模型）
# ---------------------
def create_flq_model():
    """创建FLQ量化模型（简化版本避免序列化问题）"""
    regularizer = tf.keras.regularizers.L2(l2=0.9)
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=regularizer)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ---------------------
# FLQ联邦学习服务器
# ---------------------
class FLQFederatedServer:
    """FLQ联邦学习服务器 - 实现懒惰聚合机制"""
    
    def __init__(self, model, num_workers):
        self.global_model = model
        self.num_workers = num_workers
        
        # 计算模型参数维度
        self.param_dim = sum(np.prod(var.shape) for var in model.trainable_variables)
        
        # FLQ懒惰聚合状态
        self.worker_clocks = np.zeros(num_workers)
        self.worker_errors = np.zeros(num_workers)
        self.worker_error_hats = np.zeros(num_workers)
        self.worker_last_gradients = np.zeros((num_workers, self.param_dim))
        self.communication_indicators = np.zeros((num_workers, NUM_ROUNDS))
        
        # 历史参数变化
        self.parameter_history = np.zeros((self.param_dim, NUM_ROUNDS))
        self.ksi_weights = self._initialize_ksi_weights()
        
        # 聚合梯度
        self.aggregated_gradient = np.zeros(self.param_dim)
        
        print(f"🏛️ FLQ服务器初始化完成，模型参数维度: {self.param_dim}")
    
    def _initialize_ksi_weights(self):
        """初始化ksi权重矩阵（来自原始代码）"""
        ksi = np.ones((LAZY_D, LAZY_D + 1))
        for i in range(LAZY_D + 1):
            if i == 0:
                ksi[:, i] = np.ones(LAZY_D)
            elif i <= LAZY_D and i > 0:
                ksi[:, i] = 1 / i * np.ones(LAZY_D)
        return WEIGHT_DECAY * ksi
    
    def check_communication_condition(self, worker_id, round_id, gradient, quantization_error):
        """检查FLQ懒惰聚合通信条件"""
        # 强制通信条件
        if self.worker_clocks[worker_id] >= FORCE_COMM_PERIOD:
            return True
        
        # 计算梯度差值 dL[m] = gr[m] - mgr[m]
        gradient_diff = gradient - self.worker_last_gradients[worker_id]
        gradient_diff_norm_sq = np.sum(gradient_diff ** 2)
        
        # 计算动态阈值 me[m]
        me_threshold = self._calculate_dynamic_threshold(worker_id, round_id)
        
        # FLQ通信条件判断
        threshold = (1 / (LEARNING_RATE ** 2 * self.num_workers ** 2)) * me_threshold + \
                   3 * (quantization_error + self.worker_error_hats[worker_id])
        
        should_communicate = gradient_diff_norm_sq >= threshold
        
        return should_communicate
    
    def _calculate_dynamic_threshold(self, worker_id, round_id):
        """计算动态阈值me[m]（来自原始代码逻辑）"""
        me_value = 0.0
        
        for d in range(LAZY_D):
            if round_id - d >= 0:
                if round_id <= LAZY_D:
                    weight = self.ksi_weights[d, round_id] if round_id < self.ksi_weights.shape[1] else 0.0
                else:
                    weight = self.ksi_weights[d, LAZY_D] if LAZY_D < self.ksi_weights.shape[1] else 0.0
                
                if round_id - d < self.parameter_history.shape[1]:
                    param_change = self.parameter_history[:, round_id - d]
                    me_value += weight * np.sum(param_change ** 2)
        
        return me_value
    
    def receive_worker_update(self, worker_id, round_id, gradient, quantized_gradient, quantization_error):
        """接收客户端更新并应用懒惰聚合"""
        # 检查通信条件
        should_communicate = self.check_communication_condition(worker_id, round_id, quantized_gradient, quantization_error)
        
        # 记录通信指示器
        self.communication_indicators[worker_id, round_id] = 1 if should_communicate else 0
        
        if should_communicate:
            # 计算梯度差值并累积到聚合梯度
            gradient_diff = quantized_gradient - self.worker_last_gradients[worker_id]
            self.aggregated_gradient += gradient_diff
            
            # 更新客户端状态
            self.worker_last_gradients[worker_id] = quantized_gradient.copy()
            self.worker_error_hats[worker_id] = quantization_error
            self.worker_clocks[worker_id] = 0
            
            print(f"📡 客户端{worker_id} 参与通信")
        else:
            # 增加时钟计数
            self.worker_clocks[worker_id] += 1
            print(f"⏸️ 客户端{worker_id} 跳过通信 (clock: {self.worker_clocks[worker_id]})")
        
        return should_communicate
    
    def apply_global_update(self, round_id):
        """应用全局模型更新"""
        # 获取当前模型参数
        current_params = gradtovec(self.global_model.trainable_variables)
        
        # 应用聚合梯度更新
        updated_params = current_params + self.aggregated_gradient
        
        # 更新模型参数
        updated_grads = vectograd(self.aggregated_gradient, self.global_model.trainable_variables)
        optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
        optimizer.apply_gradients(zip(updated_grads, self.global_model.trainable_variables))
        
        # 记录参数变化历史
        if round_id < NUM_ROUNDS:
            param_change = updated_params - current_params
            self.parameter_history[:, round_id] = param_change
        
        # 重置聚合梯度
        self.aggregated_gradient = np.zeros(self.param_dim)
        
        # 统计通信客户端数量
        communicating_workers = np.sum(self.communication_indicators[:, round_id])
        print(f"🔄 第{round_id}轮更新完成，通信客户端: {int(communicating_workers)}/{self.num_workers}")

# ---------------------
# FLQ联邦学习客户端
# ---------------------
class FLQFederatedClient:
    """FLQ联邦学习客户端"""
    
    def __init__(self, client_id, dataset, global_model):
        self.client_id = client_id
        self.dataset = dataset
        self.local_model = tf.keras.models.clone_model(global_model)
        self.local_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"👤 客户端{client_id}初始化完成")
    
    def local_training(self, global_model):
        """本地训练（单轮epoch）"""
        # 同步全局模型参数
        self.local_model.set_weights(global_model.get_weights())
        initial_weights = gradtovec(self.local_model.trainable_variables)
        
        # 本地训练
        for batch_idx, (images, labels) in enumerate(self.dataset.take(1)):
            with tf.GradientTape() as tape:
                logits = self.local_model(images, training=True)
                loss_value = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))
                
                # 添加L2正则化
                l2_loss = 0
                for var in self.local_model.trainable_variables:
                    l2_loss += REGULARIZATION_COEF * tf.nn.l2_loss(var)
                total_loss = loss_value + l2_loss
            
            # 计算梯度
            grads = tape.gradient(total_loss, self.local_model.trainable_variables)
            gradient_vector = gradtovec(grads)
            
            print(f"  客户端{self.client_id}: loss={total_loss.numpy():.4f}")
            
            return gradient_vector, total_loss.numpy()
    
    def apply_quantization(self, gradient, reference_gradient):
        """应用FLQ量化（使用原始quantd函数）"""
        quantized_gradient = quantd(gradient, reference_gradient, QUANTIZATION_BITS)
        quantization_error = np.sum((quantized_gradient - gradient) ** 2)
        return quantized_gradient, quantization_error

# ---------------------
# 主实验函数
# ---------------------
def run_flq_federated_learning():
    """运行FLQ联邦学习实验"""
    print(f"\n🚀 开始FLQ联邦学习实验 - {DATASET_TYPE.upper()}")
    print("=" * 50)
    
    # 1. 加载数据集
    (x_train, y_train), (x_test, y_test) = load_dataset(DATASET_TYPE)
    worker_datasets = split_data_for_workers(x_train, y_train, NUM_WORKERS)
    
    # 准备测试数据
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_test[..., tf.newaxis], tf.float32),
         tf.cast(y_test, tf.int64))
    )
    test_dataset = test_dataset.batch(len(x_test))
    test_labels_onehot = np.eye(10)[y_test]
    
    # 2. 创建全局模型
    global_model = create_flq_model()
    
    # 3. 初始化服务器和客户端
    server = FLQFederatedServer(global_model, NUM_WORKERS)
    clients = []
    for i in range(NUM_WORKERS):
        client = FLQFederatedClient(i, worker_datasets[i], global_model)
        clients.append(client)
    
    # 4. 记录实验数据
    loss_history = []
    accuracy_history = []
    communication_costs = []
    
    # 5. 开始联邦学习训练
    start_time = time.time()
    
    for round_id in range(NUM_ROUNDS):
        print(f"\n🔄 第{round_id}轮联邦学习...")
        
        round_loss = 0.0
        round_comm_workers = 0
        
        # 所有客户端参与训练
        for client in clients:
            # 本地训练
            gradient, local_loss = client.local_training(server.global_model)
            round_loss += local_loss
            
            # 获取参考梯度进行量化
            reference_gradient = server.worker_last_gradients[client.client_id]
            quantized_gradient, quantization_error = client.apply_quantization(gradient, reference_gradient)
            
            # 上报给服务器
            should_communicate = server.receive_worker_update(
                client.client_id, round_id, gradient, quantized_gradient, quantization_error
            )
            
            if should_communicate:
                round_comm_workers += 1
        
        # 应用全局更新
        server.apply_global_update(round_id)
        
        # 评估全局模型
        avg_loss = round_loss / NUM_WORKERS
        loss_history.append(avg_loss)
        
        # 计算测试准确率（每10轮）
        if round_id % 10 == 0 or round_id == NUM_ROUNDS - 1:
            test_acc = server.global_model.evaluate(x_test, test_labels_onehot, verbose=0)
            test_accuracy = test_acc[1]
            accuracy_history.append(test_accuracy)
            
            print(f"🎯 第{round_id}轮: loss={avg_loss:.4f}, test_acc={test_accuracy:.3f}, comm={round_comm_workers}/{NUM_WORKERS}")
        
        # 记录通信开销
        total_comm_bits = round_comm_workers * QUANTIZATION_BITS * server.param_dim
        communication_costs.append(total_comm_bits)
    
    training_time = time.time() - start_time
    
    # 6. 生成实验报告
    final_loss = loss_history[-1]
    final_accuracy = accuracy_history[-1] if accuracy_history else 0.0
    total_comm_bits = sum(communication_costs)
    avg_comm_rate = np.mean([np.sum(server.communication_indicators[:, r]) / NUM_WORKERS for r in range(NUM_ROUNDS)])
    
    print(f"\n✅ FLQ联邦学习实验完成！")
    print(f"📊 实验总结:")
    print(f"  训练时间: {training_time:.2f}秒")
    print(f"  最终损失: {final_loss:.4f}")
    print(f"  最终准确率: {final_accuracy:.3f}")
    print(f"  总通信开销: {total_comm_bits:,} bits")
    print(f"  平均通信率: {avg_comm_rate:.3f}")
    print(f"  压缩比: {32/QUANTIZATION_BITS:.1f}:1")
    
    # 7. 保存结果
    results = {
        "experiment_config": {
            "dataset": DATASET_TYPE,
            "num_workers": NUM_WORKERS,
            "num_rounds": NUM_ROUNDS,
            "quantization_bits": QUANTIZATION_BITS,
            "learning_rate": LEARNING_RATE
        },
        "results": {
            "final_loss": float(final_loss),
            "final_accuracy": float(final_accuracy),
            "total_communication_bits": int(total_comm_bits),
            "average_communication_rate": float(avg_comm_rate),
            "training_time": float(training_time),
            "loss_history": [float(x) for x in loss_history],
            "accuracy_history": [float(x) for x in accuracy_history],
            "communication_costs": [int(x) for x in communication_costs]
        }
    }
    
    # 保存到Excel文件
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"flq_fed_{DATASET_TYPE}_{timestamp}.xlsx"
    
    os.makedirs("results", exist_ok=True)
    
    # 使用pandas保存Excel
    try:
        import pandas as pd
        
        # 创建Excel数据
        excel_data = []
        for i, (loss, cost) in enumerate(zip(loss_history, communication_costs)):
            acc = accuracy_history[i//10] if i % 10 == 0 and i//10 < len(accuracy_history) else None
            excel_data.append({
                'round': i,
                'loss': loss,
                'accuracy': acc,
                'comm_bits': cost
            })
        
        df = pd.DataFrame(excel_data)
        df.to_excel(f"results/{results_file}", index=False)
        print(f"📁 实验结果已保存到: results/{results_file}")
    except ImportError:
        print("⚠️ pandas未安装，跳过Excel保存")
    
    # 8. 绘制结果图表
    plot_results(loss_history, accuracy_history, communication_costs, server.communication_indicators)
    
    return results, server

def plot_results(loss_history, accuracy_history, communication_costs, communication_indicators):
    """绘制实验结果图表"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失收敛曲线
    ax1.plot(loss_history, 'b-', linewidth=2)
    ax1.set_xlabel('训练轮次')
    ax1.set_ylabel('平均损失')
    ax1.set_title('FLQ损失收敛曲线')
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    if accuracy_history:
        eval_rounds = np.arange(0, len(loss_history), 10)
        eval_rounds = eval_rounds[:len(accuracy_history)]
        ax2.plot(eval_rounds, accuracy_history, 'g-o', linewidth=2)
        ax2.set_xlabel('训练轮次')
        ax2.set_ylabel('测试准确率')
        ax2.set_title('FLQ准确率曲线')
        ax2.grid(True, alpha=0.3)
    
    # 通信开销
    cumulative_bits = np.cumsum(communication_costs) / 1e6  # 转换为Mbits
    ax3.plot(cumulative_bits, 'r-', linewidth=2)
    ax3.set_xlabel('训练轮次')
    ax3.set_ylabel('累计通信开销 (Mbits)')
    ax3.set_title('FLQ通信开销')
    ax3.grid(True, alpha=0.3)
    
    # 通信模式可视化（选择几个客户端展示）
    selected_workers = [0, NUM_WORKERS//2, NUM_WORKERS-1]
    for i, worker_id in enumerate(selected_workers):
        comm_pattern = communication_indicators[worker_id, :min(100, NUM_ROUNDS)]  # 只显示前100轮
        ax4.plot(comm_pattern + i * 1.2, label=f'客户端{worker_id}', linewidth=1.5)
    
    ax4.set_xlabel('训练轮次')
    ax4.set_ylabel('通信指示器')
    ax4.set_title('FLQ懒惰聚合通信模式')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表到results文件夹
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_file = f"flq_fed_{DATASET_TYPE}_{timestamp}.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{plot_file}", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 实验图表已保存到: results/{plot_file}")


# ---------------------
# 主程序入口
# ---------------------
if __name__ == "__main__":
    print("🎯 FLQ联邦学习实现")
    print("基于legacy/FLQ.py的量化和懒惰聚合机制")
    print("支持MNIST和Fashion-MNIST数据集\n")
    
    # 选择数据集
    # dataset_choice = input(f"选择数据集 (mnist/fashion_mnist) [{DATASET_TYPE}]: ").strip() or DATASET_TYPE
    DATASET_TYPE = "mnist"
    
    results, server = run_flq_federated_learning()
    
    print(f"\n🎉 所有实验完成！")
    print("📁 请查看results/目录下的结果文件和图表")

