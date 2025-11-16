"""
聚合模块单元测试
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from flq_modules.aggregation import (
    fedavg_aggregate,
    QuantizedAggregator,
    ErrorFeedback,
    LazySelector,
    compute_aggregation_weights
)


def test_fedavg_uniform():
    """测试FedAvg均匀聚合"""
    print("\n🧪 测试1: FedAvg均匀聚合")
    np.random.seed(42)
    
    # 3个客户端，每个1000参数
    client_vecs = [
        np.random.randn(1000).astype(np.float32),
        np.random.randn(1000).astype(np.float32),
        np.random.randn(1000).astype(np.float32)
    ]
    
    # 均匀聚合
    aggregated = fedavg_aggregate(client_vecs)
    
    # 验证：应该等于平均值
    expected = (client_vecs[0] + client_vecs[1] + client_vecs[2]) / 3
    diff = np.max(np.abs(aggregated - expected))
    
    print(f"   聚合向量形状: {aggregated.shape}")
    print(f"   最大误差: {diff:.8f}")
    assert diff < 1e-6, f"均匀聚合错误: {diff}"
    
    print("   ✅ 通过")


def test_fedavg_weighted():
    """测试FedAvg加权聚合"""
    print("\n🧪 测试2: FedAvg加权聚合")
    np.random.seed(42)
    
    client_vecs = [
        np.ones(100, dtype=np.float32) * 1.0,
        np.ones(100, dtype=np.float32) * 2.0,
        np.ones(100, dtype=np.float32) * 3.0
    ]
    
    # 权重 [0.5, 0.3, 0.2]
    weights = [0.5, 0.3, 0.2]
    aggregated = fedavg_aggregate(client_vecs, weights)
    
    # 期望值: 1.0*0.5 + 2.0*0.3 + 3.0*0.2 = 1.7
    expected_val = 1.7
    actual_val = aggregated[0]
    
    print(f"   期望值: {expected_val}")
    print(f"   实际值: {actual_val}")
    assert abs(actual_val - expected_val) < 1e-6, "加权聚合错误"
    
    print("   ✅ 通过")


def test_quantized_aggregator():
    """测试量化聚合器"""
    print("\n🧪 测试3: 量化聚合器")
    np.random.seed(42)
    
    # 模拟参数
    shapes = [(100, 10), (50,)]
    total_params = 100*10 + 50
    
    ref_vec = np.random.randn(total_params).astype(np.float32)
    client_vecs = [
        ref_vec + np.random.randn(total_params).astype(np.float32) * 0.1
        for _ in range(3)
    ]
    
    # 8-bit量化聚合
    aggregator = QuantizedAggregator(bits=8, shapes=shapes, use_error_feedback=False)
    
    quantized_vecs = []
    for i, vec in enumerate(client_vecs):
        q_vec = aggregator.quantize_upload(vec, ref_vec, client_id=f"client_{i}")
        quantized_vecs.append(q_vec)
    
    aggregated = aggregator.aggregate(quantized_vecs)
    
    print(f"   量化比特数: 8")
    print(f"   客户端数量: 3")
    print(f"   聚合结果形状: {aggregated.shape}")
    assert aggregated.shape == (total_params,), "聚合结果形状错误"
    
    print("   ✅ 通过")


def test_error_feedback():
    """测试误差反馈"""
    print("\n🧪 测试4: 误差反馈")
    np.random.seed(42)
    
    shapes = [(100, 10)]
    total_params = 1000
    
    ref_vec = np.random.randn(total_params).astype(np.float32)
    client_vec = ref_vec + np.random.randn(total_params).astype(np.float32) * 0.5
    
    # 不带误差反馈
    aggregator_no_ef = QuantizedAggregator(bits=4, shapes=shapes, use_error_feedback=False)
    q1_no_ef = aggregator_no_ef.quantize_upload(client_vec, ref_vec, "client_1")
    error1_no_ef = np.linalg.norm(client_vec - q1_no_ef)
    
    # 带误差反馈
    aggregator_with_ef = QuantizedAggregator(bits=4, shapes=shapes, use_error_feedback=True)
    q1_with_ef = aggregator_with_ef.quantize_upload(client_vec, ref_vec, "client_1")
    
    # 第二次量化（应该补偿之前的误差）
    q2_with_ef = aggregator_with_ef.quantize_upload(client_vec, ref_vec, "client_1")
    
    print(f"   无误差反馈误差: {error1_no_ef:.6f}")
    print(f"   误差反馈缓存大小: {len(aggregator_with_ef.error_feedback)}")
    assert "client_1" in aggregator_with_ef.error_feedback, "误差未缓存"
    
    print("   ✅ 通过")


def test_lazy_selector_norm():
    """测试懒惰选择（基于范数）"""
    print("\n🧪 测试5: 懒惰选择（基于范数）")
    np.random.seed(42)
    
    # 5个客户端，选择前3个
    selector = LazySelector(total_clients=5, selection_ratio=0.6)
    
    client_vecs = [
        np.random.randn(100).astype(np.float32) * (i + 1)  # 范数递增
        for i in range(5)
    ]
    client_ids = [f"client_{i}" for i in range(5)]
    
    selected_indices = selector.select_by_norm(client_vecs, client_ids)
    
    print(f"   总客户端数: 5")
    print(f"   选择数量: {selector.num_selected}")
    print(f"   选中索引: {selected_indices}")
    
    # 验证：应该选择范数最大的
    assert len(selected_indices) == 3, "选择数量错误"
    assert 4 in selected_indices, "应该选择范数最大的客户端"
    
    print("   ✅ 通过")


def test_lazy_selector_random():
    """测试随机选择"""
    print("\n🧪 测试6: 随机选择")
    np.random.seed(42)
    
    selector = LazySelector(total_clients=10, selection_ratio=0.3)
    selected = selector.select_random(10)
    
    print(f"   总客户端数: 10")
    print(f"   选择数量: {len(selected)}")
    print(f"   选中索引: {selected}")
    
    assert len(selected) == 3, "随机选择数量错误"
    assert len(set(selected)) == 3, "存在重复选择"
    
    print("   ✅ 通过")


def test_aggregation_weights():
    """测试聚合权重计算"""
    print("\n🧪 测试7: 聚合权重计算")
    np.random.seed(42)
    
    client_vecs = [
        np.ones(100, dtype=np.float32) * 1.0,
        np.ones(100, dtype=np.float32) * 2.0,
        np.ones(100, dtype=np.float32) * 3.0
    ]
    
    # 均匀权重
    weights_uniform = compute_aggregation_weights(client_vecs, "uniform")
    assert np.allclose(weights_uniform, [1/3, 1/3, 1/3]), "均匀权重错误"
    print(f"   均匀权重: {weights_uniform}")
    
    # 基于范数
    weights_norm = compute_aggregation_weights(client_vecs, "norm")
    assert sum(weights_norm) - 1.0 < 1e-6, "权重和应为1"
    print(f"   范数权重: {[f'{w:.3f}' for w in weights_norm]}")
    
    # 逆范数
    weights_inv = compute_aggregation_weights(client_vecs, "inverse_norm")
    assert sum(weights_inv) - 1.0 < 1e-6, "权重和应为1"
    print(f"   逆范数权重: {[f'{w:.3f}' for w in weights_inv]}")
    
    print("   ✅ 通过")


def test_error_feedback_class():
    """测试ErrorFeedback类"""
    print("\n🧪 测试8: ErrorFeedback类")
    np.random.seed(42)
    
    shapes = [(50, 20)]
    total_params = 1000
    
    ref_vec = np.zeros(total_params, dtype=np.float32)
    vector = np.random.randn(total_params).astype(np.float32)
    
    ef = ErrorFeedback()
    
    # 第一次量化
    q1 = ef.compress_with_feedback(vector, ref_vec, bits=4, shapes=shapes)
    assert ef.accumulated_error is not None, "误差未累积"
    
    # 第二次量化（应该应用误差补偿）
    q2 = ef.compress_with_feedback(vector, ref_vec, bits=4, shapes=shapes)
    
    print(f"   第一次量化误差范数: {np.linalg.norm(vector - q1):.6f}")
    print(f"   累积误差范数: {np.linalg.norm(ef.accumulated_error):.6f}")
    
    # 重置
    ef.reset()
    assert ef.accumulated_error is None, "重置失败"
    print("   ✅ 通过")


def test_integration():
    """测试端到端集成"""
    print("\n🧪 测试9: 端到端集成")
    np.random.seed(42)
    
    # 模拟3客户端，10轮训练
    shapes = [(100, 10), (50,)]
    total_params = 1050
    num_clients = 3
    num_rounds = 10
    
    # 初始化全局模型
    global_vec = np.random.randn(total_params).astype(np.float32)
    
    # 创建量化聚合器
    aggregator = QuantizedAggregator(bits=8, shapes=shapes, use_error_feedback=True)
    
    for round_idx in range(num_rounds):
        # 模拟客户端训练
        client_vecs = [
            global_vec + np.random.randn(total_params).astype(np.float32) * 0.1
            for _ in range(num_clients)
        ]
        
        # 量化上传
        quantized_vecs = []
        for i, vec in enumerate(client_vecs):
            q_vec = aggregator.quantize_upload(vec, global_vec, f"client_{i}")
            quantized_vecs.append(q_vec)
        
        # 聚合
        global_vec = aggregator.aggregate(quantized_vecs)
    
    print(f"   训练轮数: {num_rounds}")
    print(f"   客户端数: {num_clients}")
    print(f"   最终模型范数: {np.linalg.norm(global_vec):.6f}")
    print("   ✅ 通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print("🚀 FLQ 聚合模块测试")
    print("=" * 70)
    
    tests = [
        test_fedavg_uniform,
        test_fedavg_weighted,
        test_quantized_aggregator,
        test_error_feedback,
        test_lazy_selector_norm,
        test_lazy_selector_random,
        test_aggregation_weights,
        test_error_feedback_class,
        test_integration
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"   ❌ 失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"📊 测试结果: {passed} 通过, {failed} 失败")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

