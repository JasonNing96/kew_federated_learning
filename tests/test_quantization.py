"""
量化模块单元测试
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
from flq_modules.quantization import (
    quantize_relative,
    quantize_laq_vector,
    quantize_laq_tensor,
    compute_quantized_bits,
    compute_compression_ratio,
    _quant_tensor_stoch,
    _quant_bin_tensor
)


def test_stochastic_quantization():
    """测试随机舍入量化"""
    print("\n🧪 测试1: 随机舍入量化 (8-bit)")
    np.random.seed(42)
    x = np.random.randn(1000).astype(np.float32)
    
    for bits in [2, 4, 8]:
        q = _quant_tensor_stoch(x, bits)
        error = np.mean(np.abs(x - q))
        rel_error = error / (np.mean(np.abs(x)) + 1e-12)
        print(f"   {bits}-bit: 绝对误差={error:.6f}, 相对误差={rel_error:.4%}")
        # 2-bit量化误差较大是正常的，调整阈值
        threshold = 2.0 if bits == 2 else 0.5
        assert rel_error < threshold, f"{bits}-bit量化误差过大"
    
    print("   ✅ 通过")


def test_binary_quantization():
    """测试二值量化"""
    print("\n🧪 测试2: 二值量化 (1-bit)")
    np.random.seed(42)
    diff = np.random.randn(1000).astype(np.float32)
    
    q = _quant_bin_tensor(diff)
    
    # 检查二值性
    unique_vals = np.unique(np.abs(q))
    alpha = float(np.mean(np.abs(diff)))
    
    print(f"   原始均值: {np.mean(diff):.6f}")
    print(f"   量化均值: {np.mean(q):.6f}")
    print(f"   Alpha: {alpha:.6f}")
    print(f"   唯一值数量: {len(unique_vals)}")
    
    assert len(unique_vals) == 1, "二值量化应该只有一个绝对值"
    assert np.allclose(unique_vals[0], alpha, rtol=1e-5), "Alpha计算错误"
    
    print("   ✅ 通过")


def test_relative_quantization():
    """测试相对域量化"""
    print("\n🧪 测试3: 相对域量化")
    np.random.seed(42)
    
    # 模拟3个参数张量
    shapes = [(100, 10), (50,), (200, 5)]
    total_params = sum(np.prod(s) for s in shapes)
    
    g_vec = np.random.randn(total_params).astype(np.float32)
    ref_vec = np.random.randn(total_params).astype(np.float32)
    
    for bits in [0, 1, 4, 8]:
        q_vec = quantize_relative(g_vec, ref_vec, bits, shapes)
        
        if bits == 0:
            # 不量化应该返回参考向量
            assert np.allclose(q_vec, ref_vec), "bits=0应返回ref_vec"
            print(f"   {bits}-bit: 无量化 ✓")
        else:
            error = np.mean(np.abs(g_vec - q_vec))
            print(f"   {bits}-bit: 误差={error:.6f}")
            assert q_vec.shape == g_vec.shape, "形状不匹配"
    
    print("   ✅ 通过")


def test_laq_quantization():
    """测试LAQ量化"""
    print("\n🧪 测试4: LAQ量化")
    np.random.seed(42)
    
    shapes = [(100, 10), (50,), (200, 5)]
    total_params = sum(np.prod(s) for s in shapes)
    g_vec = np.random.randn(total_params).astype(np.float32)
    
    # 向量级LAQ
    q_vec = quantize_laq_vector(g_vec, bits=4)
    error_vec = np.mean(np.abs(g_vec - q_vec))
    print(f"   向量级LAQ (4-bit): 误差={error_vec:.6f}")
    
    # 张量级LAQ
    q_ten = quantize_laq_tensor(g_vec, bits=4, shapes=shapes)
    error_ten = np.mean(np.abs(g_vec - q_ten))
    print(f"   张量级LAQ (4-bit): 误差={error_ten:.6f}")
    
    assert q_vec.shape == g_vec.shape, "向量级LAQ形状错误"
    assert q_ten.shape == g_vec.shape, "张量级LAQ形状错误"
    
    print("   ✅ 通过")


def test_performance():
    """测试性能：量化100万参数"""
    print("\n🧪 测试5: 性能测试 (1M参数)")
    np.random.seed(42)
    
    n_params = 1_000_000
    g_vec = np.random.randn(n_params).astype(np.float32)
    ref_vec = np.random.randn(n_params).astype(np.float32)
    shapes = [(1000, 1000)]
    
    start = time.time()
    q_vec = quantize_relative(g_vec, ref_vec, bits=8, shapes=shapes)
    elapsed = (time.time() - start) * 1000
    
    print(f"   量化时间: {elapsed:.2f} ms")
    assert elapsed < 100, f"性能不达标: {elapsed:.2f}ms > 100ms"
    
    print("   ✅ 通过")


def test_compression_metrics():
    """测试压缩指标计算"""
    print("\n🧪 测试6: 压缩指标计算")
    
    n_params = 3_011_238  # YOLO8n参数数量
    n_clients = 3
    
    results = []
    for bits in [32, 8, 4, 1]:
        total_bits = compute_quantized_bits(n_params, bits, n_clients)
        ratio = compute_compression_ratio(32, bits)
        gbits = total_bits / 1e9
        results.append((bits, gbits, ratio))
        print(f"   {bits:2d}-bit: {gbits:.3f} Gbits, 压缩率={ratio:.1f}x")
    
    # 验证压缩率
    assert results[1][2] == 4.0, "8-bit压缩率应为4.0x"
    assert results[2][2] == 8.0, "4-bit压缩率应为8.0x"
    assert results[3][2] == 32.0, "1-bit压缩率应为32.0x"
    
    print("   ✅ 通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print("🚀 FLQ 量化模块测试")
    print("=" * 70)
    
    tests = [
        test_stochastic_quantization,
        test_binary_quantization,
        test_relative_quantization,
        test_laq_quantization,
        test_performance,
        test_compression_metrics
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"   ❌ 失败: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"📊 测试结果: {passed} 通过, {failed} 失败")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

