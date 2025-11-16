"""
工具函数模块单元测试
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
from flq_modules.utils import (
    get_model_params,
    get_param_shapes,
    count_parameters,
    model_to_vector,
    vector_to_model,
    ParamMetadata,
    compute_communication_cost,
    format_bytes,
    compute_model_diff,
    apply_model_diff
)


# 创建简单测试模型
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_model_params_extraction():
    """测试模型参数提取"""
    print("\n🧪 测试1: 模型参数提取")
    model = SimpleNet()
    
    # 测试从模型提取
    params = get_model_params(model)
    assert len(params) == 4, f"应该有4个参数，实际{len(params)}"
    
    # 测试从state_dict提取
    state_dict = model.state_dict()
    params2 = get_model_params(state_dict)
    assert len(params2) == 4, "从state_dict提取参数失败"
    
    print(f"   提取参数: {list(params.keys())}")
    print("   ✅ 通过")


def test_param_shapes():
    """测试参数形状提取"""
    print("\n🧪 测试2: 参数形状提取")
    model = SimpleNet()
    params = get_model_params(model)
    shapes = get_param_shapes(params)
    
    expected_shapes = [
        (20, 10),  # fc1.weight
        (20,),     # fc1.bias
        (5, 20),   # fc2.weight
        (5,)       # fc2.bias
    ]
    
    assert shapes == expected_shapes, f"形状不匹配: {shapes} vs {expected_shapes}"
    print(f"   参数形状: {shapes}")
    print("   ✅ 通过")


def test_count_parameters():
    """测试参数计数"""
    print("\n🧪 测试3: 参数计数")
    model = SimpleNet()
    params = get_model_params(model)
    count = count_parameters(params)
    
    # 计算预期值: (20*10 + 20) + (5*20 + 5) = 325
    expected = 20*10 + 20 + 5*20 + 5
    assert count == expected, f"参数数量错误: {count} vs {expected}"
    
    print(f"   总参数数: {count:,}")
    print("   ✅ 通过")


def test_model_vector_conversion():
    """测试模型↔向量转换（无损）"""
    print("\n🧪 测试4: 模型↔向量转换")
    model = SimpleNet()
    params_orig = get_model_params(model)
    
    # 模型→向量
    vec = model_to_vector(params_orig)
    print(f"   向量形状: {vec.shape}, dtype: {vec.dtype}")
    
    # 向量→模型
    param_names = list(params_orig.keys())
    shapes = get_param_shapes(params_orig)
    params_restored = vector_to_model(vec, param_names, shapes)
    
    # 验证无损
    for name in param_names:
        orig = params_orig[name].cpu().numpy()
        restored = params_restored[name].cpu().numpy()
        diff = np.max(np.abs(orig - restored))
        assert diff < 1e-6, f"{name} 转换有损失: {diff}"
    
    print(f"   ✅ 通过（无损转换）")


def test_param_metadata():
    """测试ParamMetadata类"""
    print("\n🧪 测试5: ParamMetadata类")
    model = SimpleNet()
    metadata = ParamMetadata(model)
    
    print(f"   {metadata}")
    assert metadata.num_params == 325, "参数数量错误"
    assert len(metadata.param_names) == 4, "参数名称数量错误"
    
    # 测试to_vector
    vec = metadata.to_vector()
    assert vec.shape[0] == 325, "向量长度错误"
    
    # 测试from_vector
    params = metadata.from_vector(vec)
    assert len(params) == 4, "恢复参数数量错误"
    
    print("   ✅ 通过")


def test_communication_cost():
    """测试通信开销计算"""
    print("\n🧪 测试6: 通信开销计算")
    
    # YOLO8n参数
    num_params = 3_011_238
    num_clients = 3
    num_rounds = 10
    
    for bits in [32, 8, 4, 1]:
        cost = compute_communication_cost(num_params, bits, num_clients, num_rounds)
        print(f"   {bits:2d}-bit: 上行={cost['upload_gbits']:.3f}GB, "
              f"下行={cost['download_gbits']:.3f}GB, "
              f"总计={cost['total_gbits']:.3f}GB")
        
        assert cost['upload_gbits'] > 0, "上行流量应>0"
        assert cost['download_gbits'] > 0, "下行流量应>0"
    
    print("   ✅ 通过")


def test_format_bytes():
    """测试字节格式化"""
    print("\n🧪 测试7: 字节格式化")
    
    test_cases = [
        (1024, "1.0 KB"),
        (1024**2, "1.0 MB"),
        (1024**3, "1.0 GB"),
        (123456789, "117.7 MB")
    ]
    
    for num_bytes, expected_prefix in test_cases:
        result = format_bytes(num_bytes)
        # 只检查单位是否正确
        assert expected_prefix.split()[-1] in result, f"{result} 应包含 {expected_prefix.split()[-1]}"
    
    print(f"   示例: {format_bytes(123456789)}")
    print("   ✅ 通过")


def test_model_diff():
    """测试模型差值计算"""
    print("\n🧪 测试8: 模型差值计算")
    
    model1 = SimpleNet()
    model2 = SimpleNet()
    
    # 修改model2的参数
    with torch.no_grad():
        for p in model2.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    
    params1 = get_model_params(model1)
    params2 = get_model_params(model2)
    
    # 计算差值
    diff = compute_model_diff(params2, params1)
    assert len(diff) == 4, "差值字典应有4个元素"
    
    # 应用差值
    params_new = apply_model_diff(params1, diff, alpha=1.0)
    
    # 验证：params1 + diff = params2
    for name in params1.keys():
        expected = params2[name].cpu().numpy()
        actual = params_new[name].cpu().numpy()
        max_diff = np.max(np.abs(expected - actual))
        assert max_diff < 1e-5, f"{name} 差值应用错误: {max_diff}"
    
    print("   ✅ 通过")


def test_cuda_compatibility():
    """测试CUDA兼容性（如果可用）"""
    print("\n🧪 测试9: CUDA兼容性")
    
    if torch.cuda.is_available():
        model = SimpleNet().cuda()
        metadata = ParamMetadata(model)
        
        vec = metadata.to_vector()
        assert vec.dtype == np.float32, "向量应为float32"
        
        params = metadata.from_vector(vec)
        assert next(iter(params.values())).device.type == 'cuda', "参数应在CUDA上"
        
        print("   ✅ 通过（CUDA可用）")
    else:
        print("   ⚠️  跳过（CUDA不可用）")


def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print("🚀 FLQ 工具函数模块测试")
    print("=" * 70)
    
    tests = [
        test_model_params_extraction,
        test_param_shapes,
        test_count_parameters,
        test_model_vector_conversion,
        test_param_metadata,
        test_communication_cost,
        test_format_bytes,
        test_model_diff,
        test_cuda_compatibility
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

