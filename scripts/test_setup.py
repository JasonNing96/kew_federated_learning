#!/usr/bin/env python3
"""
测试新架构是否正确设置
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_imports():
    """测试模块导入"""
    print("🧪 测试模块导入...")
    
    try:
        from app.config import Config
        print("  ✅ app.config")
        
        from app.model_utils import (
            model_to_vector, quantize_vector, fedavg_aggregate
        )
        print("  ✅ app.model_utils")
        
        # 检查文件存在（不实际导入，避免触发 Ultralytics）
        import importlib.util
        for module in ['app.server', 'app.client', 'app.runner']:
            spec = importlib.util.find_spec(module)
            if spec is None:
                raise ImportError(f"{module} not found")
        print("  ✅ app.server/client/runner (文件存在)")
        
        return True
    except Exception as e:
        print(f"  ❌ 导入失败: {e}")
        return False


def test_config():
    """测试配置加载"""
    print("\n🧪 测试配置加载...")
    
    try:
        from app.config import Config
        config = Config()
        
        assert config.rounds > 0
        assert config.clients_per_round > 0
        assert config.device in ["cuda:0", "cpu"]
        
        print(f"  ✅ 配置加载成功: {config}")
        return True
    except Exception as e:
        print(f"  ❌ 配置加载失败: {e}")
        return False


def test_utils():
    """测试工具函数"""
    print("\n🧪 测试工具函数...")
    
    try:
        import torch
        from app.model_utils import (
            quantize_vector, dequantize_vector,
            fedavg_aggregate, compute_model_size
        )
        
        # 测试量化
        vec = torch.randn(100)
        q, scale, zp = quantize_vector(vec, bits=8)
        dq = dequantize_vector(q, scale, zp, bits=8)
        print("  ✅ 量化/反量化")
        
        # 测试聚合
        sd1 = {"w": torch.randn(10)}
        sd2 = {"w": torch.randn(10)}
        agg = fedavg_aggregate([sd1, sd2], [100, 100])
        print("  ✅ FedAvg聚合")
        
        # 测试统计
        n_params, size_mb = compute_model_size(sd1, 32)
        print(f"  ✅ 模型统计 ({n_params} 参数, {size_mb:.2f} MB)")
        
        return True
    except Exception as e:
        print(f"  ❌ 工具函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_structure():
    """测试目录结构"""
    print("\n🧪 测试目录结构...")
    
    required_dirs = [
        "app",
        "configs",
        "scripts",
        "models",
        "data",
        "docs",
        "legacy"
    ]
    
    required_files = [
        "app/__init__.py",
        "app/runner.py",
        "app/server.py",
        "app/client.py",
        "app/model_utils.py",
        "app/config.py",
        "configs/flq_config.yaml",
        "scripts/run_fl.sh",
        "scripts/stop_fl.sh",
        "scripts/status.sh",
        "README.md",
        "QUICKSTART.md",
        "MIGRATION.md"
    ]
    
    all_ok = True
    
    for dir_name in required_dirs:
        dir_path = PROJECT_ROOT / dir_name
        if dir_path.exists():
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ 不存在")
            all_ok = False
    
    for file_name in required_files:
        file_path = PROJECT_ROOT / file_name
        if file_path.exists():
            print(f"  ✅ {file_name}")
        else:
            print(f"  ❌ {file_name} 不存在")
            all_ok = False
    
    return all_ok


def main():
    """主函数"""
    print("="*70)
    print("  FLQ-Fed 新架构测试")
    print("="*70)
    
    results = []
    
    results.append(("目录结构", test_structure()))
    results.append(("模块导入", test_imports()))
    results.append(("配置加载", test_config()))
    results.append(("工具函数", test_utils()))
    
    print("\n" + "="*70)
    print("  测试总结")
    print("="*70)
    
    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    
    if all_passed:
        print("  🎉 所有测试通过！新架构已就绪。")
        print("\n  下一步:")
        print("    1. 运行 ./scripts/run_fl.sh 开始训练")
        print("    2. 或查看 QUICKSTART.md 了解更多")
    else:
        print("  ⚠️  部分测试未通过，请检查上述错误。")
    
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

