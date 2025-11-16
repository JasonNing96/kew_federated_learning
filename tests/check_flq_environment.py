#!/usr/bin/env python3
"""
FLQ-Fed 环境检查脚本 - 验证运行前的环境配置
"""
import os
import sys
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("✓ Python版本检查...")
    version = sys.version_info
    print(f"  当前版本: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 8:
        print("  ✅ Python版本符合要求 (>=3.8)")
        return True
    else:
        print("  ❌ Python版本过低，需要 >=3.8")
        return False

def check_torch():
    """检查PyTorch和CUDA"""
    print("\n✓ PyTorch检查...")
    try:
        import torch
        print(f"  PyTorch版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"  ✅ CUDA可用")
            print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA版本: {torch.version.cuda}")
        else:
            print("  ⚠️  CUDA不可用，将使用CPU训练（速度较慢）")
        return True
    except ImportError:
        print("  ❌ PyTorch未安装")
        return False

def check_ultralytics():
    """检查Ultralytics YOLO"""
    print("\n✓ Ultralytics检查...")
    try:
        import ultralytics
        print(f"  Ultralytics版本: {ultralytics.__version__}")
        print("  ✅ Ultralytics已安装")
        return True
    except ImportError:
        print("  ❌ Ultralytics未安装")
        return False

def check_web_frameworks():
    """检查Web框架（联邦学习通信需要）"""
    print("\n✓ Web框架检查（联邦学习通信需要）...")
    
    all_ok = True
    
    try:
        import fastapi
        print(f"  ✅ FastAPI版本: {fastapi.__version__}")
    except ImportError:
        print("  ❌ FastAPI未安装")
        all_ok = False
    
    try:
        import uvicorn
        print(f"  ✅ Uvicorn已安装")
    except ImportError:
        print("  ❌ Uvicorn未安装")
        all_ok = False
    
    try:
        import requests
        print(f"  ✅ Requests版本: {requests.__version__}")
    except ImportError:
        print("  ❌ Requests未安装")
        all_ok = False
    
    return all_ok

def check_scientific_packages():
    """检查科学计算包"""
    print("\n✓ 科学计算包检查...")
    
    all_ok = True
    
    try:
        import numpy
        print(f"  ✅ NumPy版本: {numpy.__version__}")
    except ImportError:
        print("  ❌ NumPy未安装")
        all_ok = False
    
    try:
        import yaml
        print(f"  ✅ PyYAML已安装")
    except ImportError:
        print("  ❌ PyYAML未安装")
        all_ok = False
    
    return all_ok

def check_flq_modules():
    """检查FLQ模块"""
    print("\n✓ FLQ模块检查...")
    
    required_modules = [
        "flq_modules/__init__.py",
        "flq_modules/quantization.py",
        "flq_modules/aggregation.py",
        "flq_modules/utils.py",
        "flq_modules/config.py"
    ]
    
    all_ok = True
    for module in required_modules:
        if os.path.exists(module):
            print(f"  ✅ {module}")
        else:
            print(f"  ❌ {module} 缺失")
            all_ok = False
    
    # 尝试导入模块
    if all_ok:
        try:
            from flq_modules import quantization, aggregation, utils, config
            print("  ✅ 所有FLQ模块可正常导入")
        except ImportError as e:
            print(f"  ⚠️  模块导入失败: {e}")
            all_ok = False
    
    return all_ok

def check_core_modules():
    """检查核心模块（服务器和客户端）"""
    print("\n✓ 核心模块检查...")
    
    core_files = [
        "core/server.py",
        "core/client.py"
    ]
    
    all_ok = True
    for file in core_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ⚠️  {file} 未找到")
            all_ok = False
    
    return all_ok

def check_test_modules():
    """检查测试模块"""
    print("\n✓ 测试模块检查...")
    
    test_files = [
        "tests/test_quantization.py",
        "tests/test_aggregation.py",
        "tests/test_utils.py"
    ]
    
    all_ok = True
    for file in test_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} 缺失")
            all_ok = False
    
    return all_ok

def check_configs():
    """检查配置文件"""
    print("\n✓ 配置文件检查...")
    
    config_files = [
        "configs/flq_config.yaml"
    ]
    
    all_ok = True
    for file in config_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} 缺失")
            all_ok = False
    
    return all_ok

def check_main_script():
    """检查主脚本"""
    print("\n✓ 主脚本检查...")
    
    if os.path.exists("flq-fed.py"):
        print("  ✅ flq-fed.py")
        
        # 检查脚本是否可执行
        if os.access("flq-fed.py", os.X_OK):
            print("  ✅ flq-fed.py 具有执行权限")
        else:
            print("  ⚠️  flq-fed.py 没有执行权限（可选）")
        
        return True
    else:
        print("  ❌ flq-fed.py 缺失")
        return False

def check_model_file():
    """检查YOLO模型文件"""
    print("\n✓ YOLO模型文件检查...")
    
    model_files = ["yolov8n.pt", "yolov8s.pt"]
    found = False
    
    for model in model_files:
        if os.path.exists(model):
            print(f"  ✅ {model}")
            found = True
    
    if not found:
        print("  ⚠️  未找到YOLO预训练模型（yolov8n.pt或yolov8s.pt）")
        print("  💡 提示: 首次运行时会自动下载")
    
    return True  # 不强制要求

def check_optional_tools():
    """检查可选工具"""
    print("\n✓ 可选工具检查...")
    
    tools = [
        ("matplotlib", "可视化工具"),
        ("pandas", "数据分析"),
        ("pytest", "单元测试框架")
    ]
    
    for pkg, desc in tools:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg} ({desc})")
        except ImportError:
            print(f"  ⚠️  {pkg} 未安装 ({desc}，可选)")

def main():
    """主函数"""
    print("=" * 70)
    print("  FLQ-Fed 联邦学习量化框架 - 环境检查")
    print("=" * 70)
    
    # 切换到脚本所在目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"\n当前工作目录: {os.getcwd()}\n")
    
    results = []
    
    # 执行各项检查
    results.append(("Python版本", check_python_version()))
    results.append(("PyTorch", check_torch()))
    results.append(("Ultralytics", check_ultralytics()))
    results.append(("Web框架", check_web_frameworks()))
    results.append(("科学计算包", check_scientific_packages()))
    results.append(("FLQ模块", check_flq_modules()))
    results.append(("核心模块", check_core_modules()))
    results.append(("测试模块", check_test_modules()))
    results.append(("配置文件", check_configs()))
    results.append(("主脚本", check_main_script()))
    results.append(("模型文件", check_model_file()))
    
    # 可选工具检查
    check_optional_tools()
    
    # 总结
    print("\n" + "=" * 70)
    print("  检查总结")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 70)
    
    if all_passed:
        print("  🎉 所有检查通过！可以开始使用FLQ-Fed了。")
        print("\n  使用方法:")
        print("    # 运行单元测试:")
        print("    pytest tests/")
        print("\n    # 启动完整训练（推荐）:")
        print("    python flq-fed.py train --config configs/flq_config.yaml")
        print("\n    # 或手动启动服务器和客户端:")
        print("    终端1: python flq-fed.py server --config configs/flq_config.yaml")
        print("    终端2: python flq-fed.py client --id 1 --server http://localhost:8087")
    else:
        print("  ⚠️  部分检查未通过，请先解决问题。")
        print("\n  安装依赖（如果有requirements.txt）:")
        print("    pip install -r requirements.txt")
        print("\n  或手动安装:")
        print("    pip install torch ultralytics fastapi uvicorn requests numpy pyyaml")
    
    print("=" * 70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

