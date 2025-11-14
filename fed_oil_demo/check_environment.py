#!/usr/bin/env python3
"""
环境检查脚本 - 验证运行前的环境配置
"""
import os
import sys

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
            return True
        else:
            print("  ⚠️  CUDA不可用，将使用CPU训练（速度较慢）")
            return True
    except ImportError:
        print("  ❌ PyTorch未安装")
        return False

def check_ultralytics():
    """检查Ultralytics"""
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
    """检查Web框架（联邦学习需要）"""
    print("\n✓ Web框架检查（联邦学习需要）...")
    
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

def check_files():
    """检查必要的文件和目录"""
    print("\n✓ 文件和目录检查...")
    
    required_files = [
        "train_gpu_only.py",
        "server.py",
        "client.py",
        "split_dataset.py",
        "start_server.sh",
        "start_clients.sh",
        "stop_all.sh",
        "yolov8s.pt",
        "yolov8n.pt",
        "requirements.txt",
        "README.md"
    ]
    
    required_dirs = [
        "oil-detection-2-2",
    ]
    
    all_ok = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} 缺失")
            all_ok = False
    
    for dir in required_dirs:
        if os.path.isdir(dir):
            print(f"  ✅ {dir}/")
        else:
            print(f"  ❌ {dir}/ 缺失")
            all_ok = False
    
    return all_ok

def check_datasets():
    """检查数据集配置"""
    print("\n✓ 数据集配置检查...")
    
    # 检查原始数据集
    if os.path.exists("oil-detection-2-2/data.yaml"):
        print(f"  ✅ oil-detection-2-2/data.yaml (原始数据集)")
    else:
        print(f"  ❌ oil-detection-2-2/data.yaml 缺失")
        return False
    
    # 检查客户端数据（可能还未生成）
    client_data_exists = True
    for i in range(1, 4):
        client_dir = f"client{i}"
        if os.path.exists(f"{client_dir}/oil.yaml"):
            print(f"  ✅ {client_dir}/oil.yaml")
        else:
            print(f"  ⚠️  {client_dir}/oil.yaml 未生成（需要运行 split_dataset.py）")
            client_data_exists = False
    
    if not client_data_exists:
        print(f"  💡 提示: 运行 'python split_dataset.py' 生成客户端数据")
    
    return True  # 原始数据集存在即可

def check_optional_packages():
    """检查可选包"""
    print("\n✓ 可选包检查...")
    
    packages = [
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("yaml", "yaml")
    ]
    
    for pkg_name, import_name in packages:
        try:
            __import__(import_name)
            print(f"  ✅ {pkg_name}")
        except ImportError:
            print(f"  ⚠️  {pkg_name} 未安装（可选）")

def main():
    """主函数"""
    print("=" * 60)
    print("  Fed Oil Detection Demo - 环境检查")
    print("=" * 60)
    
    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"\n当前工作目录: {os.getcwd()}\n")
    
    results = []
    
    # 执行各项检查
    results.append(("Python版本", check_python_version()))
    results.append(("PyTorch", check_torch()))
    results.append(("Ultralytics", check_ultralytics()))
    results.append(("Web框架", check_web_frameworks()))
    results.append(("文件和目录", check_files()))
    results.append(("数据集配置", check_datasets()))
    
    # 可选包检查
    check_optional_packages()
    
    # 总结
    print("\n" + "=" * 60)
    print("  检查总结")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("  🎉 所有检查通过！可以开始训练了。")
        print("\n  步骤1: 切分数据集（首次使用）:")
        print("    python split_dataset.py")
        print("\n  步骤2: 运行中心化训练:")
        print("    python train_gpu_only.py")
        print("    或: ./run_centralized.sh")
        print("\n  步骤3: 运行联邦学习训练:")
        print("    终端1: ./start_server.sh")
        print("    终端2: ./start_clients.sh")
    else:
        print("  ⚠️  部分检查未通过，请先解决问题。")
        print("\n  安装依赖:")
        print("    pip install -r requirements.txt")
    
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
