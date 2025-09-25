#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLQ 联邦学习环境测试脚本
验证 conda fed 环境是否正确配置
"""

import sys
import subprocess
import importlib
from typing import Tuple, List

# 颜色输出
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def check_python_version() -> Tuple[bool, str]:
    """检查 Python 版本"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} (需要 3.9+)"

def check_package(package_name: str, min_version: str = None) -> Tuple[bool, str]:
    """检查包是否安装及版本"""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        
        if min_version and version != 'unknown':
            from packaging import version as pkg_version
            if pkg_version.parse(version) >= pkg_version.parse(min_version):
                return True, f"{package_name} {version}"
            else:
                return False, f"{package_name} {version} (需要 {min_version}+)"
        else:
            return True, f"{package_name} {version}"
    except ImportError:
        return False, f"{package_name} 未安装"
    except Exception as e:
        return False, f"{package_name} 错误: {str(e)}"

def check_tensorflow_gpu() -> Tuple[bool, str]:
    """检查 TensorFlow GPU 支持"""
    try:
        import tensorflow as tf
        
        # 检查版本
        tf_version = tf.__version__
        
        # 检查 GPU 可用性
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            gpu_info = []
            for i, gpu in enumerate(gpus):
                gpu_info.append(f"GPU {i}: {gpu.name}")
            return True, f"TensorFlow {tf_version}, GPU可用: {', '.join(gpu_info)}"
        else:
            return False, f"TensorFlow {tf_version}, 无可用GPU"
    except Exception as e:
        return False, f"TensorFlow 错误: {str(e)}"

def check_cuda() -> Tuple[bool, str]:
    """检查 CUDA 环境"""
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # 提取 CUDA 版本
            lines = result.stdout.split('\n')
            for line in lines:
                if 'release' in line.lower():
                    return True, f"CUDA 可用: {line.strip()}"
            return True, "CUDA 可用"
        else:
            return False, "CUDA 不可用"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "CUDA 命令未找到"
    except Exception as e:
        return False, f"CUDA 检查错误: {str(e)}"

def check_conda_env() -> Tuple[bool, str]:
    """检查 conda 环境"""
    import os
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'None')
    if conda_env == 'fed':
        return True, f"Conda 环境: {conda_env}"
    else:
        return False, f"当前环境: {conda_env} (应该是 'fed')"

def run_simple_test() -> Tuple[bool, str]:
    """运行简单的 TensorFlow 测试"""
    try:
        import tensorflow as tf
        import numpy as np
        
        # 创建简单模型
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(1)
        ])
        
        # 编译模型
        model.compile(optimizer='adam', loss='mse')
        
        # 生成测试数据
        X = np.random.random((100, 5))
        y = np.random.random((100, 1))
        
        # 训练一步
        history = model.fit(X, y, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        return True, f"TensorFlow 功能测试通过，Loss: {loss:.4f}"
    except Exception as e:
        return False, f"TensorFlow 功能测试失败: {str(e)}"

def main():
    print(f"{Colors.BLUE}{Colors.BOLD}========================================{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}    FLQ 联邦学习环境检查{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}========================================{Colors.END}\n")
    
    # 检查项目列表
    checks = [
        ("Python 版本", check_python_version),
        ("Conda 环境", check_conda_env),
        ("CUDA 支持", check_cuda),
        ("TensorFlow GPU", check_tensorflow_gpu),
        ("numpy", lambda: check_package("numpy", "1.21.0")),
        ("pandas", lambda: check_package("pandas")),
        ("flask", lambda: check_package("flask")),
        ("requests", lambda: check_package("requests")),
        ("openpyxl", lambda: check_package("openpyxl")),
        ("scipy", lambda: check_package("scipy")),
        ("matplotlib", lambda: check_package("matplotlib")),
        ("TensorFlow 功能", run_simple_test),
    ]
    
    results = []
    all_passed = True
    
    for name, check_func in checks:
        try:
            passed, message = check_func()
            results.append((name, passed, message))
            
            if passed:
                print(f"{Colors.GREEN}✓{Colors.END} {name}: {message}")
            else:
                print(f"{Colors.RED}✗{Colors.END} {name}: {message}")
                all_passed = False
        except Exception as e:
            print(f"{Colors.RED}✗{Colors.END} {name}: 检查失败 - {str(e)}")
            all_passed = False
    
    # 总结
    print(f"\n{Colors.BLUE}{Colors.BOLD}========================================{Colors.END}")
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 所有检查通过！环境配置正确。{Colors.END}")
        print(f"{Colors.GREEN}现在可以运行 FLQ 联邦学习系统了。{Colors.END}")
        
        print(f"\n{Colors.BLUE}快速启动命令:{Colors.END}")
        print(f"  Master: {Colors.YELLOW}./start_master.sh{Colors.END}")
        print(f"  Worker: {Colors.YELLOW}./start_worker.sh --worker_id 0 --master_ip 127.0.0.1{Colors.END}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ 环境检查失败！{Colors.END}")
        print(f"{Colors.RED}请按照 README.md 重新配置环境。{Colors.END}")
        
        print(f"\n{Colors.YELLOW}修复建议:{Colors.END}")
        print(f"  1. 确保已激活 conda 环境: {Colors.YELLOW}conda activate fed{Colors.END}")
        print(f"  2. 重新创建环境: {Colors.YELLOW}conda env create -f environment.yml{Colors.END}")
        print(f"  3. 检查 Jetson Nano CUDA 安装")
    
    print(f"{Colors.BLUE}{Colors.BOLD}========================================{Colors.END}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
