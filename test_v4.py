#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLQ v4 测试脚本
验证PyTorch版本的所有功能
"""
import subprocess
import sys
import os
import pandas as pd

def run_test(cmd, description):
    """运行测试命令"""
    print(f"\n{'='*60}")
    print(f"测试: {description}")
    print(f"命令: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ 测试通过")
            print("输出:", result.stdout.split('\n')[-3:-1])  # 显示最后几行
            return True
        else:
            print("❌ 测试失败")
            print("错误:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("⏰ 测试超时")
        return False
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

def verify_output(filename, expected_sheets):
    """验证输出文件"""
    try:
        if not os.path.exists(filename):
            print(f"❌ 文件不存在: {filename}")
            return False
        
        df_dict = pd.read_excel(filename, sheet_name=None)
        sheets = list(df_dict.keys())
        
        print(f"📊 文件: {filename}")
        print(f"   工作表: {sheets}")
        
        for sheet in expected_sheets:
            if sheet in sheets:
                print(f"   ✅ {sheet}: {df_dict[sheet].shape}")
            else:
                print(f"   ❌ 缺少工作表: {sheet}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ 验证文件失败: {e}")
        return False

def main():
    print("🚀 开始FLQ v4版本功能测试")
    
    # 测试用例
    tests = [
        {
            "cmd": "python flq_fed_v4.py --dataset mnist --mode bbit --iters 5 --M 3 --batch 32",
            "desc": "MNIST + BBIT模式",
            "output": "results/results_mnist_bbit_5.xlsx",
            "sheets": ["curve_bbit"]
        },
        {
            "cmd": "python flq_fed_v4.py --dataset mnist --mode bin --iters 5 --M 3 --batch 32",
            "desc": "MNIST + BIN模式 (二值量化)",
            "output": "results/results_mnist_bin_5.xlsx", 
            "sheets": ["curve_bin", "bin_bin"]
        },
        {
            "cmd": "python flq_fed_v4.py --dataset fmnist --mode laq8 --iters 5 --M 3 --batch 32",
            "desc": "Fashion-MNIST + LAQ8模式",
            "output": "results/results_fmnist_laq8_5.xlsx",
            "sheets": ["curve_laq8"]
        },
        {
            "cmd": "python flq_fed_v4.py --dataset mnist --mode fedavg --iters 5 --M 3 --batch 32",
            "desc": "MNIST + FedAvg模式",
            "output": "results/results_mnist_fedavg_5.xlsx",
            "sheets": ["curve_fedavg"]
        },
        {
            "cmd": "python flq_fed_v4.py --dataset mnist --mode bbit --partition non_iid --dir_alpha 0.1 --iters 5 --M 3",
            "desc": "非IID数据分布",
            "output": "results/results_mnist_bbit_5.xlsx",
            "sheets": ["curve_bbit"]
        }
    ]
    
    passed = 0
    total = len(tests)
    
    for i, test in enumerate(tests, 1):
        print(f"\n[{i}/{total}] ", end="")
        
        # 运行训练测试
        if run_test(test["cmd"], test["desc"]):
            # 验证输出文件
            if verify_output(test["output"], test["sheets"]):
                passed += 1
            else:
                print("❌ 输出文件验证失败")
        
    # 测试结果汇总
    print(f"\n{'='*60}")
    print(f"测试完成: {passed}/{total} 通过")
    print('='*60)
    
    if passed == total:
        print("🎉 所有测试通过！FLQ v4版本功能正常")
        return 0
    else:
        print(f"⚠️  有 {total-passed} 个测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
