#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLQ v3 vs v4 对比测试
验证PyTorch版本与TensorFlow版本的一致性
"""
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_experiment(version, dataset, mode, iters=50):
    """运行实验"""
    if version == "v3":
        cmd = f"python flq_fed_v3.py --dataset {dataset} --mode {mode} --iters {iters} --M 5 --batch 64"
    else:
        cmd = f"python flq_fed_v4.py --dataset {dataset} --mode {mode} --iters {iters} --M 5 --batch 64"
    
    print(f"运行 {version}: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ {version} 运行失败: {result.stderr}")
        return None
    
    # 读取结果
    filename = f"results/results_{dataset}_{mode}_{iters}.xlsx"
    try:
        df = pd.read_excel(filename, sheet_name=f"curve_{mode}")
        return df
    except Exception as e:
        print(f"❌ 读取结果失败: {e}")
        return None

def compare_results(df_v3, df_v4, metric="acc"):
    """对比结果"""
    if df_v3 is None or df_v4 is None:
        return False
    
    v3_values = df_v3[metric].values
    v4_values = df_v4[metric].values
    
    # 计算相关性
    correlation = np.corrcoef(v3_values, v4_values)[0, 1]
    
    # 计算RMSE
    rmse = np.sqrt(np.mean((v3_values - v4_values) ** 2))
    
    # 最终值对比
    final_v3 = v3_values[-1]
    final_v4 = v4_values[-1]
    final_diff = abs(final_v3 - final_v4)
    
    print(f"📊 {metric.upper()} 对比:")
    print(f"   相关性: {correlation:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   最终值 v3: {final_v3:.4f}")
    print(f"   最终值 v4: {final_v4:.4f}")
    print(f"   最终差异: {final_diff:.4f}")
    
    # 判断是否一致（相关性>0.8，最终差异<0.1）
    is_consistent = correlation > 0.8 and final_diff < 0.1
    print(f"   一致性: {'✅ 通过' if is_consistent else '❌ 不一致'}")
    
    return is_consistent

def plot_comparison(df_v3, df_v4, dataset, mode, metric="acc"):
    """绘制对比图"""
    plt.figure(figsize=(10, 6))
    
    iters = len(df_v3)
    x = np.arange(1, iters + 1)
    
    plt.plot(x, df_v3[metric], 'b-', label=f'TensorFlow v3', linewidth=2)
    plt.plot(x, df_v4[metric], 'r--', label=f'PyTorch v4', linewidth=2)
    
    plt.xlabel('迭代轮次')
    plt.ylabel(metric.upper())
    plt.title(f'{dataset.upper()} {mode.upper()} - {metric.upper()} 对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = f"figures/compare_v3_v4_{dataset}_{mode}_{metric}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📈 对比图已保存: {filename}")
    plt.close()

def main():
    print("🔍 开始FLQ v3 vs v4 对比测试")
    
    # 测试配置
    tests = [
        {"dataset": "mnist", "mode": "bbit"},
        {"dataset": "mnist", "mode": "bin"},
        {"dataset": "fmnist", "mode": "laq8"},
    ]
    
    results = []
    
    for test in tests:
        dataset = test["dataset"]
        mode = test["mode"]
        
        print(f"\n{'='*60}")
        print(f"测试: {dataset.upper()} + {mode.upper()}")
        print('='*60)
        
        # 运行v3和v4
        df_v3 = run_experiment("v3", dataset, mode, iters=30)
        df_v4 = run_experiment("v4", dataset, mode, iters=30)
        
        if df_v3 is not None and df_v4 is not None:
            # 对比准确率
            acc_consistent = compare_results(df_v3, df_v4, "acc")
            
            # 对比损失
            print()
            entropy_consistent = compare_results(df_v3, df_v4, "entropy")
            
            # 绘制对比图
            plot_comparison(df_v3, df_v4, dataset, mode, "acc")
            plot_comparison(df_v3, df_v4, dataset, mode, "entropy")
            
            results.append({
                "test": f"{dataset}_{mode}",
                "acc_consistent": acc_consistent,
                "entropy_consistent": entropy_consistent,
                "overall": acc_consistent and entropy_consistent
            })
        else:
            results.append({
                "test": f"{dataset}_{mode}",
                "acc_consistent": False,
                "entropy_consistent": False,
                "overall": False
            })
    
    # 汇总结果
    print(f"\n{'='*60}")
    print("对比测试汇总")
    print('='*60)
    
    passed = 0
    for result in results:
        status = "✅ 通过" if result["overall"] else "❌ 不一致"
        print(f"{result['test']}: {status}")
        if result["overall"]:
            passed += 1
    
    print(f"\n总体结果: {passed}/{len(results)} 通过")
    
    if passed == len(results):
        print("🎉 v4版本与v3版本结果一致！PyTorch转换成功")
        return 0
    else:
        print("⚠️  存在不一致，需要进一步调试")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
