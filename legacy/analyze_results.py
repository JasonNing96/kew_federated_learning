#!/usr/bin/env python3
"""
FLQ-Federated MNIST 实验结果分析脚本
解析worker日志，绘制精度-轮次和通信比特-轮次曲线
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def parse_worker_logs(log_pattern="worker_*.log"):
    """解析worker日志文件，提取训练指标"""
    data = []
    
    # 查找所有匹配的日志文件
    log_files = glob.glob(log_pattern)
    if not log_files:
        print(f"未找到日志文件: {log_pattern}")
        return pd.DataFrame()
    
    for log_file in log_files:
        print(f"解析日志文件: {log_file}")
        
        with open(log_file, 'r') as f:
            for line in f:
                # 匹配日志格式: [worker_id] round=X, loss=Y, acc=Z, logical_bits=W, flq_mode=MODE
                match = re.search(
                    r'\[([^\]]+)\] round=(\d+), loss=([\d.]+), acc=([\d.]+), logical_bits=(\d+), flq_mode=(\w+)',
                    line
                )
                
                if match:
                    worker_id, round_num, loss, acc, logical_bits, flq_mode = match.groups()
                    data.append({
                        'worker_id': worker_id,
                        'round': int(round_num),
                        'loss': float(loss),
                        'accuracy': float(acc),
                        'logical_bits': int(logical_bits),
                        'flq_mode': flq_mode,
                        'log_file': log_file
                    })
    
    return pd.DataFrame(data)

def aggregate_round_metrics(df):
    """按轮次聚合多个worker的指标"""
    if df.empty:
        return pd.DataFrame()
    
    # 按轮次和FLQ模式分组，计算平均值
    aggregated = df.groupby(['round', 'flq_mode']).agg({
        'loss': 'mean',
        'accuracy': 'mean',
        'logical_bits': 'sum'  # 总通信比特数
    }).reset_index()
    
    # 计算累计通信比特数
    for mode in aggregated['flq_mode'].unique():
        mode_mask = aggregated['flq_mode'] == mode
        aggregated.loc[mode_mask, 'cumulative_bits'] = aggregated.loc[mode_mask, 'logical_bits'].cumsum()
    
    return aggregated

def plot_results(aggregated_df, output_dir="plots"):
    """绘制实验结果图表"""
    if aggregated_df.empty:
        print("没有数据可绘制")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文字体和图表样式
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 精度-轮次曲线
    plt.figure(figsize=(10, 6))
    for mode in aggregated_df['flq_mode'].unique():
        mode_data = aggregated_df[aggregated_df['flq_mode'] == mode]
        plt.plot(mode_data['round'], mode_data['accuracy'], 
                marker='o', label=f'FLQ_{mode}', linewidth=2)
    
    plt.xlabel('通信轮次')
    plt.ylabel('测试准确率')
    plt.title('FLQ-Federated MNIST: 精度-轮次曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_vs_round.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 累计通信比特数-轮次曲线
    plt.figure(figsize=(10, 6))
    for mode in aggregated_df['flq_mode'].unique():
        mode_data = aggregated_df[aggregated_df['flq_mode'] == mode]
        # 转换为MB单位
        cumulative_mb = mode_data['cumulative_bits'] / (8 * 1024 * 1024)
        plt.plot(mode_data['round'], cumulative_mb, 
                marker='s', label=f'FLQ_{mode}', linewidth=2)
    
    plt.xlabel('通信轮次')
    plt.ylabel('累计通信量 (MB)')
    plt.title('FLQ-Federated MNIST: 通信开销-轮次曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 对数坐标，便于对比
    plt.tight_layout()
    plt.savefig(f'{output_dir}/communication_vs_round.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 精度-通信比特效率图
    plt.figure(figsize=(10, 6))
    for mode in aggregated_df['flq_mode'].unique():
        mode_data = aggregated_df[aggregated_df['flq_mode'] == mode]
        cumulative_mb = mode_data['cumulative_bits'] / (8 * 1024 * 1024)
        plt.plot(cumulative_mb, mode_data['accuracy'], 
                marker='d', label=f'FLQ_{mode}', linewidth=2)
    
    plt.xlabel('累计通信量 (MB)')
    plt.ylabel('测试准确率')
    plt.title('FLQ-Federated MNIST: 精度-通信效率')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_vs_communication.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"图表已保存到 {output_dir}/ 目录")

def generate_summary_report(aggregated_df, output_file="experiment_summary.txt"):
    """生成实验总结报告"""
    if aggregated_df.empty:
        print("没有数据生成报告")
        return
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("FLQ-Federated MNIST 实验总结报告\n")
        f.write("=" * 50 + "\n\n")
        
        for mode in aggregated_df['flq_mode'].unique():
            mode_data = aggregated_df[aggregated_df['flq_mode'] == mode]
            
            final_accuracy = mode_data['accuracy'].iloc[-1]
            final_comm_mb = mode_data['cumulative_bits'].iloc[-1] / (8 * 1024 * 1024)
            avg_loss = mode_data['loss'].mean()
            
            compression_ratio = 32.0  # 基线是32位
            if mode == 'sign1':
                compression_ratio = 32.0  # 1位 vs 32位
            elif mode == 'int8':
                compression_ratio = 4.0   # 8位 vs 32位
            
            f.write(f"FLQ模式: {mode}\n")
            f.write(f"  最终准确率: {final_accuracy:.4f}\n")
            f.write(f"  最终通信量: {final_comm_mb:.2f} MB\n")
            f.write(f"  平均损失: {avg_loss:.4f}\n")
            f.write(f"  理论压缩比: {compression_ratio}:1\n")
            f.write(f"  轮次数: {len(mode_data)}\n\n")
        
        # 对比分析
        if len(aggregated_df['flq_mode'].unique()) > 1:
            f.write("对比分析:\n")
            f.write("-" * 30 + "\n")
            
            baseline = aggregated_df[aggregated_df['flq_mode'] == 'off']
            if not baseline.empty:
                baseline_acc = baseline['accuracy'].iloc[-1]
                baseline_comm = baseline['cumulative_bits'].iloc[-1] / (8 * 1024 * 1024)
                
                for mode in ['sign1', 'int8']:
                    mode_data = aggregated_df[aggregated_df['flq_mode'] == mode]
                    if not mode_data.empty:
                        mode_acc = mode_data['accuracy'].iloc[-1]
                        mode_comm = mode_data['cumulative_bits'].iloc[-1] / (8 * 1024 * 1024)
                        
                        acc_diff = mode_acc - baseline_acc
                        comm_reduction = (baseline_comm - mode_comm) / baseline_comm * 100
                        
                        f.write(f"{mode} vs 基线:\n")
                        f.write(f"  准确率变化: {acc_diff:+.4f}\n")
                        f.write(f"  通信量减少: {comm_reduction:.1f}%\n\n")
    
    print(f"实验总结报告已保存到: {output_file}")

def main():
    """主函数"""
    print("FLQ-Federated MNIST 实验结果分析")
    print("=" * 40)
    
    # 解析日志
    print("正在解析worker日志...")
    df = parse_worker_logs("*.log")
    
    if df.empty:
        print("未找到worker日志文件，请确保日志文件存在且格式正确")
        print("预期格式: [worker_id] round=X, loss=Y, acc=Z, logical_bits=W, flq_mode=MODE")
        return
    
    print(f"解析到 {len(df)} 条训练记录")
    print(f"FLQ模式: {df['flq_mode'].unique()}")
    
    # 按轮次聚合
    print("正在聚合轮次指标...")
    aggregated = aggregate_round_metrics(df)
    
    if aggregated.empty:
        print("聚合数据为空")
        return
    
    print(f"聚合后数据: {len(aggregated)} 轮次")
    
    # 绘制图表
    print("正在生成图表...")
    plot_results(aggregated)
    
    # 生成报告
    print("正在生成实验报告...")
    generate_summary_report(aggregated)
    
    print("\n分析完成！")
    print("输出文件:")
    print("  - plots/accuracy_vs_round.png")
    print("  - plots/communication_vs_round.png")
    print("  - plots/accuracy_vs_communication.png")
    print("  - experiment_summary.txt")

if __name__ == "__main__":
    main()