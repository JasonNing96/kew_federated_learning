#!/usr/bin/env python3
"""
简单测试脚本 - 验证3个客户端是否真的并行运行
"""
import subprocess
import time
import sys
from pathlib import Path

def main():
    print("="*70)
    print("🔬 测试并行客户端启动")
    print("="*70)

    # 1. 在后台启动服务器
    print("\n[1/3] 启动服务器...")
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "app.server", "--config", "configs/test_local.yaml"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    print("⏳ 等待服务器启动...")
    time.sleep(10)
    print("✅ 服务器应该已启动\n")

    # 2. 并行启动3个客户端
    print("[2/3] 并行启动3个客户端...")
    client_procs = []

    for i in range(1, 4):
        print(f"  🚀 启动客户端 #{i}")
        proc = subprocess.Popen(
            [sys.executable, "-m", "app.client", str(i),
             "--server", "http://0.0.0.0:8087",
             "--config", "configs/test_local.yaml"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        client_procs.append((i, proc))
        time.sleep(1)

    print(f"\n✅ 所有3个客户端已启动")
    print("⏳ 等待客户端完成...\n")

    # 3. 等待所有客户端完成
    for i, proc in client_procs:
        returncode = proc.wait(timeout=120)
        print(f"✅ 客户端 #{i} 完成 (返回码: {returncode})")

    # 4. 停止服务器
    print("\n[3/3] 停止服务器...")
    server_proc.terminate()
    server_proc.wait(timeout=5)

    print("\n" + "="*70)
    print("🎉 测试完成！")
    print("="*70)

    # 检查结果
    checkpoint_dir = Path("outputs/server/checkpoints")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        print(f"\n📦 生成的checkpoints: {len(checkpoints)}")
        for cp in checkpoints:
            print(f"   - {cp.name}")
    else:
        print("\n⚠️  没有找到checkpoints目录")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  收到中断信号")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
