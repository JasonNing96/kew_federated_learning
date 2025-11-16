"""
FLQ-Fed 单节点训练脚本
在本地同时运行 server 和 client，适合快速测试
"""
import threading
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_server_thread(config_path: str = None):
    """在后台线程运行服务器"""
    from app.server import start_server
    try:
        start_server(config_path)
    except KeyboardInterrupt:
        pass


def run_client_sequential(client_id: int, server_url: str, config_path: str = None):
    """顺序运行客户端"""
    from app.client import start_client
    try:
        start_client(client_id, server_url, config_path)
    except KeyboardInterrupt:
        pass


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FLQ-Fed 单节点训练（快速测试）")
    parser.add_argument("--config", type=str, default="configs/flq_config.yaml", help="配置文件路径")
    parser.add_argument("--clients", type=int, default=1, help="客户端数量")
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 FLQ-Fed 单节点训练")
    print("="*70)
    
    # 加载配置
    from app.config import Config
    config = Config(args.config)
    
    print(f"\n📊 配置: {config}")
    print(f"👥 客户端数: {args.clients}")
    print(f"🔄 训练轮数: {config.rounds}")
    print(f"📦 本地 Epoch: {config.local_epochs}")
    print("="*70 + "\n")
    
    server_url = f"http://{config.server_host}:{config.server_port}"
    
    try:
        # 1. 在后台线程启动服务器
        print("[1/2] 启动服务器（后台线程）...")
        server_thread = threading.Thread(
            target=run_server_thread,
            args=(args.config,),
            daemon=True
        )
        server_thread.start()
        
        # 等待服务器就绪
        print("⏳ 等待服务器启动...")
        time.sleep(10)
        print("✅ 服务器已就绪\n")
        
        # 2. 顺序启动客户端（一个接一个，避免资源竞争）
        print(f"[2/2] 顺序启动 {args.clients} 个客户端...\n")
        
        for i in range(1, args.clients + 1):
            print(f"\n{'='*70}")
            print(f"🏃 运行客户端 #{i}")
            print(f"{'='*70}\n")
            
            run_client_sequential(i, server_url, args.config)
            
            print(f"\n✅ 客户端 #{i} 完成")
            
            # 如果不是最后一个客户端，等待一下
            if i < args.clients:
                print("⏳ 等待 3 秒后启动下一个客户端...\n")
                time.sleep(3)
        
        print("\n" + "="*70)
        print("🎉 所有客户端训练完成！")
        print("="*70)
        print("\n📁 查看结果:")
        print("   - 全局模型: outputs/server/checkpoints/")
        print("   - 客户端结果: outputs/client*/runs/")
        print("="*70)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  收到中断信号，停止训练...")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

