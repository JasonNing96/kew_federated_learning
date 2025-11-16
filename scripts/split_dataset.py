"""
数据集IID分片脚本 - 用于联邦学习
将oil-detection-2-2数据集均分给N个客户端
"""
import os
import shutil
import random
from pathlib import Path
import yaml

# 配置
SOURCE_DIR = "oil-detection-2-2"  # 使用相对路径，指向当前目录的数据集
NUM_CLIENTS = 3
SEED = 42

def split_dataset(source_dir, num_clients, seed=42):
    """
    IID分片策略：训练集随机均分，验证集每个客户端保留完整副本

    Args:
        source_dir: 源数据集路径
        num_clients: 客户端数量
        seed: 随机种子
    """
    random.seed(seed)
    source_path = Path(source_dir)

    # 获取训练集图片列表
    train_images = list((source_path / "train" / "images").glob("*.jpg"))
    print(f"📊 总训练图片数: {len(train_images)}")

    # 随机打乱
    random.shuffle(train_images)

    # 计算每个客户端的数据量
    images_per_client = len(train_images) // num_clients
    print(f"📦 每个客户端分配: {images_per_client} 张图片")

    # 分配数据到各客户端
    for client_id in range(1, num_clients + 1):
        print(f"\n🔧 处理 Client {client_id}...")

        client_dir = Path(f"client{client_id}")
        dataset_dir = client_dir / "dataset"

        # 创建目录结构
        for split in ["train", "val"]:
            (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        # 分配训练集
        start_idx = (client_id - 1) * images_per_client
        end_idx = start_idx + images_per_client if client_id < num_clients else len(train_images)
        client_images = train_images[start_idx:end_idx]

        print(f"  ✓ 分配训练集: {len(client_images)} 张")

        for img_path in client_images:
            # 复制图片
            dst_img = dataset_dir / "train" / "images" / img_path.name
            shutil.copy2(img_path, dst_img)

            # 复制对应的标签文件
            label_name = img_path.stem + ".txt"
            src_label = source_path / "train" / "labels" / label_name
            if src_label.exists():
                dst_label = dataset_dir / "train" / "labels" / label_name
                shutil.copy2(src_label, dst_label)

        # 复制完整验证集（所有客户端共享）
        val_images = list((source_path / "valid" / "images").glob("*.jpg"))
        print(f"  ✓ 复制验证集: {len(val_images)} 张（共享）")

        for img_path in val_images:
            # 复制图片
            dst_img = dataset_dir / "val" / "images" / img_path.name
            shutil.copy2(img_path, dst_img)

            # 复制标签
            label_name = img_path.stem + ".txt"
            src_label = source_path / "valid" / "labels" / label_name
            if src_label.exists():
                dst_label = dataset_dir / "val" / "labels" / label_name
                shutil.copy2(src_label, dst_label)

        # 生成oil.yaml配置文件
        yaml_content = {
            "path": str(dataset_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "names": {
                0: "no-oil",
                1: "oil"
            },
            "nc": 2
        }

        yaml_path = client_dir / "oil.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

        print(f"  ✓ 生成配置: {yaml_path}")

    print("\n" + "="*50)
    print("✅ 数据分片完成！")
    print("="*50)

    # 统计信息
    print("\n📈 分片统计:")
    for client_id in range(1, num_clients + 1):
        client_dir = Path(f"client{client_id}/dataset")
        train_count = len(list((client_dir / "train" / "images").glob("*.jpg")))
        val_count = len(list((client_dir / "val" / "images").glob("*.jpg")))
        print(f"  Client {client_id}: 训练={train_count}, 验证={val_count}")

if __name__ == "__main__":
    print("🚀 开始IID数据分片...")
    print(f"源数据集: {SOURCE_DIR}")
    print(f"客户端数: {NUM_CLIENTS}")
    print(f"随机种子: {SEED}")
    print("="*50)

    split_dataset(SOURCE_DIR, NUM_CLIENTS, SEED)

