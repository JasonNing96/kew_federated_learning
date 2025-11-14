"""
联邦学习客户端 - 本地训练与模型更新
工作流程：
1. 从服务器拉取全局模型
2. 在本地数据分片上训练EPR轮
3. 上传更新后的权重到服务器
4. 循环直到所有轮次完成
"""
import requests
import torch
import io
import os
import glob
import yaml
import time
from ultralytics import YOLO
from datetime import datetime
import sys

# ============= 配置参数 =============
# 服务器地址（支持内网IP切换）
SERVER = os.getenv("FL_SERVER", "http://127.0.0.1:8080")

# 客户端ID（从命令行参数或环境变量获取）
CLIENT_ID = os.getenv("CLIENT_ID", sys.argv[1] if len(sys.argv) > 1 else "1")

# 训练参数
EPR = 1          # 快速测试用（CPU模式）
IMGZ = 640       # 图像尺寸
BATCH = 8       # 批次大小
DEVICE = 'cpu'   # 使用CPU避免指令集问题

# 数据配置
DATA_YAML = f"client{CLIENT_ID}/oil.yaml"
BASE_MODEL = "yolov8n.pt"

# 日志目录
RUNS_DIR = f"client{CLIENT_ID}/runs_fed"

print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 启动联邦学习客户端 #{CLIENT_ID}")
print(f"[{datetime.now().strftime('%H:%M:%S')}] 🌐 服务器地址: {SERVER}")
print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 数据配置: {DATA_YAML}")
print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️  训练参数: EPR={EPR}, IMGZ={IMGZ}, BATCH={BATCH}, DEVICE={DEVICE}")
print("=" * 60)


def num_train_images(yaml_path):
    """
    统计训练集图片数量
    Args:
        yaml_path: 数据配置文件路径
    Returns:
        图片数量
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    dataset_path = config["path"]
    train_subpath = config["train"]

    # 构建完整路径
    train_path = os.path.join(dataset_path, train_subpath)

    # 统计图片
    image_count = len(glob.glob(os.path.join(train_path, "**", "*.*"), recursive=True))
    return max(1, image_count)


def pull_global_model(model: YOLO):
    """
    从服务器拉取全局模型权重
    Args:
        model: YOLO模型实例
    Returns:
        (round_id, is_done): 当前轮次和是否完成标志
    """
    try:
        response = requests.get(f"{SERVER}/global", timeout=30)
        response.raise_for_status()

        # 加载state_dict
        state_dict = torch.load(io.BytesIO(response.content), map_location="cpu")
        model.model.load_state_dict(state_dict, strict=False)  # 改为strict=False，避免首次不匹配

        # 获取轮次信息
        round_id = int(response.headers.get("X-Round", "0"))
        is_done = response.headers.get("X-Done", "False") == "True"

        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📥 拉取全局模型成功 (Round {round_id})")
        return round_id, is_done

    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 拉取模型失败: {e}")
        raise


def push_local_update(model: YOLO, n_samples: int):
    """
    上传本地训练后的权重到服务器
    Args:
        model: 训练后的YOLO模型
        n_samples: 本地训练样本数
    Returns:
        服务器响应 (dict)
    """
    try:
        # 序列化state_dict
        state_dict = {k: v.cpu() for k, v in model.model.state_dict().items()}
        bio = io.BytesIO()
        torch.save(state_dict, bio)
        bio.seek(0)

        # 构建请求
        files = {"file": ("state.pt", bio.getvalue(), "application/octet-stream")}
        data = {"n": str(n_samples)}

        # 上传
        response = requests.post(f"{SERVER}/update", files=files, data=data, timeout=60)
        response.raise_for_status()

        result = response.json()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📤 上传本地更新成功 (样本数={n_samples})")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]    服务器状态: Round {result['round']}, Buffered={result.get('buffered', '?')}")

        return result

    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 上传更新失败: {e}")
        raise


def train_local(model: YOLO, round_id: int):
    """
    在本地数据上训练模型
    Args:
        model: YOLO模型实例
        round_id: 当前轮次
    """
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🎯 开始本地训练 Round {round_id}...")

    try:
        results = model.train(
            data=DATA_YAML,
            epochs=EPR,
            imgsz=IMGZ,
            batch=BATCH,
            device=DEVICE,
            project=RUNS_DIR,
            name=f"round_{round_id}",
            exist_ok=True,
            verbose=False  # 关闭详细输出，减少日志混乱
        )

        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 本地训练完成 (mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.3f})")
        return results

    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 训练失败: {e}")
        raise


def main():
    """主函数：联邦学习客户端主循环"""

    # 检查数据配置
    if not os.path.exists(DATA_YAML):
        print(f"❌ 数据配置文件不存在: {DATA_YAML}")
        print(f"💡 提示: 请先运行 split_dataset.py 进行数据分片")
        return

    # 统计本地样本数
    n_samples = num_train_images(DATA_YAML)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 本地训练样本数: {n_samples}\n")

    # 加载基础模型并初始化为正确的类别数
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📦 初始化模型架构...")
    model = YOLO(BASE_MODEL)

    # 读取类别数并重建模型
    import yaml
    with open(DATA_YAML, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    nc = data_config.get('nc', 80)

    from ultralytics.nn.tasks import DetectionModel
    model.model = DetectionModel(model.model.yaml, ch=3, nc=nc)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 模型架构初始化完成 (nc={nc})")

    # 记录上一轮ID（避免重复训练）
    last_round = -1

    # 主循环
    while True:
        try:
            # 1. 拉取全局模型
            current_round, is_done = pull_global_model(model)

            # 检查是否完成所有轮次
            if is_done:
                print(f"\n{'='*60}")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎉 所有联邦训练轮次已完成！")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 📁 本地训练结果: {RUNS_DIR}")
                print(f"{'='*60}")
                break

            # 检查是否已经训练过当前轮次
            if current_round == last_round:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ 等待服务器聚合...")
                time.sleep(5)  # 等待5秒后重试
                continue

            # 2. 本地训练
            train_local(model, current_round)

            # 3. 上传更新
            response = push_local_update(model, n_samples)

            # 更新轮次记录
            last_round = current_round

            # 检查服务器返回的done标志
            if response.get("done", False):
                print(f"\n{'='*60}")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎉 联邦训练完成（服务器通知）")
                print(f"{'='*60}")
                break

            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Round {current_round} 完成\n")

        except KeyboardInterrupt:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ⚠️  用户中断")
            break

        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 错误: {e}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 5秒后重试...")
            time.sleep(5)


if __name__ == "__main__":
    main()
