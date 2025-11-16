# 预训练模型目录

## 模型列表

- `yolov8n.pt` - YOLOv8 Nano（默认使用）
- `yolov8s.pt` - YOLOv8 Small
- `yolo11n.pt` - YOLOv11 Nano

## 使用方法

在 `configs/flq_config.yaml` 中配置：

```yaml
model:
  name: "models/yolov8n.pt"  # 固定路径
```

## 下载模型

如果模型不存在，可以从 Ultralytics 官方下载：

```bash
# 在项目根目录运行
cd models
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
```

或者在代码中自动下载（首次运行时）。
