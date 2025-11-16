# 数据集目录

## 目录结构

```
data/
├── oil.yaml           # 数据集主配置（模板）
├── client1/           # 客户端1数据分片
│   ├── oil.yaml       # 客户端配置
│   └── dataset/       # 实际数据
├── client2/           # 客户端2数据分片
└── client3/           # 客户端3数据分片
```

## 数据切分

使用 `scripts/split_dataset.py` 进行数据切分：

```bash
python scripts/split_dataset.py --input /path/to/original/dataset --clients 3
```

## 路径配置

所有路径使用项目根目录的相对路径：

```yaml
# data/client1/oil.yaml
path: data/client1/dataset  # 相对于项目根目录
train: images/train
val: images/val
```
