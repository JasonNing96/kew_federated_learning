## 当前施工记录（2025-11-16）

- **已完成**
  - 调整 `fed_flq_local.py` 中 `monitor_status_to_csv` 的退出条件和轮次上限逻辑：
    - 兼容 `is_finished` / `finished` / `training_done` 三种结束标志。
    - 轮次上限支持 `max_rounds` 和 `total_rounds`（当前 `server.py` 使用 `total_rounds`）。

- **后续可选工作（待定）**
  - 在 `server.py` 的 `/status` 返回中加入训练指标（如 mAP、accuracy）以及通信比特字段（`bits_up` / `bits_down`），以便 `fed_flq_local.py` 的 `plot_flq` 能绘制更完整的收敛与通信曲线。
  - 将 `fed_flq_local.py` 中 CSV、图像输出路径统一改为基于项目根目录的路径（使用 `PROJECT_ROOT / "outputs/..."`），保证从任意工作目录启动脚本时输出位置一致。


