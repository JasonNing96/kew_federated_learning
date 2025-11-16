## 修改记录

- **2025-11-16：更新 `fed_flq_local.py` 的监控逻辑**
  - 文件：`app/fed_flq_local.py`
  - 函数：`monitor_status_to_csv`
  - 变更：
    - 结束条件兼容 `is_finished` / `finished` 以及当前服务器使用的 `training_done` 字段。
    - 轮次上限判断由仅使用 `max_rounds` 改为优先 `max_rounds`，否则回退到 `total_rounds`，适配 `server.py` 的 `/status` 返回结构。
  - 目的：
    - 确保本地并行 Fed-FLQ 启动脚本在当前服务器实现下能正确感知训练结束，监控线程可以自动退出，并保证 CSV 记录轮次完整。


