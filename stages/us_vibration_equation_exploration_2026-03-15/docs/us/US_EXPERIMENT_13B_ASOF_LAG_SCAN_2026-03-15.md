# 试验 13b：As-of 发布滞后配置扫描（2026-03-15）

## 本次尝试目标
避免单一 lag 假设偏差，扫描多组发布滞后配置，检查是否存在可把 `best_lag` 直接推到 `0` 的配置。

候选配置（5 组）：
- `all_0`
- `macro1_ff0`
- `macro1_ff1`
- `core1_slow0_ff0`
- `macro2_ff1`

输出：
- `scripts/us/stage_2026_03_15/scan_us_asof_release_lag_configs.py`
- `data/us/experiments/step13b_asof_lag_scan/asof_lag_scan_summary.json`
- `data/us/experiments/step13b_asof_lag_scan/asof_lag_scan_table.csv`

## 结果与主版本对比
核心发现：
- 5 组配置全部仍为 `best_lag=-1`，无一达到 `0`。

最优相位候选（按延迟优先）：
- `macro2_ff1`
- `turn_mean_delay=0.142857`（比主版本 `0.454545` 更好）
- 但 `x_r2=0.715956`（`Δ=-0.026631`），`x_rmse=0.034655`（`Δ=+0.000503`）

最优整体候选：
- `all_0`（即现有主版本口径）
- `x_r2=0.742587`, `x_rmse=0.034153`
- 但 `best_lag` 仍是 `-1`

## 最终结论
1. 在本轮可行 lag 假设范围内，as-of 发布滞后调整无法决定性消除相位滞后（`best_lag` 全部为 `-1`）。
2. 某些配置可改善拐点延迟，但会明显伤害整体拟合。
3. 结论：下一步必须进入更强结构改造（例如显式“潜在相位状态”/状态空间模型），仅靠发布时间滞后映射不足以根治问题。


