# 试验 16：可观测误差驱动的状态更新（Kalman-style）2026-03-15

## 本次尝试目标
在不使用未来信息的前提下，将“延迟/相位”建模为可递推状态，并用 `t` 时刻可观测误差更新状态后再预测 `t+1`：

- 预测：`z_pred_t = rho * z_(t-1)`
- 观测：`obs_t = y_t - y_hat_raw_(t-1)`（在 `t` 时刻已知）
- 更新：`z_t = z_pred_t + K * (obs_t - z_pred_t)`
- 修正预测：`y_hat_state_(t+1|t) = y_hat_raw_(t+1|t) + gamma * z_t`

扫描参数：`rho ∈ [0,0.95]`、`K ∈ [0,1]`、`gamma ∈ [0,1.5]`，在 `valid` 选型、`test` 评估。

输出：
- `scripts/us/stage_2026_03_15/experiment_us_error_state_kalman.py`
- `data/us/experiments/step16_error_state_kalman/error_state_summary.json`
- `data/us/experiments/step16_error_state_kalman/error_state_grid_scan.csv`

## 结果与主版本对比
主版本（test）：
- `x_r2=0.742587`
- `x_rmse=0.034153`
- `best_lag=-1`
- `turn_recall=0.6471`
- `turn_precision=0.6111`
- `turn_mean_delay=0.4545`

step16 最优参数（按 valid 选型）：
- `rho=0.95, K=0.1, gamma=0.4`
- `valid: sel_best_lag=-1, sel_x_r2=0.693431`

step16 在 test：
- `x_r2: 0.742587 -> 0.735068`（`Δ=-0.007519`）
- `x_rmse: 0.034153 -> 0.034648`（`Δ=+0.000495`）
- `best_lag: -1 -> -1`（未改变）
- `turn_recall: 0.6471 -> 0.7059`
- `turn_precision: 0.6111 -> 0.6000`
- `turn_mean_delay: 0.4545 -> 0.3333`

补充观察：
- 总扫描组合 3520 组，`sel_best_lag` 全部为 `-1`（无一达到 `0`）。

## 最终结论
1. 可观测误差状态更新在“拐点延迟”上有改善（`mean_delay` 下降），但对核心相位偏差没有实质改变（`best_lag` 仍固定为 `-1`）。  
2. 整体精度较主版本略差（`x_r2` 下降、`x_rmse` 上升），不适合作为主版本替换。  
3. 结论：即使把延迟做成显式动态状态，当前观测方程仍不足以把相位推回 0，下一步需要改“目标定义/状态方程形态”（例如直接建模 `ΔY` 或 `v` 的演化）。  


