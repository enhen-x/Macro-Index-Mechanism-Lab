# 试验 15：显式相位状态（Phase-State）建模 2026-03-15

## 本次尝试目标
在主模型一步预测 `y_hat_raw(t+1|t)` 之上，引入显式相位状态 `z_t`，不再仅依赖静态系数去“隐含吸收”延迟：

- 状态更新：`z_t = rho * z_(t-1) + kappa * (y_hat_raw_t - y_t) + eta * (y_t - y_(t-1))`
- 输出修正：`y_hat_state_t = y_hat_raw_t + z_t`

其中 `rho` 显式表示相位记忆/延迟惯性；`kappa, eta` 控制对当前预测增量与惯性增量的响应。  
采用网格扫描（`rho/kappa/eta`）在 `valid` 选型、`test` 评估。

输出：
- `scripts/us/stage_2026_03_15/experiment_us_explicit_phase_state.py`
- `data/us/experiments/step15_explicit_phase_state/phase_state_summary.json`
- `data/us/experiments/step15_explicit_phase_state/phase_state_grid_scan.csv`

## 结果与主版本对比
主版本（test）：
- `x_r2=0.742587`
- `x_rmse=0.034153`
- `best_lag=-1`
- `turn_recall=0.6471`
- `turn_precision=0.6111`
- `turn_mean_delay=0.4545`

step15 最优参数（按 valid 选型）：
- `rho=0.55, kappa=-0.6, eta=-0.1`
- `valid: sel_best_lag=-1, sel_x_r2=0.736138`

step15 在 test：
- `x_r2: 0.742587 -> 0.685207`（`Δ=-0.057381`）
- `x_rmse: 0.034153 -> 0.037768`（`Δ=+0.003615`）
- `best_lag: -1 -> -1`（未改变）
- `turn_recall: 0.6471 -> 0.5882`
- `turn_precision: 0.6111 -> 0.6250`
- `turn_mean_delay: 0.4545 -> 0.4000`

补充观察（扫描分布）：
- 总扫描组合 4420 组；
- `sel_best_lag` 仅出现 `-1`（4387 组）和 `-2`（33 组），没有出现 `0`。

## 最终结论
1. 将“延迟”显式化为相位状态参数（`rho/kappa/eta`）在当前框架下仍未解决核心相位偏差（`best_lag` 依旧为 `-1`）。  
2. 该方案带来一定拐点延迟改善（`mean_delay` 略降），但整体预测精度显著退化（`x_r2` 下滑、`x_rmse` 上升）。  
3. 结论：当前这个线性状态更新形式不够，下一步应尝试“状态+观测联合估计”（如 Kalman/EM 或切换到直接建模 `ΔY` 的误差状态方程），否则难以决定性消除相位偏差。  


