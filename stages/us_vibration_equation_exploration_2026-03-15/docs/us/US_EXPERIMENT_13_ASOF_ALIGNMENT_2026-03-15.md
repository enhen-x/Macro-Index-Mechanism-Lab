# 试验 13：As-of 时间对齐（基线滞后假设）2026-03-15

## 本次尝试目标
验证“信息发布时间错配”是否是相位滞后主因。  
采用一组保守 as-of 假设重建月度面板，再走完整训练评估：
- 市场变量（`sp500/ust10y/dxy_broad/vix`）滞后 `0`
- `fed_funds` 滞后 `0`
- `walcl/bus_loans/cpi/indpro/unrate` 滞后 `1`

流程：
1. 构建 `us_monthly_panel_asof.csv`
2. 构建回归面板并重训主模型配置（`y_next + nonlinear_absv + ridge=0.05`）
3. 在 test 做主指标 + phase 指标对比

输出：
- `scripts/us/stage_2026_03_15/build_us_asof_monthly_panel.py`
- `data/us/experiments/step13_asof_alignment/*`

## 结果与主版本对比
主指标（test）：
- `x_r2: 0.742587 -> 0.718804`（`Δ=-0.023783`）
- `x_rmse: 0.034153 -> 0.034971`（`Δ=+0.000818`）
- `direction_accuracy(a): 0.739130 -> 0.760870`

相位指标（test）：
- `best_lag: -1 -> -1`（未改变）
- `best_lag_corr: 0.964269 -> 0.967679`

拐点指标（lag0）：
- `turn_recall: 0.6471 -> 0.7059`
- `turn_precision: 0.6111 -> 0.5714`
- `mean_delay: 0.4545 -> 0.1667`（延迟明显下降）

## 最终结论
1. as-of 对齐对“拐点延迟”有明显帮助，但没有把相位主问题从 `best_lag=-1` 拉回 `0`。
2. 同时付出了整体 `x_r2/x_rmse` 的代价，不适合直接替换主版本。
3. 结论：as-of 对齐应作为必要约束保留，但不是单独可“决定性解决”相位问题的手段。


