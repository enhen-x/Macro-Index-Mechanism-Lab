# 试验 11：Lead 补偿（valid 选参）2026-03-15

## 本次尝试目标
针对试验10“train 选参过拟合”问题，改为：
- 在 `valid` 上选择补偿参数 `(gamma, eta)`
- 固定后在 `test` 上评估

目标是验证是否能在不破坏主指标的前提下改善相位。

输出：
- `data/us/experiments/step11_phase_lead_comp_valid_select/lead_comp_summary.json`
- `data/us/experiments/step11_phase_lead_comp_valid_select/grid_scan.csv`
- `data/us/experiments/step11_phase_lead_comp_valid_select/test_predictions.csv`

## 结果与主版本对比
选参结果（valid）：
- `gamma=0.0`, `eta=-0.1`
- `train_best_lag=-1`（valid 上也未出现相位归零）

test 对比主版本：
- `x_r2: 0.742587 -> 0.751490`（`Δ=+0.008902`）
- `x_rmse: 0.034153 -> 0.033557`（`Δ=-0.000596`）
- `x_corr: 0.873779 -> 0.876682`
- `best_lag: -1 -> -1`（无改善）

拐点指标（test）：
- recall/precision/mean_delay 与主版本基本相同。

## 最终结论
1. valid 选参可以抑制试验10的过拟合，并带来小幅精度增益。
2. 但相位核心问题仍未解决（`best_lag` 固定为 `-1`）。
3. 该方案最多作为“轻量精度增强”，不能作为相位修复方案替换主版本。


