# 试验 08：ARX 残差校正层（2026-03-15）

## 本次尝试目标
在不修改主动力方程参数（`c/c_nl/k/β`）的前提下，增加一个因果 ARX 残差层，尝试优先缓解相位偏差，并同时观察整体 one-step 指标变化。

校正形式（训练于 train，应用于 test）：
- `e_t = y_t - y_hat_raw_t`
- `corr_t = alpha + phi * e_{t-1} + theta * Δy_{t-1}`
- `y_hat_corr_t = y_hat_raw_t + corr_t`

输出：
- `scripts/us/stage_2026_03_15/experiment_us_phase_arx_correction.py`
- `data/us/experiments/step8_phase_arx_correction/arx_phase_summary.json`
- `data/us/experiments/step8_phase_arx_correction/test_predictions.csv`
- `data/us/experiments/step8_phase_arx_correction/lag_scan_raw.csv`
- `data/us/experiments/step8_phase_arx_correction/lag_scan_corr.csv`

## 结果与主版本对比
主版本（raw）：
- `x_r2=0.742587`
- `x_rmse=0.034153`
- `best_lag=-1`
- `turn_recall=0.6471`, `turn_precision=0.6111`, `mean_delay=0.4545`

ARX 校正后（corr）：
- `x_r2=0.753070`（`Δ=+0.010483`）
- `x_rmse=0.033450`（`Δ=-0.000703`）
- `x_corr=0.876529`（较 raw 提升）
- `best_lag=-1`（未改善）
- 拐点指标与 raw 基本不变：
- `turn_recall=0.6471`
- `turn_precision=0.6111`
- `mean_delay=0.4545`

ARX 系数（train）：
- `alpha=-4.11e-05`
- `phi=-0.1206`
- `theta=0.00962`
- `train_r2=0.0124`（残差层解释度较弱）

## 最终结论
1. ARX 残差层可以小幅提升整体 one-step 精度（`x_r2`、`x_rmse`），说明对幅度误差有修正价值。
2. 但相位核心问题未被解决（`best_lag` 仍 `-1`，拐点延迟指标无改善）。
3. 结论：该方案可作为“精度增强层”保留，但不能作为“相位修复”的主解法；下一步仍需做结构性去滞后改造。


