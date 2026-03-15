# 试验 10：因果超前补偿层（2026-03-15）

## 本次尝试目标
在主模型输出层叠加“因果超前补偿”以尝试修复相位滞后：
- `y_hat_corr = y_hat_raw + gamma*(y_hat_raw - y_t) + eta*(y_t - y_{t-1})`
- 在 train 上网格搜索 `gamma/eta`，目标优先让 `train_best_lag` 靠近 `0`，再看测试泛化。

输出：
- `scripts/us/stage_2026_03_15/experiment_us_phase_lead_compensation.py`
- `data/us/experiments/step10_phase_lead_compensation/lead_comp_summary.json`
- `data/us/experiments/step10_phase_lead_compensation/grid_scan.csv`
- `data/us/experiments/step10_phase_lead_compensation/test_predictions.csv`

## 结果与主版本对比
选中的参数（train 最优）：
- `gamma=1.2`, `eta=0.2`
- train 上 `best_lag=0`

但 test 对比主版本：
- `x_r2: 0.742587 -> 0.498604`（`Δ=-0.243984`）
- `x_rmse: 0.034153 -> 0.047665`（`Δ=+0.013512`）
- `best_lag: -1 -> -1`（未改善）

拐点指标（test）：
- recall/precision/mean_delay 有小幅改善（例如 `mean_delay 0.4545 -> 0.3333`），
- 但代价是整体预测质量显著崩塌。

## 最终结论
1. 该补偿层在 train 上出现明显过拟合，泛化失败。
2. 相位核心指标在 test 上仍未修复（`best_lag` 依旧 `-1`）。
3. 结论：该方案不进入主版本；若继续此路线，必须改为“valid 选参 + 稳定性约束”后再评估。


