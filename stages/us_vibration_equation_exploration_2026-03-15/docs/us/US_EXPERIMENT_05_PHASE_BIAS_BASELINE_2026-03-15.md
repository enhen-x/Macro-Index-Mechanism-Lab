# 试验 05：主版本相位偏差量化（2026-03-15）

## 本次尝试目标
针对当前主版本（`nonlinear_absv`）先做“纯诊断”：
1. 量化 `Y(t+1|t)` 预测的相位偏差（`best_lag`）。
2. 增加拐点指标（`turn_hit_rate/precision/F1/delay`）。

输出：
- `data/us/experiments/step5_phase_bias_baseline/phase_bias_summary.json`
- `data/us/experiments/step5_phase_bias_baseline/lag_scan.csv`
- `data/us/experiments/step5_phase_bias_baseline/phase_lag_scan.png`
- `data/us/experiments/step5_phase_bias_baseline/turning_points_overlay.png`

## 结果与主版本对比
主版本 lag0（同口径 one-step `x`）：
- `x_r2=0.742587`
- `x_rmse=0.034153`
- `x_corr=0.873779`

相位扫描（`lag=-3..3`）结果：
- `best_lag_by_corr = -1`
- `best_lag_corr = 0.964269`（lag0 为 `0.873779`）
- `best_lag_rmse = 0.017938`（lag0 为 `0.034153`）

拐点指标（lag0）：
- `n_turn_true=17`
- `n_turn_pred=18`
- `turn_hit_rate_recall=0.6471`
- `turn_precision=0.6111`
- `turn_f1=0.6286`
- `mean_delay=0.4545`（正值表示预测拐点偏晚）

解释：
- `best_lag=-1` 表示把预测序列向前平移 1 期后显著对齐，等价于当前预测存在约 1 期滞后。

## 最终结论
1. 主版本确实存在稳定的一期相位偏差（滞后）。
2. 拐点命中率中等，且平均有正延迟，说明相位问题在拐点场景更明显。
3. 该实验是诊断实验，不替换主版本；下一步进入“因果离散化与相位一致性”验证。


