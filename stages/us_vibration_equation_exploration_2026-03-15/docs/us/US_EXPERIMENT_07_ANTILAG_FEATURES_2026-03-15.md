# 试验 07：抗滞后特征组 i_d_*（2026-03-15）

## 本次尝试目标
在不改变主方程的前提下，新增“变化率特征”以提升拐点响应速度：
- 新增 `i_d_* = feature_t - feature_{t-1}`（对每个 `g_*/c_*` 基础特征构造）
- 仍使用主版本估计配置（`y_next + nonlinear_absv + ridge=0.05`）

输出：
- `scripts/us/stage_2026_03_15/build_us_phase_antilag_panel.py`
- `data/us/experiments/step7_phase_antilag_features/us_regression_panel_antilag.csv`
- `data/us/experiments/step7_phase_antilag_features/us_ols_estimation_antilag.json`
- `data/us/experiments/step7_phase_antilag_features/plot_test/fit_metrics.json`
- `data/us/experiments/step7_phase_antilag_features/phase_test/phase_bias_summary.json`

## 结果与主版本对比
整体 test（one-step `x`）：
- `x_r2`: `0.740512 vs 0.742587`（`Δ=-0.002075`）
- `x_rmse`: `0.034290 vs 0.034153`（`Δ=+0.000137`）
- `direction_accuracy`: `0.739130 vs 0.739130`（持平）

相位与拐点：
- `best_lag`: `-1`（未改善）
- lag0 拐点指标与主版本基本一致：
- `recall=0.6471`
- `precision=0.6111`
- `mean_delay=0.4545`

数值稳定性：
- 训练条件数从主版本约 `739.9` 激增到 `5.79e16`，共线性显著恶化。

## 最终结论
1. 直接叠加 `i_d_*` 未改善相位偏差，`best_lag` 仍为 `-1`。
2. 主指标略退化，同时共线性风险显著上升，不适合替换主版本。
3. 该方向若要继续，应先做特征降维/正交化（否则只是放大共线性和噪声）。


