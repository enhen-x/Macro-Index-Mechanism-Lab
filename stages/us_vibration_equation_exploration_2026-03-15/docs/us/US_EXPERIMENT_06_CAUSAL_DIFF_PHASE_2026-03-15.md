# 试验 06：因果差分面板对相位的影响（2026-03-15）

## 本次尝试目标
验证将面板差分从 `central` 改为 `causal` 后，是否能降低主版本的相位滞后。

实施：
1. 构造面板：`build_us_regression_panel.py --diff-mode causal`
2. 同配置重训：`target_mode=y_next, damping=nonlinear_absv, ridge=0.05`
3. 同口径评估 test 与 phase 指标。

输出：
- `data/us/experiments/step6_causal_diff_phase/us_regression_panel_causal_diff.csv`
- `data/us/experiments/step6_causal_diff_phase/us_ols_estimation_causal_diff.json`
- `data/us/experiments/step6_causal_diff_phase/plot_test/fit_metrics.json`
- `data/us/experiments/step6_causal_diff_phase/phase_test/phase_bias_summary.json`

## 结果与主版本对比
核心对比（one-step `x` 与 phase）：
- `x_r2`：`0.742587`（与主版本相同）
- `x_rmse`：`0.034153`（与主版本相同）
- `best_lag`：`-1`（与主版本相同）
- 拐点指标（lag0）也与主版本一致（`recall=0.6471`, `precision=0.6111`）。

参数层面：
- `c/c_nl/k/β` 与主版本逐项一致（差值为 0）。

现象说明：
- `plot_test` 里的 `a` 拟合指标明显变化（甚至变差），是因为 `a` 列的定义变了；
- 但当前主模型训练目标是 `y_next`，本质由 `y_{t-1}, y_t, y_{t+1}, u_t` 决定，不直接使用面板中的 `a/v` 作为训练目标。

## 最终结论
1. 仅切换 `diff_mode=causal` 对当前 `y_next` 主模型的相位与 `x` 预测没有实质影响。
2. 因果差分不是当前一期相位偏差的主因，不建议作为主升级方向。
3. 下一步应从特征相位与结构层面入手（抗滞后特征/结构性改造），而不是只改 `a/v` 离散化口径。


