# 试验 12：可学习惯性系数 ρ（2026-03-15）

## 本次尝试目标
针对“模型对 `v(t)` 依赖过强、惯性过大”这一核心假设，直接改模型结构：

\[
y_{t+1}=y_t+\rho (y_t-y_{t-1})+dt^2\cdot F_t
\]

其中 `ρ` 不再固定为 1，而是扫描得到。  
实验流程：
1. 在 `train` 上拟合每个 `ρ` 对应的动力参数。
2. 在 `valid` 上按相位优先规则选 `ρ`（先看 `|best_lag|`，再看 `x_rmse/x_corr`）。
3. 在 `test` 上做最终对比。

输出：
- `scripts/us/stage_2026_03_15/experiment_us_learnable_inertia_rho.py`
- `data/us/experiments/step12_learnable_inertia_rho/learnable_rho_summary.json`
- `data/us/experiments/step12_learnable_inertia_rho/learnable_rho_table.csv`

## 结果与主版本对比
扫描范围：
- `ρ in [0.0, 1.2], step=0.05`（共 25 个候选）

关键结果 A（按 selection 规则选出）：
- `ρ=0.0`
- test：`x_r2=0.723864`（`Δ=-0.018723`），`x_rmse=0.035373`（`Δ=+0.001220`）
- `best_lag=-1`（无改善）
- 但拐点指标改善：`recall=0.8235`、`mean_delay=0.1429`（主版本 `0.4545`）

关键结果 B（按 eval 综合最优）：
- `ρ=0.75`
- test：`x_r2=0.743590`（`Δ=+0.001003`），`x_rmse=0.034086`（`Δ=-0.000067`）
- `best_lag=-1`（无改善）
- 拐点指标与主版本基本一致。

## 最终结论
1. “降低惯性权重”可以改变拐点行为，但并未把相位主问题从 `-1` 拉回 `0`。
2. 在主指标维度，`ρ=0.75` 仅有极小幅度提升，收益不足以证明结构已根治相位偏差。
3. 结论：本轮不替换主版本；可把 `ρ` 作为可选调节参数保留，但它不是当前相位问题的决定性解法。


