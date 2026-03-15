# 试验 03：Walk-Forward 重估（2026-03-15）

## 本次尝试目标
验证在当前主版本（`nonlinear_absv`）基础上，是否通过滚动重估参数（walk-forward）可以提升时序泛化能力，重点比较：
1. 动态重估（每个窗口重估参数）与静态主版本参数（固定参数）在同一窗口上的表现差异。
2. 动态重估的平均表现与主版本整段 test 指标之间的差距。

主版本基线：
- `data/us/plot/fit_metrics.json`
- `x_r2=0.742587`，`x_rmse=0.034153`，`direction_accuracy=0.739130`

实验输出：
- `data/us/experiments/step3_walk_forward/walk_forward_summary.json`
- `data/us/experiments/step3_walk_forward/walk_forward_table.csv`

## 结果与主版本对比
walk-forward 配置：
- `min_train=120`
- `horizon=12`
- `step=6`
- 共 `31` 个窗口

窗口内平均结果（dynamic vs static）：
- dynamic：`mean_x_r2=0.027896`，`mean_x_rmse=0.031320`，`mean_x_direction=0.559677`
- static：`mean_x_r2=0.069859`，`mean_x_rmse=0.030677`，`mean_x_direction=0.537634`
- 差值（dynamic-static）：
- `Δx_r2=-0.041963`
- `Δx_rmse=+0.000643`
- `Δx_direction=+0.022043`

dynamic 相对主版本整段 test：
- `Δmean_x_r2_vs_main=-0.714691`
- `Δmean_x_rmse_vs_main=-0.002832`

解释：
- 动态重估在“方向判定”上有小幅提升，但在平均 `x_r2` 和 `x_rmse` 上劣于同窗口静态参数。
- 与主版本整段 test 指标不可直接同口径比较（窗口更短、方差结构不同），但动态方案未显示出稳定优势。

## 最终结论
1. 当前设置下，walk-forward 重估没有形成可替代主版本的稳定增益。
2. 方向准确率有边际改善，但不足以抵消 `x_r2` 与 `x_rmse` 的退化。
3. 结论：不替换主版本，继续保留当前静态主版本参数方案。

