# 试验 01：Ridge + 共线性处理（2026-03-15）

## 本次尝试目标
在当前主版本（`nonlinear_absv`）基础上，验证两件事：
1. 是否可通过调整 `ridge_alpha` 提升 test 预测表现。
2. 是否可通过相关性阈值做特征降共线，在不明显伤害指标的前提下降低条件数。

主版本基线：
- 估计文件：`data/us/us_ols_estimation.json`
- 指标文件：`data/us/plot/fit_metrics.json`
- baseline(test)：`r2=0.473201`，`x_r2=0.742587`，`direction_accuracy=0.739130`，`x_rmse=0.034153`
- baseline 条件数：`739.9096`

## 结果与主版本对比
扫描设置：
- `ridge_alpha`：`[0.01,0.03,0.05,0.08,0.1,0.15,0.2]`
- 相关阈值：`[none,0.99,0.97,0.95,0.93,0.90,0.88,0.85]`
- 共 56 组，全部成功。

输出：
- `data/us/experiments/step1_ridge_collinearity/scan_summary.json`
- `data/us/experiments/step1_ridge_collinearity/scan_table.csv`

最佳组（按 `x_r2` 优先）：
- `ridge_alpha=0.03`
- `corr_threshold=None`
- `n_features=18`
- test：`r2=0.475251`，`x_r2=0.743862`，`direction_accuracy=0.760870`，`x_rmse=0.034068`
- 条件数：`739.9096`

相对主版本差值：
- `Δr2 = +0.002050`
- `Δx_r2 = +0.001275`
- `Δdirection_accuracy = +0.021739`
- `Δx_rmse = -0.000085`
- `Δcondition_number = 0`（无改善）

降共线有效性观察：
- 最小特征数仅降到 `17`（大多数组仍为 `18`），说明当前特征在该筛选规则下可删空间有限。
- 低条件数组（约 `688.81`）相对主版本在 `x_r2` 上多数不增反降，收益不稳定。

## 最终结论
1. 本轮最优组合可带来小幅性能提升，但幅度很小（`x_r2 +0.0013` 量级）。
2. 共线性筛选没有显著降低风险（条件数降幅有限且会牺牲一部分效果）。
3. 结论上不建议把“降共线筛选”作为当前主升级方向；可将 `ridge_alpha` 从 `0.05` 微调到 `0.03` 作为可选小优化。
4. 下一步按计划进入试验 02：不对称阻尼（上涨/下跌分离）。

