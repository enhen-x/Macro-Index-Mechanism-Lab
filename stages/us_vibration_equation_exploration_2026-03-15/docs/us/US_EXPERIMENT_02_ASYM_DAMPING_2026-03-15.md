# 试验 02：不对称阻尼（上涨/下跌分离）（2026-03-15）

## 本次尝试目标
在主版本 `nonlinear_absv` 基础上，验证不对称阻尼是否能提升预测：
- 将阻尼拆为 `c_up`（v>0）和 `c_down`（v<0），并保留 `c_nl*|v|v`。
- 比较其相对主版本在 test 上的增益/损失。

主版本基线（来自 `data/us/plot/fit_metrics.json`）：
- `r2=0.473201`
- `x_r2=0.742587`
- `direction_accuracy=0.739130`
- `x_rmse=0.034153`

## 结果与主版本对比
扫描设置：
- `ridge_alpha`：`[0.01,0.03,0.05,0.08,0.1,0.15]`

输出：
- `data/us/experiments/step2_asym_damping/asym_scan_summary.json`
- `data/us/experiments/step2_asym_damping/asym_scan_table.csv`

最佳组：
- `ridge_alpha=0.01`
- 参数：`c_up=0.523290`，`c_down=0.697132`，`c_nl=0.018726`，`k=0.092337`
- 条件数：`808.2446`
- test：`r2=0.462011`，`x_r2=0.737589`，`direction_accuracy=0.760870`，`x_rmse=0.034483`

相对主版本差值：
- `Δr2 = -0.011190`
- `Δx_r2 = -0.004998`
- `Δdirection_accuracy = +0.021739`
- `Δx_rmse = +0.000330`
- 条件数更高（`808.24 > 739.91`）

## 最终结论
1. 不对称阻尼提高了方向判定，但在核心回归与一步预测精度上整体退化（`r2/x_r2` 下滑，`x_rmse` 变差）。
2. 同时数值稳定性变差（条件数上升）。
3. 结论：当前阶段不建议将不对称阻尼替换主版本，主版本仍保持 `nonlinear_absv`。
4. 下一步进入试验 03：walk-forward 时序泛化验证。

