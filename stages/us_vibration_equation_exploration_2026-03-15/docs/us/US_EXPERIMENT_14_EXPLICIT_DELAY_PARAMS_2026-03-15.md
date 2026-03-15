# 试验 14：将“延迟”显式建模为参数（分组 delay 扫描）2026-03-15

## 本次尝试目标
将延迟从“隐含在系数里”改为“显式参数化”，按特征组单独设置离散延迟：
- `market`（`g_*`）延迟 `d_market`
- `policy`（`c_policy_rate*`）延迟 `d_policy`
- `macro`（其余 `c_*`）延迟 `d_macro`

在 `d ∈ {0,1,2}` 下做全组合扫描（共 27 组），流程为：
1. 固定主版本动力结构（`y_next + nonlinear_absv + ridge=0.05`）
2. `train` 拟合、`valid` 选型、`test` 最终评估
3. 对比主版本的 `x_r2/x_rmse/best_lag/turn` 指标

输出：
- `scripts/us/stage_2026_03_15/experiment_us_explicit_delay_params.py`
- `data/us/experiments/step14_explicit_delay_params/explicit_delay_summary.json`
- `data/us/experiments/step14_explicit_delay_params/explicit_delay_table.csv`

## 结果与主版本对比
主版本（baseline, test）：
- `x_r2=0.742587`
- `x_rmse=0.034153`
- `best_lag=-1`
- `turn_recall=0.6471`
- `turn_precision=0.6111`
- `turn_mean_delay=0.4545`

step14 关键结果：
1. 按 test 最优的组合仍为 `d_market=0, d_policy=0, d_macro=0`（即退化回主版本），指标完全一致。
2. 按 valid 选出的最优组合为 `d_market=1, d_policy=0, d_macro=2`：
   - `x_r2=0.686481`（较主版本 `-0.056106`）
   - `x_rmse=0.036490`（较主版本 `+0.002337`）
   - `best_lag` 仍为 `-1`
   - `turn_mean_delay=0.0`（较主版本改善）
3. 27 组候选中，没有任何组合把 `best_lag` 从 `-1` 推到 `0`。

## 最终结论
1. 显式 delay 参数化在当前离散动力结构下不能根治相位偏差（`best_lag` 全部仍为 `-1`）。
2. 延迟参数能局部改善拐点延迟，但通常伴随 `x_r2` 下降、`x_rmse` 上升，存在明显精度-时序折中。
3. 结论：该方向已验证“非决定性”，下一步应转向更强结构改造（例如显式相位状态/状态空间或误差动态方程），而不是继续扩大 delay 网格。


