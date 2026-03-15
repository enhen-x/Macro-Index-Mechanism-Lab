# 试验 04：Tail Weight（极端样本加权）（2026-03-15）

## 本次尝试目标
验证“训练阶段对尾部样本加权”是否能改善极端行情拟合，同时评估其对整体 test 指标的副作用。

基线主版本：
- `data/us/us_ols_estimation.json`
- `data/us/plot/fit_metrics.json`
- baseline(test)：`x_r2=0.742587`，`x_rmse=0.034153`，`direction_accuracy=0.739130`
- baseline tail（按 `|a|` 的 `q=0.9`，尾部样本 `n=5`）：`tail_x_r2=-1.678469`，`tail_x_rmse=0.063311`

实验输出：
- `data/us/experiments/step4_tail_weight_extreme/tail_weight_summary.json`
- `data/us/experiments/step4_tail_weight_extreme/tail_weight_table.csv`

## 结果与主版本对比
扫描配置：
- `tail_weight_mode in {none, abs_power}`
- `tail_weight_q in {0.85, 0.9, 0.93}`
- `tail_weight_scale in {2,3,4,5}`
- `tail_weight_power in {1.0,1.5}`
- 共 `8` 组，全部成功。

按“尾部 `x_r2` 优先”选出的最优组：
- `tw_q90_s5_p15`
- 参数：`tail_weight_mode=abs_power`，`q=0.9`，`scale=5.0`，`power=1.5`
- `condition_number=673.665`

最优组与主版本对比：
- 整体（overall）：
- `x_r2: 0.684264 vs 0.742587`（`Δ=-0.058324`）
- `x_rmse: 0.037824 vs 0.034153`（`Δ=+0.003672`）
- `direction_accuracy: 0.717391 vs 0.739130`（`Δ=-0.021739`）
- 尾部（tail）：
- `tail_x_r2: -0.347873 vs -1.678469`（显著改善）
- `tail_x_rmse: 0.044912 vs 0.063311`（显著改善）
- `tail_a_direction: 1.0 vs 1.0`（持平）

## 最终结论
1. Tail weighting 能明显改善“极端样本子集”的拟合，但会显著牺牲整体 test 质量。
2. 当前目标仍以整体稳定预测为主，因此不建议将 tail-weight 方案替换为主版本。
3. 可将该方案保留为“风险事件专用备选模型”，仅在强调极端阶段表现时启用。

