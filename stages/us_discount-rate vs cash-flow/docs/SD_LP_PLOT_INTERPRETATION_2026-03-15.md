# SD-LP 图表解读报告（2026-03-15）

## 1. 解读口径
- 数据：`data/us_macro_mechanism_panel.csv`，样本 298 行。
- 模型：状态依赖局部投影（SD-LP），状态变量为 `state_high_infl`（高通胀=1）。
- 冲击：`shock_policy`, `shock_inflation`, `shock_growth`, `shock_labor`。
- 期限：h=1..12（月）。
- 显著性粗判：`|t_interaction| >= 1.96`。

## 2. 图1：政策冲击（shock_policy）
![shock_policy](../data/outputs/sd_lp/plots/irf_shock_policy.png)

图怎么看：
- 上图：蓝线是低通胀状态，红线是高通胀状态；纵轴是累计收益响应系数。
- 下图：交互项（高通胀相对低通胀的增量效应），绿色柱代表显著期。

结论：
- 政策冲击在高通胀状态下更负，且在多期限显著（显著期：h=1,2,5,6,12）。
- `h=12` 时：
  - 低通胀：`beta_state0=-0.024999`
  - 高通胀：`beta_state1=-0.103425`
  - 增量：`beta_interaction=-0.078426`（`t=-2.281`）
- 解释：同样的政策冲击，在高通胀环境里对估值压制更强（贴现率通道更“硬”）。

## 3. 图2：通胀冲击（shock_inflation）
![shock_inflation](../data/outputs/sd_lp/plots/irf_shock_inflation.png)

结论：
- 两个状态下总体都偏负，但状态差异弱。
- 交互项在 1-12 月均未达到显著阈值。
- `h=12`：
  - 低通胀：`-0.010666`
  - 高通胀：`-0.030075`
  - 增量：`-0.019409`（`t=-0.917`）
- 解释：方向上“高通胀更负”存在，但统计稳健性不足，当前样本不能下强结论。

## 4. 图3：增长冲击（shock_growth）
![shock_growth](../data/outputs/sd_lp/plots/irf_shock_growth.png)

结论：
- 状态差异非常清楚：
  - 低通胀状态：效应接近 0 或略负（均值 `-0.000948`）
  - 高通胀状态：效应稳定为正（均值 `0.031586`）
- 交互项显著期集中在中长期（h=4..12）。
- 峰值显著大约在 `h=10`（`t=2.662`, `beta_interaction=0.048395`）。
- `h=12`：高通胀相对低通胀增量 `+0.040645`（显著）。
- 解释：高通胀背景下，增长冲击对权益的正向“缓冲/支撑”更明显。

## 5. 图4：就业冲击（shock_labor）
![shock_labor](../data/outputs/sd_lp/plots/irf_shock_labor.png)

结论：
- 低通胀状态下接近零偏负（均值 `-0.002592`），高通胀状态下明显偏正（均值 `0.033963`）。
- 交互项显著期集中在短中期（h=1..6）。
- 峰值在 `h=2`（`t=2.518`, `beta_interaction=0.018150`）。
- `h=12` 仍为正增量（`+0.046632`），但显著性下降（`t=1.650`）。
- 解释：就业改善冲击在高通胀阶段的正向传导更偏前端期限。

## 6. 图5：交互项热力图总览
![heatmap](../data/outputs/sd_lp/plots/interaction_heatmaps.png)

左图（interaction beta）：
- 红色为正增量、蓝绿色为负增量。
- 直观看到：
  - `shock_policy` 全期限偏负增量（高通胀更差）。
  - `shock_growth`、`shock_labor` 多数期限偏正增量（高通胀更好）。
  - `shock_inflation` 颜色弱，增量幅度整体小。

右图（interaction t-stat）：
- 颜色越深代表统计证据越强。
- 强证据主要集中在：
  - `shock_policy`（若干期限负向显著）
  - `shock_growth`（中长期正向显著）
  - `shock_labor`（短中期正向显著）

## 7. 图6：交互项显著性强度汇总
![t_summary](../data/outputs/sd_lp/plots/interaction_t_summary.png)

结论：
- 平均 `|t_interaction|` 排名：
  1. `shock_growth` ≈ 2.081
  2. `shock_policy` ≈ 2.035
  3. `shock_labor` ≈ 2.006
  4. `shock_inflation` ≈ 0.542
- 前三者平均超过 1.96，说明状态差异具备系统性证据；通胀冲击状态差异最弱。

## 8. 综合结论（详细）
1. 当前样本明确支持“状态依赖机制”：宏观冲击对指数影响不是常数，而是随通胀状态变化。  
2. 政策冲击在高通胀状态下更强烈压制回报，是最稳健的负向状态放大效应。  
3. 增长冲击在高通胀状态下转为显著正向，且中长期（h=4..12）更稳定。  
4. 就业冲击在高通胀状态下也偏正，但主要体现在短中期（h=1..6），期限结构与增长冲击不同。  
5. 通胀冲击本身虽偏负，但“高通胀 vs 非高通胀”的额外差异目前统计证据不足。  
6. 机制含义上，已有结果与“贴现率敏感性在高通胀状态上升”一致；但目前仍是总效应层面，尚未完成 CF/DR 通道拆分。  

## 9. 边界与注意事项
- 当前冲击是月频代理冲击，不是高频公告 surprise 识别，因果解释需谨慎。
- 显著性阈值是经验规则，不等于经济意义大小；需结合系数量级解读。
- 下一步应完成 `cash-flow vs discount-rate` 分解，形成机制闭环。
