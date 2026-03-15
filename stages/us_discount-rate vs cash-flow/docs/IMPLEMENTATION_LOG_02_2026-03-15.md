# 阶段实施记录 02（2026-03-15）

## 本次目标
将 SD-LP 结果可视化，输出可直接阅读的机制图表。

## 已实施内容
- 新增脚本：`scripts/plot_us_sd_lp_results.py`
- 读取：`data/outputs/sd_lp/sd_lp_coefficients.csv`
- 输出目录：`data/outputs/sd_lp/plots/`

## 输出图表
- `irf_shock_policy.png`
- `irf_shock_inflation.png`
- `irf_shock_growth.png`
- `irf_shock_labor.png`
- `interaction_heatmaps.png`
- `interaction_t_summary.png`
- `plot_notes.json`

## 图表解读（首轮）
- `shock_policy`：高通胀状态下负向影响更强（中长端更明显）。
- `shock_growth`：高通胀状态下正向影响增强（h4-h12 更稳定）。
- `shock_labor`：短中期状态差异明显（h1-h6）。
- `shock_inflation`：状态差异较弱，交互项普遍不显著。

## 下一步
进入 CF/DR 分解脚本开发，形成“总效应 -> 通道贡献”的完整解释链。
