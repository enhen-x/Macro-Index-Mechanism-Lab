# 阶段实施记录 01（2026-03-15）

## 本次目标
搭建“解释优先”第一版可执行框架：
1. 构造机制面板（shock/state/control）。
2. 跑通状态依赖 LP（baseline + interaction）。

## 已实施内容
- 新增脚本：
  - `scripts/build_us_macro_mechanism_panel.py`
  - `scripts/run_us_sd_lp.py`
- 生成数据：
  - `data/us_macro_mechanism_panel.csv`
  - `data/us_macro_mechanism_panel_meta.json`
  - `data/outputs/sd_lp/sd_lp_coefficients.csv`
  - `data/outputs/sd_lp/sd_lp_summary.json`

## 关键结果（首轮）
- 面板样本：`rows_kept=298/315`。
- LP 结果：`4 shocks x 12 horizons = 48` 条系数记录。
- 在 `state_high_infl` 下，`shock_policy` 的高状态效应更负（h=12 时状态交互为负），符合“紧缩冲击通过贴现率压制估值”的先验方向。

## 下一步
1. 增加 CF/DR 分解脚本（机制拆解核心）。
2. 增加状态切换稳健性（state_low_growth, state_high_vol）。
3. 增加图表脚本（IRF + 通道贡献图）。
