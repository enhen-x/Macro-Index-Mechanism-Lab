# US 试验结果总览与脚本整理（2026-03-15）

## 1. 一句话结论
当前这条“二阶动力 + 相位修复”路线在 US 月频数据上形成了稳定结论：  
**可用精度区间内，`best_lag` 几乎始终停在 `-1`；把 `best_lag` 强行拉到 `0` 会显著破坏拟合精度。**

## 2. 当前主版本（建议基线）
- 面板：`data/us/us_regression_panel.csv`
- 估计：`data/us/us_ols_estimation.json`
- 测试指标（`data/us/plot/fit_metrics.json`）：
  - `x_r2=0.7426`
  - `x_rmse=0.03415`
  - `x_corr=0.8738`
  - `best_lag=-1`（见 `data/us/experiments/step5_phase_bias_baseline/phase_bias_summary.json`）
- 当前主阻尼：`nonlinear_absv`（`c=0.4879`, `c_nl=0.03556`）

## 3. 试验阶段总结

### 3.1 早期稳健化（step1~step5）
- 目标：处理共线性、尾部噪声、建立相位基线。
- 结论：
  - Ridge/稳健化对拟合稳定性有帮助。
  - 非线性阻尼显著优于线性阻尼，成为后续主版本基础。
  - 相位偏差在基线中已明确存在（`best_lag=-1`）。

### 3.2 数据对齐与特征相位修复（step6~step14）
- 目标：通过因果离散、反滞后特征、发布时间 as-of 对齐、显式 delay 参数修复相位。
- 结论：
  - 可改善部分 turning-point delay。
  - 但核心 `best_lag` 基本不变（大多仍 `-1`）。

### 3.3 状态空间与多步一致性（step15~step19）
- 目标：引入显式相位状态、Kalman 风格误差状态、多步一致性。
- 结论：
  - 有些配置可改善延迟指标。
  - `best_lag` 仍难回到 `0`，且常伴随 `x_r2` 下滑。

### 3.4 双任务/损失重构/结构耦合（step20~step25）
- 目标：相位分类头、序列状态机、相位损失、ranking 约束、结构内生相位状态（含非线性门控）。
- 结论：
  - 大规模扫描后仍未出现“高精度 + `best_lag=0`”的稳定解。
  - 复杂结构多数带来精度损失或过补偿风险。

### 3.5 显式分数位移层（step26）
- 目标：把“相位修正”从幅度模型中解耦（`y_shift_t = y_raw_t + tau*(y_raw_t-y_raw_(t-1))`）。
- 结论：
  - 扫描出现了 `best_lag=0`，但对应解精度崩塌（例如 `x_r2≈0.091`）。
  - 高精度最优点（`x_r2≈0.760`）仍是 `best_lag=-1`。
  - 验证了“相位-精度硬冲突”。

## 4. 脚本与产物索引（按步骤）
| Step | 脚本 | 主要输出目录 | 结论摘要 |
|---|---|---|---|
| 01 | `scripts/us/stage_2026_03_15/scan_us_ridge_collinearity.py` | `data/us/experiments/step1_ridge_collinearity` | Ridge 缓解共线性，但不解相位根因 |
| 02 | `scripts/us/stage_2026_03_15/experiment_us_asym_damping.py` | `data/us/experiments/step2_asym_damping` | 非线性阻尼路线可行 |
| 03 | `scripts/us/stage_2026_03_15/experiment_us_walk_forward.py` | `data/us/experiments/step3_walk_forward` | walk-forward 可跑通，但 lag 问题延续 |
| 04 | `scripts/us/stage_2026_03_15/scan_us_tail_weight_extreme.py` | `data/us/experiments/step4_tail_weight_extreme` | 尾部加权改善有限 |
| 05 | `scripts/us/stage_2026_03_15/experiment_us_phase_bias_baseline.py` | `data/us/experiments/step5_phase_bias_baseline` | 建立基线：`best_lag=-1` |
| 06 | `scripts/us/stage_2026_03_15/build_us_regression_panel.py` + `scripts/us/stage_2026_03_15/run_us_ols_estimation.py` + `scripts/us/stage_2026_03_15/plot_us_ols_fit.py` | `data/us/experiments/step6_causal_diff_phase` | 因果差分版未根治相位 |
| 07 | `scripts/us/stage_2026_03_15/build_us_phase_antilag_panel.py` | `data/us/experiments/step7_phase_antilag_features` | 反滞后特征改善有限 |
| 08 | `scripts/us/stage_2026_03_15/experiment_us_phase_arx_correction.py` | `data/us/experiments/step8_phase_arx_correction` | ARX 校正有局部提升，非根治 |
| 09 | `scripts/us/stage_2026_03_15/experiment_us_phase_feature_structure.py` | `data/us/experiments/step9_phase_feature_structure` | 特征结构重排无决定性改善 |
| 10 | `scripts/us/stage_2026_03_15/experiment_us_phase_lead_compensation.py` | `data/us/experiments/step10_phase_lead_compensation` | lead 补偿存在过补偿风险 |
| 11 | `scripts/us/stage_2026_03_15/experiment_us_phase_lead_compensation.py`（valid 选型） | `data/us/experiments/step11_phase_lead_comp_valid_select` | valid 选型仍未把 lag 推到 0 |
| 12 | `scripts/us/stage_2026_03_15/experiment_us_learnable_inertia_rho.py` | `data/us/experiments/step12_learnable_inertia_rho` | 学习惯性参数仍保留 lag 偏差 |
| 13 | `scripts/us/stage_2026_03_15/build_us_asof_monthly_panel.py` | `data/us/experiments/step13_asof_alignment` | as-of 对齐改善延迟，不解根因 |
| 13B | `scripts/us/stage_2026_03_15/scan_us_asof_release_lag_configs.py` | `data/us/experiments/step13b_asof_lag_scan` | 滞后配置扫描无 0-lag 稳定解 |
| 14 | `scripts/us/stage_2026_03_15/experiment_us_explicit_delay_params.py` | `data/us/experiments/step14_explicit_delay_params` | 显式 delay 多退化回基线 |
| 15 | `scripts/us/stage_2026_03_15/experiment_us_explicit_phase_state.py` | `data/us/experiments/step15_explicit_phase_state` | 相位状态化后精度下降 |
| 16 | `scripts/us/stage_2026_03_15/experiment_us_error_state_kalman.py` | `data/us/experiments/step16_error_state_kalman` | 延迟略改善，lag 不变 |
| 17 | `scripts/us/stage_2026_03_15/experiment_us_delta_target_dynamics.py` | `data/us/experiments/step17_delta_target_dynamics` | 改 `ΔY` 目标后仍 `lag=-1` |
| 18 | `scripts/us/stage_2026_03_15/experiment_us_multistep_consistency_state.py` | `data/us/experiments/step18_multistep_consistency_state` | 多步一致性仍无根治 |
| 19 | `scripts/us/stage_2026_03_15/experiment_us_kalman_phase_2state.py` | `data/us/experiments/step19_kalman_phase_2state` | 二维状态仍无法回到 0-lag |
| 20 | `scripts/us/stage_2026_03_15/experiment_us_dual_task_phase.py` | `data/us/experiments/step20_dual_task_phase` | 双任务融合贡献有限 |
| 21 | `scripts/us/stage_2026_03_15/experiment_us_sequence_phase_state.py` | `data/us/experiments/step21_sequence_phase_state` | 状态机过滤仍未解决核心相位 |
| 22 | `scripts/us/stage_2026_03_15/experiment_us_phase_aware_training_loss.py` | `data/us/experiments/step22_phase_aware_training_loss` | 相位损失可降延迟但有过补偿 |
| 23 | `scripts/us/stage_2026_03_15/experiment_us_lag_ranking_loss.py` | `data/us/experiments/step23_lag_ranking_loss` | ranking 约束基本失效 |
| 24 | `scripts/us/stage_2026_03_15/experiment_us_structural_phase_state.py` | `data/us/experiments/step24_structural_phase_state` | 结构耦合版精度下降且 lag 不变 |
| 25 | `scripts/us/stage_2026_03_15/experiment_us_structural_phase_state_nonlinear.py` | `data/us/experiments/step25_structural_phase_state_nonlinear` | 非线性门控引入过补偿与精度劣化 |
| 26 | `scripts/us/stage_2026_03_15/experiment_us_fractional_shift_layer.py` | `data/us/experiments/step26_fractional_shift_layer` | 证明相位-精度硬冲突 |

## 5. 当前建议的目录使用方式
- 主流程（继续维护）：
  - `scripts/us/stage_2026_03_15/build_us_regression_panel.py`
  - `scripts/us/stage_2026_03_15/run_us_ols_estimation.py`
  - `scripts/us/stage_2026_03_15/plot_us_ols_fit.py`
- 试验脚本（归档对照，不再作为主线）：
  - `scripts/us/stage_2026_03_15/*.py`
- 总结与复盘入口：
  - 总日志：`docs/us/US_EXPERIMENT_LOG_2026-03-15.md`
  - 本文档：`docs/us/US_EXPERIMENT_SUMMARY_2026-03-15.md`

## 6. 最终判断
在现有数据与评估框架下，这条建模路径已经完成了足够多的“反证式扫描”。  
下一步应切换到新范式（例如直接监督 `Y(t+1)` + 状态条件化/集成），而不是继续在当前相位修复链路上做局部微调。


