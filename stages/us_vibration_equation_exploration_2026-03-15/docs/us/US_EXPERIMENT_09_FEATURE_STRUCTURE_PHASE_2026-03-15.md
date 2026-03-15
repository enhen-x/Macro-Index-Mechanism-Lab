# 试验 09：特征结构去滞后扫描（2026-03-15）

## 本次尝试目标
围绕“`*_lag1` 可能导致相位滞后”的假设，扫描不同特征结构并比较：
1. 相位指标是否改善（`best_lag` 是否从 `-1` 向 `0` 收敛）。
2. 主指标是否可接受（`x_r2/x_rmse` 不明显退化）。

候选结构：
- `full_main`（主版本全量）
- `no_lag1`
- `base_plus_g_lag1`
- `base_plus_c_lag1`
- `base_plus_top4_lag1`
- `lag1_only`

输出：
- `scripts/us/stage_2026_03_15/experiment_us_phase_feature_structure.py`
- `data/us/experiments/step9_phase_feature_structure/feature_structure_summary.json`
- `data/us/experiments/step9_phase_feature_structure/feature_structure_table.csv`

## 结果与主版本对比
共同结果：
- 所有候选均为 `best_lag=-1`，相位滞后没有被消除。

代表性对比：
- `no_lag1`：
- `x_r2=0.732648`（`Δ=-0.00994`）
- `x_rmse=0.034806`（`Δ=+0.000653`）
- `best_lag=-1`（无改善）
- 拐点 recall/precision 有提升，但 `mean_delay` 反而变大（`0.5833`）。

- `base_plus_top4_lag1`：
- `x_r2=0.737068`（`Δ=-0.00552`）
- `x_rmse=0.034517`（`Δ=+0.000364`）
- `best_lag=-1`（无改善）

- `lag1_only`（相位候选里“相对最优”）：
- `x_r2=0.726164`（`Δ=-0.01642`）
- `x_rmse=0.035225`（`Δ=+0.001073`）
- `best_lag=-1`（无改善）

## 最终结论
1. 调整 `lag1` 特征结构并不能解决当前一期相位滞后（`best_lag` 固定在 `-1`）。
2. 大多数结构都会牺牲整体主指标，代价高于收益。
3. 结论：不替换主版本；相位修复需要更强的结构改造，而非仅做 lag1 特征裁剪。


