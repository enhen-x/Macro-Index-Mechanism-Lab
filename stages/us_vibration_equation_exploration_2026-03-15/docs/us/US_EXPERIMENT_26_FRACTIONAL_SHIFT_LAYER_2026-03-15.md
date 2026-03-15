# 试验 26：显式分数位移层（幅相解耦）2026-03-15

## 本次尝试目标
把“相位修正”从幅度模型中解耦出来，显式建模为一个分数位移层：

- 基础一阶预测：`y_raw_t`
- 分数位移：`y_shift_t = y_raw_t + tau * (y_raw_t - y_raw_(t-1))`
- 可选步长裁剪：`step_clip_std`（抑制极端跳变导致的过补偿）

希望验证：是否能在尽量不损失幅度拟合的情况下，把 `best_lag` 从 `-1` 推到 `0`。

输出：
- `scripts/us/stage_2026_03_15/experiment_us_fractional_shift_layer.py`
- `data/us/experiments/step26_fractional_shift_layer/fractional_shift_summary.json`
- `data/us/experiments/step26_fractional_shift_layer/fractional_shift_scan.csv`

## 结果与主版本对比
主版本（test）：
- `x_r2=0.742587`
- `x_rmse=0.034153`
- `best_lag=-1`

step26 扫描：
- 组合数：204（`tau=-0.5..2.0`，步长 `0.05`，`step_clip_std∈{0,2,3,4}`）
- `eval_best_lag` 分布：`-1` 有 126 组，`0` 有 78 组

按 valid 选型最优（当前主流程一致）：
- `tau=-0.45, step_clip_std=3`
- test：
  - `x_r2=0.713419`（较主版本 `-0.029168`）
  - `x_rmse=0.036036`（较主版本 `+0.001883`）
  - `best_lag=-1`

按 test 全局最优（仅看精度）：
- `tau=-0.2, step_clip_std=2`
- test：
  - `x_r2=0.759583`（较主版本 `+0.016995`）
  - `x_rmse=0.033006`（较主版本 `-0.001147`）
  - `best_lag=-1`

按 test 中 `best_lag=0` 的最优：
- `tau=1.0, step_clip_std=0`
- test：
  - `x_r2=0.091020`
  - `x_rmse=0.064178`
  - `best_lag=0`

## 最终结论
1. 显式分数位移层确实能把 `best_lag` 推到 `0`，但只在明显“过前移”的配置下出现，且幅度拟合显著崩塌。  
2. 当保持合理精度时（甚至优于主版本），最优解仍稳定在 `best_lag=-1`。  
3. 这说明当前相位偏差不是“单一常数时间平移”能根治的问题，更像是状态相关/非线性耦合偏差；该层可作为后处理旋钮，但不足以单独解决核心相位问题。  


