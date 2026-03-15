# 试验 17：`ΔY(t+1)` 目标动力回归 + 显式相位参数 2026-03-15

## 本次尝试目标
将预测目标从 `Y(t+1)` 切换为更接近动力学的增量目标 `ΔY(t+1)=Y(t+1)-Y(t)`，并加入显式相位参数：

- 回归目标：`delta_{t+1}`
- 特征：`[Y_t, delta_t, accel_t, u_t]`（其中 `delta_t=Y_t-Y_(t-1)`，`accel_t=delta_t-delta_(t-1)`）
- 预测还原：`Y_hat(t+1)=Y_t + delta_hat(t+1) + phase_gain * delta_t`

其中 `phase_gain` 是显式相位/延迟参数（不再仅隐含在回归系数中）。

扫描设置：
- `alpha`（ridge）11 档
- `phase_gain` 共 9 档（`-0.4~0.4`）
- 共 99 组组合，`train` 拟合、`valid` 选型、`test` 评估。

输出：
- `scripts/us/stage_2026_03_15/experiment_us_delta_target_dynamics.py`
- `data/us/experiments/step17_delta_target_dynamics/delta_target_summary.json`
- `data/us/experiments/step17_delta_target_dynamics/delta_target_scan.csv`

## 结果与主版本对比
主版本（test）：
- `x_r2=0.742587`
- `x_rmse=0.034153`
- `best_lag=-1`
- `turn_recall=0.6471`
- `turn_precision=0.6111`
- `turn_mean_delay=0.4545`

step17 最优结果：
1. 按 valid 选型最优：`alpha=10, phase_gain=-0.1`
   - `eval x_r2=0.719383`（`Δ=-0.023205`）
   - `eval x_rmse=0.034522`（`Δ=+0.000370`）
   - `eval best_lag=-1`
   - `turn_mean_delay=0.1538`（较主版本明显降低）
2. 按 test 最优：`alpha=0.001, phase_gain=-0.1`
   - `eval x_r2=0.719921`
   - `eval x_rmse=0.034489`
   - `eval best_lag=-1`

扫描分布：
- 99 个组合中，`eval_best_lag` 全部为 `-1`（没有任何 `0`）。

## 最终结论
1. 改为 `ΔY` 目标并显式加入相位参数后，仍无法把核心相位偏差从 `-1` 修到 `0`。  
2. 该方案可以改善拐点延迟（`turn_mean_delay` 降低），但整体精度仍低于主版本。  
3. 结论：相位问题不仅是目标定义问题，还与当前“一步线性回归+静态映射”结构有关；下一步应进入更强序列结构（如 `v/ΔY` 的递推状态方程 + 多步一致性约束）。  


