# 试验 18：递推状态方程 + 多步一致性约束 2026-03-15

## 本次尝试目标
在 `step17` 基础上进一步增强时序结构：不只做一步误差最小化，而是把多步一致性（h=1/2/3）显式纳入选型。

模型结构：
- `delta_(t+1) = rho * delta_t + r_theta(s_t)`
- `Y_hat(t+1) = Y_t + delta_(t+1) + phase_gain * delta_t`
- `s_t = [Y_t, delta_t, accel_t, u_t]`

其中：
- `rho`：显式惯性/延迟参数
- `phase_gain`：显式相位补偿参数
- `r_theta`：ridge 残差回归项

选型标准（valid）：
1. 优先最小化 `|best_lag|`
2. 再按多步组合误差 `combo_rmse(h=1,2,3; 权重=1.0,0.7,0.5)` 排序
3. 再看一步 `x_rmse/x_corr`

输出：
- `scripts/us/stage_2026_03_15/experiment_us_multistep_consistency_state.py`
- `data/us/experiments/step18_multistep_consistency_state/multistep_state_summary.json`
- `data/us/experiments/step18_multistep_consistency_state/multistep_state_scan.csv`

## 结果与主版本对比
主版本（test）：
- `x_r2=0.742587`
- `x_rmse=0.034153`
- `best_lag=-1`
- `turn_mean_delay=0.4545`

step18 扫描：
- 组合数：`11(alpha) × 11(rho) × 9(phase_gain) = 1089`
- `eval_best_lag` 分布：1089 组全部为 `-1`（无 `0`）

按 valid 选型最优：
- `alpha=10, rho=0.0, phase_gain=-0.3`
- test：
  - `x_r2=0.704891`（较主版本 `-0.037697`）
  - `x_rmse=0.035402`（较主版本 `+0.001250`）
  - `best_lag=-1`
  - `turn_mean_delay=0.3077`（较主版本改善）
  - 多步 RMSE：`h1=0.03540, h2=0.05072, h3=0.06001`

按 test 排序最优：
- `alpha=10, rho=0.1, phase_gain=0.0`
- `x_r2=0.717774`, `x_rmse=0.034621`, `best_lag=-1`
- 多步组合误差 `combo_rmse=0.04977`

## 最终结论
1. 把“多步一致性”显式纳入后，仍无法将相位从 `best_lag=-1` 推到 `0`。  
2. 该方向可继续改善延迟类指标（`turn_mean_delay` 下降），但整体精度仍低于主版本。  
3. 结论：当前可观测状态变量仍不足以表达真实相位机理；下一步应尝试引入“不可观测相位状态 + 联合估计”（例如 Kalman/EM）或改为直接建模拐点事件强度。  


