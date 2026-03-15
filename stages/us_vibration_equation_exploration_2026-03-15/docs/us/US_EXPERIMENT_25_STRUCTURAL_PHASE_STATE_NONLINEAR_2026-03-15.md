# 试验 25：结构内生相位状态（非线性门控版）2026-03-15

## 本次尝试目标
在 step24 的结构内生相位框架上，引入非线性相位驱动与门控递推，提高相位状态表达力：

- 幅度：`delta_base = f_amp(z)`
- 非线性相位驱动：`drive = f_phase_nl(z, delta_base)`（`[x, x^2, tanh(x)]` 基函数）
- 门控递推：
  - `gate = sigmoid(k_gate * gate_signal)`
  - `rho_eff = rho_low*(1-gate) + rho_high*gate`
  - `d_t = rho_eff * d_(t-1) + drive_t`
- 输出：`y_hat = y_t + delta_base * (1 + gamma*d_t)`

输出：
- `scripts/us/stage_2026_03_15/experiment_us_structural_phase_state_nonlinear.py`
- `data/us/experiments/step25_structural_phase_state_nonlinear/structural_phase_nl_summary.json`
- `data/us/experiments/step25_structural_phase_state_nonlinear/structural_phase_nl_scan.csv`

## 结果与主版本对比
主版本（test）：
- `x_r2=0.742587`
- `x_rmse=0.034153`
- `best_lag=-1`

step25 扫描：
- 组合数：48（`phase_ridge × rho_low × rho_high × gate_k × gamma`）
- `eval_best_lag`：48 组全部为 `-1`（无 `0`）

按 valid 选型最优：
- `phase_ridge=0.1, rho_low=0.3, rho_high=0.9, gate_k=2, gamma=1.5`
- test：
  - `x_r2=0.607301`（较主版本 `-0.135287`）
  - `x_rmse=0.040839`（较主版本 `+0.006686`）
  - `best_lag=-1`
  - `turn_mean_delay=0.0714`（显著改善）
  - `mean_abs_d_state≈0.650`（状态幅度较大）

按 test 最优：
- `phase_ridge=0.01, rho_low=0, rho_high=0.9, gate_k=2, gamma=1.0`
- test：
  - `x_r2=0.661662`
  - `x_rmse=0.037907`
  - `best_lag=-1`

## 最终结论
1. 非线性门控让相位状态更“有动作”，但依旧无法把 `best_lag` 推到 `0`。  
2. 代价是显著精度恶化，说明当前耦合形式出现过补偿。  
3. 结论：继续加强这一路线风险高，下一步应改为“显式时移参数层（可学习分数延迟）+ 幅度模型分离”而非继续放大门控非线性。  


