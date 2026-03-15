# 试验 24：结构内生相位状态（主方程耦合）2026-03-15

## 本次尝试目标
把相位状态放进主方程内部，而非后处理补丁：

- 幅度动力：`delta_base_t = f_amp(z_t)`
- 相位状态递推：`d_t = rho_d * d_(t-1) + g_phase(z_t, delta_base_t)`
- 耦合输出：`y_hat_(t+1) = y_t + delta_base_t * (1 + gamma * d_t)`

其中 `d_t` 为内生相位状态，直接调节下一期位移幅度与方向敏感性。

输出：
- `scripts/us/stage_2026_03_15/experiment_us_structural_phase_state.py`
- `data/us/experiments/step24_structural_phase_state/structural_phase_summary.json`
- `data/us/experiments/step24_structural_phase_state/structural_phase_scan.csv`

## 结果与主版本对比
主版本（test）：
- `x_r2=0.742587`
- `x_rmse=0.034153`
- `best_lag=-1`

step24 扫描：
- 组合数：36（`rho × phase_ridge × gamma`）
- `eval_best_lag`：36 组全部为 `-1`（无 `0`）

按 valid 选型最优：
- `rho=0.3, phase_ridge=0.1, gamma=1.0`
- test：
  - `x_r2=0.705571`（较主版本 `-0.037017`）
  - `x_rmse=0.035362`（较主版本 `+0.001209`）
  - `best_lag=-1`
  - `turn_mean_delay=0.1818`（改善）

按 test 最优：
- `rho=0, phase_ridge=0.001, gamma=1.5`
- test：
  - `x_r2=0.700245`
  - `x_rmse=0.035680`
  - `best_lag=-1`

## 最终结论
1. 相位状态“内生化”的方向是对的，但当前线性化实现仍无法把 `best_lag` 推到 `0`。  
2. 代价是显著的精度下降（`x_r2/x_rmse` 均劣于主版本）。  
3. 结论：下一步需要提升相位状态方程的非线性表达（例如非线性 `g_phase`、状态转移门控），否则会出现“有状态但无有效相位纠偏”的情况。  


