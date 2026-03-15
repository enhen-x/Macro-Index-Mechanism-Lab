# 试验 20：双任务后处理（幅度回归 + 相位分类）2026-03-15

## 本次尝试目标
将预测拆成两个任务，并在输出层做融合：

1. 幅度任务（回归）  
- 学习 `residual = y_true - y_hat_raw`，得到 `y_hat_amp`

2. 相位任务（分类）  
- 任务 A：拐点概率 `p_turn`  
- 任务 B：上行概率 `p_up`

融合规则：  
- `delta_amp = y_hat_amp - y_t`  
- `g = clip((p_turn - tau)/(1-tau), 0, 1)`  
- `delta_target = |delta_amp| * sign(p_up-0.5)`  
- `delta_final = (1-lambda*g)*delta_amp + lambda*g*delta_target`  
- `y_hat_final = y_t + delta_final`

扫描参数：`lambda` 与 `tau`。

输出：
- `scripts/us/stage_2026_03_15/experiment_us_dual_task_phase.py`
- `data/us/experiments/step20_dual_task_phase/dual_task_summary.json`
- `data/us/experiments/step20_dual_task_phase/dual_task_scan.csv`

## 结果与主版本对比
主版本（test）：
- `x_r2=0.742587`
- `x_rmse=0.034153`
- `best_lag=-1`
- `turn_mean_delay=0.4545`

模型可学性（fit）：
- 幅度头 `amp_fit_r2=0.0303`（可学但较弱）
- 拐点头/方向头非退化（`turn_pos_rate=0.408`, `up_pos_rate=0.578`）

扫描结果：
- 共 80 组（`lambda × tau`），`eval_best_lag` 全部为 `-1`（无 `0`）

按 valid 选型最优（相位优先）：
- `lambda=1.5, tau=0.5`
- test：
  - `x_r2=0.712674`（较主版本 `-0.029913`）
  - `x_rmse=0.036082`（较主版本 `+0.001930`）
  - `best_lag=-1`
  - `turn_mean_delay=0.3846`（改善）

按 test 最优：
- `lambda=0, tau=0.4`（等价不启用相位融合，仅幅度头）
- test：
  - `x_r2=0.739314`
  - `x_rmse=0.034369`
  - `best_lag=-1`
  - `turn_mean_delay=0.3077`

## 最终结论
1. 双任务结构在当前实现下仍不能把相位偏差从 `-1` 修到 `0`。  
2. 按 test 最优参数退化到 `lambda=0`，说明当前相位分类头并未提供稳定增益。  
3. 结论：下一步应升级为“序列分类器/危机状态机”（而非逐点 logistic），让相位任务显式建模持续状态与转移。  


