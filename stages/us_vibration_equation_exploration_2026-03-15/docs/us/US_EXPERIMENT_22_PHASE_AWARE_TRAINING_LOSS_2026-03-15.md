# 试验 22：训练时显式加入相位损失 2026-03-15

## 本次尝试目标
不再做后处理补偿，而是在训练目标中直接加入相位项。

模型：
- `delta_hat = X * theta`
- `y_hat = y_t + delta_hat`

训练损失：
- `L_amp = MSE(y_hat - y_true)`（幅度）
- `L_phase = MSE(diff(y_hat) - diff(y_true))`（相位时序）
- `L_turn = MSE(sigmoid(k*delta_hat) - sigmoid(k*delta_true))`（方向软约束）
- `L_total = L_amp + lambda_phase*L_phase + lambda_turn*L_turn + ridge`

扫描：
- `lambda_phase × lambda_turn × ridge`，共 180 组。

输出：
- `scripts/us/stage_2026_03_15/experiment_us_phase_aware_training_loss.py`
- `data/us/experiments/step22_phase_aware_training_loss/phase_aware_loss_summary.json`
- `data/us/experiments/step22_phase_aware_training_loss/phase_aware_loss_scan.csv`

## 结果与主版本对比
主版本（test）：
- `x_r2=0.742587`
- `x_rmse=0.034153`
- `best_lag=-1`
- `turn_mean_delay=0.4545`

step22 扫描结果：
- `eval_best_lag` 分布：`-1`（150 组）、`+3`（18 组）、`-3`（12 组）
- 仍然没有 `0`

按 valid 选型最优：
- `lambda_phase=0.5, lambda_turn=0.2, ridge=0.05`
- test：
  - `x_r2=0.721630`（较主版本 `-0.020958`）
  - `x_rmse=0.034384`（较主版本 `+0.000231`）
  - `best_lag=-1`
  - `turn_mean_delay=0.1000`（明显改善）

按 test 最优：
- `lambda_phase=0.2, lambda_turn=0.0, ridge=0.01`
- test：
  - `x_r2=0.722607`
  - `x_rmse=0.034323`
  - `best_lag=-1`
  - `turn_mean_delay=0.0`

## 最终结论
1. 相位损失确实改变了训练行为（延迟显著下降），但仍未把核心相位偏差从 `-1` 推到 `0`。  
2. 新增了 `±3` 极端 lag 候选，说明相位项存在过补偿风险，稳定性不足。  
3. 结论：仅加“导数相位损失”不够，需要在结构上加入对 lag0 的硬约束或可微对齐项（例如 lag0-vs-lag-1 对比损失）。  


