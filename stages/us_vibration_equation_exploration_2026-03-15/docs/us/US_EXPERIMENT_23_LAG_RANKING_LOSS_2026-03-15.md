# 试验 23：lag0-vs-lag-1 Ranking Margin 硬约束 2026-03-15

## 本次尝试目标
把“lag0 应优于 lag-1”直接写进训练损失，避免仅靠间接相位项。

模型：
- `delta_hat = X * theta`
- `y_hat = y_t + delta_hat`

损失函数：
- `L_amp`：幅度误差
- `L_phase`：导数相位项
- `L_turn`：方向软约束
- `L_rank = relu(margin + mse_lag0 - mse_lag-1)`（核心硬约束）
- `L_total = L_amp + lambda_phase*L_phase + lambda_turn*L_turn + lambda_rank*L_rank + ridge`

输出：
- `scripts/us/stage_2026_03_15/experiment_us_lag_ranking_loss.py`
- `data/us/experiments/step23_lag_ranking_loss/lag_ranking_loss_summary.json`
- `data/us/experiments/step23_lag_ranking_loss/lag_ranking_loss_scan.csv`

## 结果与主版本对比
主版本（test）：
- `x_r2=0.742587`
- `x_rmse=0.034153`
- `best_lag=-1`

step23 扫描：
- 组合数：160（`lambda_phase × lambda_turn × ridge × lambda_rank × margin`）
- `eval_best_lag` 分布：160 组全部为 `-1`（无 `0`）
- 在 `lambda_rank>0` 的 128 组中，`eval_best_lag` 也全部为 `-1`

最优解（按 valid / 按 test）都退化为：
- `lambda_rank=0, margin=0`（即不启用 ranking 约束）

按 valid 最优（退化解）：
- `lambda_phase=0.5, lambda_turn=0.2, ridge=0.05`
- test：
  - `x_r2=0.721630`
  - `x_rmse=0.034384`
  - `best_lag=-1`

## 最终结论
1. 这次“硬约束”并未把相位拉到 `0`，并且一旦启用 ranking 惩罚，整体排序就不占优。  
2. 最优参数自动退化到 `lambda_rank=0`，说明当前这类点对点线性结构对该约束不可有效吸收。  
3. 结论：问题已不是“损失项是否包含相位”，而是“模型结构是否能承载相位约束”；下一步应转向结构改造（非线性序列模型或显式时移参数可学习层）。  


