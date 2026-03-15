# 试验 19：二维隐含相位状态 Kalman（bias+drift）2026-03-15

## 本次尝试目标
在 `step18` 后进一步强化“隐含相位状态”建模：使用二维状态（偏置+漂移）并通过观测误差在线更新。

状态模型：
- `x_t = [bias_t, drift_t]^T`
- `x_t = A x_(t-1) + w_t`, `A=[[1,1],[0,phi]]`
- 观测：`obs_t = y_t - y_hat_raw_(t|t-1)`（实现上为 `y_anchor[i] - y_hat_raw[i-1]`）
- 预测修正：`y_hat_corr_(t+1|t) = y_hat_raw_(t+1|t) + gamma * bias_(t+1|t)`

扫描参数：
- `phi` 6 档
- `q_bias` 3 档
- `q_drift` 3 档
- `r_meas` 3 档
- `gamma` 9 档  
合计 1458 组。

输出：
- `scripts/us/stage_2026_03_15/experiment_us_kalman_phase_2state.py`
- `data/us/experiments/step19_kalman_phase_2state/kalman2_state_summary.json`
- `data/us/experiments/step19_kalman_phase_2state/kalman2_state_scan.csv`

## 结果与主版本对比
主版本（test）：
- `x_r2=0.742587`
- `x_rmse=0.034153`
- `best_lag=-1`
- `turn_mean_delay=0.4545`

step19 扫描结果：
- 1458 组全部 `eval_best_lag=-1`（无 `0`）

按 valid 选型最优：
- `phi=0, q_bias=1e-6, q_drift=1e-5, r=1e-3, gamma=0.4`
- test：
  - `x_r2=0.733057`（较主版本 `-0.009531`）
  - `x_rmse=0.034779`（较主版本 `+0.000626`）
  - `best_lag=-1`
  - `turn_mean_delay=0.3333`（较主版本改善）

按 test 最优：
- `phi=0, q_bias=1e-6, q_drift=1e-7, r=1e-3, gamma=0.2`
- test：
  - `x_r2=0.740030`
  - `x_rmse=0.034322`
  - `best_lag=-1`
  - 拐点指标基本与主版本接近（`turn_mean_delay=0.4545`）

## 最终结论
1. 二维隐含相位状态（Kalman）仍未把核心相位偏差从 `-1` 推到 `0`。  
2. 该方向在“延迟指标”上可获得一定改善，但仍伴随精度小幅损失，或与主版本几乎持平。  
3. 结论：问题已不只是“滤波器阶数”不足，下一步应把目标拆成“幅度预测 + 拐点概率/相位分类”双任务，避免单一 MSE 回归对相位结构的系统性偏置。  


