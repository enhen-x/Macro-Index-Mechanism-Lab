# 试验 21：序列相位状态机（持续态过滤）2026-03-15

## 本次尝试目标
针对 step20“逐点分类头不稳定”的问题，引入序列状态过滤：

- 先训练与 step20 相同的幅度头与分类头（`p_turn`, `p_up_inst`）
- 再对 `p_up_inst` 做持续态滤波（带状态留存概率 `stay`）得到 `p_up_filt`
- 用 `p_up_filt` 参与方向融合，避免逐点抖动导致相位噪声

核心思路：
1. 幅度头：`y_hat_amp = y_hat_raw + residual_hat`
2. 相位门控：`g = clip((p_turn - tau)/(1-tau), 0, 1)`
3. 序列状态：`p_up_filt = BayesFilter(p_up_inst, stay)`
4. 最终融合：同 step20，但方向由 `p_up_filt` 给出

输出：
- `scripts/us/stage_2026_03_15/experiment_us_sequence_phase_state.py`
- `data/us/experiments/step21_sequence_phase_state/sequence_phase_summary.json`
- `data/us/experiments/step21_sequence_phase_state/sequence_phase_scan.csv`

## 结果与主版本对比
主版本（test）：
- `x_r2=0.742587`
- `x_rmse=0.034153`
- `best_lag=-1`
- `turn_mean_delay=0.4545`

step21 扫描结果：
- 组合数：400（`lambda × tau × stay`）
- `eval_best_lag`：400 组全部为 `-1`（无 `0`）

按 valid 选型最优：
- `lambda=1.4, tau=0.5, stay=0.98`
- test：
  - `x_r2=0.718716`（较主版本 `-0.023871`）
  - `x_rmse=0.035701`（较主版本 `+0.001548`）
  - `best_lag=-1`
  - `turn_mean_delay=0.3846`（改善）

按 test 最优：
- `lambda=0.2, tau=0.4, stay=0.8`
- test：
  - `x_r2=0.739801`
  - `x_rmse=0.034337`
  - `best_lag=-1`
  - `turn_mean_delay=0.3077`

## 最终结论
1. 引入序列持续态后，指标稳定性有所提升，但核心相位偏差依然固定在 `best_lag=-1`。  
2. 延迟指标可继续改善，但以整体精度小幅下降为代价。  
3. 结论：仅靠输出端相位融合难以根治问题；下一步应改为“结构内生相位”路线（例如在主动力方程中同时拟合 `Y` 与相位状态转移）。  


