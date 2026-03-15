# US 两层动力模型试验日志（2026-03-15）

## 1. 目标与数据
- 目标：验证 US 月频面板下，二阶动力建模在一步预测 `Y(t+1)` 与方向判定上的有效性。
- 原始数据：`data/us/us_monthly_panel.csv`
- 回归面板：`data/us/us_regression_panel.csv`
- 当前核心离散化：中心差分构造 `a(t)=Y(t+1)+Y(t-1)-2Y(t)`，并在 `y_next` 目标上做一步预测评估。

## 2. 关键试验结论

### 2.1 线性阻尼相位问题（历史）
- 在 `phase_grid` 中，不同 `slow_log_window` 与 `lag1` 组合的最优相关均出现在 `lag=-1`，说明存在系统性相位提前/滞后问题。
- 代表结果（w3_lag1_on）：
  - `lag0_corr=0.8661`
  - `best_lag=-1, best_lag_corr=0.9333`

### 2.2 AR(1) 误差修正（历史）
- 对一步预测误差做 AR(1) 修正，`phi≈-0.2177`。
- test 指标改善（x 预测）：
  - 无修正：`x_r2=0.7050, x_rmse=0.03656`
  - AR(1)：`x_r2=0.7345, x_rmse=0.03468`
- 但相位主问题并未根治。

### 2.3 时变参数（历史）
- rolling / expanding 已验证可跑通，但相位最优仍多在 `lag=-1`。
- test 代表指标：
  - rolling dynamic：`r2=0.7087, rmse=0.0409, best_lag=-1`
  - expanding dynamic：`r2=0.7559, rmse=0.0374, best_lag=-1`

### 2.4 非线性阻尼（当前）
- 新增阻尼形式：`nonlinear_absv`（项为 `|v|v`，即二次阻尼）。
- 训练估计结果对比：
  - 线性：`c=0.0001(贴下界), c_nl=0, train_r2=0.3692`
  - 非线性：`c=0.4879, c_nl=0.03556, train_r2=0.4679`

- test 指标对比（同配置，`ridge=0.05`）：

| 模型 | r2 | rmse | direction_accuracy | x_r2 | x_rmse |
|---|---:|---:|---:|---:|---:|
| linear | 0.4008 | 0.03561 | 0.5652 | 0.7061 | 0.03649 |
| nonlinear_absv | 0.4732 | 0.03339 | 0.7391 | 0.7426 | 0.03415 |

- 结论：非线性阻尼在当前数据上显著优于线性阻尼，并且 `c` 不再失效（不再贴下界）。

### 2.5 相位修复专项（step13/13b/14）
- step13（as-of 对齐）：
  - `best_lag` 仍为 `-1`，但 `turn_mean_delay` 从 `0.4545` 降到 `0.1667`。
  - 代价是 `x_r2` 下滑（约 `-0.0238`）。
- step13b（as-of 滞后配置扫描）：
  - 5 组配置均未把 `best_lag` 推到 `0`。
  - 最优整体仍是 `all_0`（等价主版本）。
- step14（显式 delay 参数化，27 组扫描）：
  - 按 test 最优仍退化为 `d=(0,0,0)`（即主版本）。
  - 按 valid 选出的延迟组合可改善拐点延迟，但 `x_r2/x_rmse` 变差。
- 阶段结论：仅靠“发布时间/分组延迟”类手段无法决定性修复相位偏差，需进入更强结构改造。

### 2.6 step15 显式相位状态（phase-state）
- 在主模型输出上增加状态递推：
  - `z_t = rho*z_(t-1) + kappa*(y_hat_raw_t-y_t) + eta*(y_t-y_(t-1))`
  - `y_hat_state_t = y_hat_raw_t + z_t`
- 扫描结果（4420 组）：
  - `sel_best_lag` 仅出现 `-1` 或 `-2`，没有 `0`。
  - 最优参数：`rho=0.55, kappa=-0.6, eta=-0.1`（按 valid 选型）。
- test 对比主版本：
  - `x_r2: 0.7426 -> 0.6852`（下降）
  - `x_rmse: 0.03415 -> 0.03777`（变差）
  - `best_lag: -1 -> -1`（未改善）
- 结论：显式状态参数化本身仍不足以根治相位问题，下一步需用“状态+观测联合估计”或改目标动态方程。

### 2.7 step16 可观测误差状态（Kalman-style）
- 状态更新：
  - `z_pred_t = rho*z_(t-1)`
  - `obs_t = y_t - y_hat_raw_(t-1)`
  - `z_t = z_pred_t + K*(obs_t - z_pred_t)`
  - `y_hat_state_(t+1|t) = y_hat_raw_(t+1|t) + gamma*z_t`
- 扫描结果（3520 组）：
  - `sel_best_lag` 全部为 `-1`，没有 `0`。
  - 最优参数：`rho=0.95, K=0.1, gamma=0.4`。
- test 对比主版本：
  - `x_r2: 0.7426 -> 0.7351`（小幅下降）
  - `x_rmse: 0.03415 -> 0.03465`（小幅变差）
  - `best_lag: -1 -> -1`（未改善）
  - `turn_mean_delay: 0.4545 -> 0.3333`（有改善）
- 结论：对拐点延迟有帮助，但无法解决核心相位偏差；相位问题更可能来自目标定义与动力方程错配。

### 2.8 step17 `ΔY` 目标动力回归 + 显式 phase_gain
- 目标改写：
  - 拟合 `delta_{t+1}=Y_{t+1}-Y_t`
  - 用 `Y_hat_{t+1}=Y_t + delta_hat_{t+1} + phase_gain*delta_t` 还原
- 扫描设置：
  - `alpha`（11 档）× `phase_gain`（9 档）= 99 组
  - `train` 拟合、`valid` 选型、`test` 评估
- 结果：
  - `eval_best_lag` 在 99 组里全部为 `-1`（无 `0`）
  - 最优组合（按 valid）：`alpha=10, phase_gain=-0.1`
  - test：`x_r2=0.7194`, `x_rmse=0.03452`, `best_lag=-1`
  - `turn_mean_delay` 由 `0.4545` 降至 `0.1538`，但精度仍低于主版本
- 结论：目标切换到 `ΔY` + 显式相位项后，延迟指标改善但核心相位偏差仍未根治。

### 2.9 step18 递推状态 + 多步一致性约束
- 模型：
  - `delta_{t+1} = rho*delta_t + r_theta(s_t)`
  - `Y_hat_{t+1} = Y_t + delta_{t+1} + phase_gain*delta_t`
  - `s_t=[Y_t, delta_t, accel_t, u_t]`
- 选型引入多步一致性（h=1/2/3，权重 1.0/0.7/0.5）。
- 扫描规模：`11(alpha) × 11(rho) × 9(phase_gain) = 1089`。
- 结果：
  - `eval_best_lag` 1089 组全部为 `-1`（无 `0`）。
  - 按 valid 最优：`alpha=10, rho=0.0, phase_gain=-0.3`，test `x_r2=0.7049`, `x_rmse=0.03540`, `best_lag=-1`。
  - 多步 RMSE（该最优）：`h1=0.03540, h2=0.05072, h3=0.06001`。
- 结论：多步一致性可改善延迟指标，但仍不能根治相位偏差，且精度仍低于主版本。

### 2.10 step19 二维隐含相位状态 Kalman（bias+drift）
- 模型：
  - `x_t=[bias_t, drift_t]^T`
  - `x_t = A x_(t-1) + w_t`, `A=[[1,1],[0,phi]]`
  - 观测 `obs_t = y_t - y_hat_raw_(t|t-1)`，修正 `y_hat_corr_(t+1|t)=y_hat_raw_(t+1|t)+gamma*bias_(t+1|t)`
- 扫描规模：1458 组。
- 结果：
  - `eval_best_lag` 1458 组全部为 `-1`（无 `0`）。
  - 按 valid 最优：`phi=0, q_bias=1e-6, q_drift=1e-5, r=1e-3, gamma=0.4`
    - test：`x_r2=0.7331`, `x_rmse=0.03478`, `best_lag=-1`
    - `turn_mean_delay` 从 `0.4545` 降至 `0.3333`
  - 按 test 最优：`x_r2=0.7400`, `x_rmse=0.03432`，但 `best_lag` 仍 `-1`
- 结论：提高状态维度后依然无法根治相位偏差，仅能在延迟指标与精度之间做小幅折中。

### 2.11 step20 双任务后处理（幅度 + 相位）
- 结构：
  - 幅度头：回归 `residual = y_true - y_hat_raw`
  - 相位头：分类 `p_turn`、`p_up`
  - 融合：`delta_final = (1-lambda*g)*delta_amp + lambda*g*delta_target`
- 扫描：80 组（`lambda × tau`）。
- 结果：
  - `eval_best_lag` 80 组全部 `-1`（无 `0`）。
  - 按 valid 最优：`lambda=1.5, tau=0.5`，test `x_r2=0.7127`, `x_rmse=0.03608`, `best_lag=-1`。
  - 按 test 最优退化到 `lambda=0`（不启用相位融合），`x_r2=0.7393`, `x_rmse=0.03437`, `best_lag=-1`。
- 结论：当前逐点相位分类头对相位修复贡献有限，下一步需尝试“序列状态机/持续态分类”。

### 2.12 step21 序列相位状态机（持续态过滤）
- 在 step20 基础上加入 `p_up` 的持续态滤波（参数 `stay`），构建序列相位状态。
- 扫描规模：400 组（`lambda × tau × stay`）。
- 结果：
  - `eval_best_lag` 400 组全部 `-1`（无 `0`）。
  - 按 valid 最优：`lambda=1.4, tau=0.5, stay=0.98`，test `x_r2=0.7187`, `x_rmse=0.03570`, `best_lag=-1`。
  - 按 test 最优：`lambda=0.2, tau=0.4, stay=0.8`，test `x_r2=0.7398`, `x_rmse=0.03434`, `best_lag=-1`。
- 结论：序列过滤提高了平滑性与部分延迟指标，但仍无法把核心相位偏差推回 `0`。

### 2.13 step22 训练时显式相位损失
- 训练目标改为：
  - `L = L_amp + lambda_phase*L_phase + lambda_turn*L_turn + ridge`
  - 其中 `L_phase=MSE(diff(y_hat)-diff(y_true))`
- 扫描规模：180 组。
- 结果：
  - `eval_best_lag` 分布：`-1`(150), `+3`(18), `-3`(12)，仍无 `0`。
  - 按 valid 最优：`lambda_phase=0.5, lambda_turn=0.2, ridge=0.05`
    - test：`x_r2=0.7216`, `x_rmse=0.03438`, `best_lag=-1`
    - `turn_mean_delay=0.10`（显著降低）
  - 按 test 最优：`x_r2=0.7226`, `x_rmse=0.03432`, `best_lag=-1`
- 结论：相位损失能改善延迟，但出现过补偿（`±3` lag）且仍无法根治核心相位偏差。

### 2.14 step23 lag0-vs-lag-1 Ranking 硬约束
- 在 step22 基础上新增硬约束：
  - `L_rank = relu(margin + mse_lag0 - mse_lag-1)`
  - 目标是训练时直接强迫 lag0 优于 lag-1
- 扫描规模：160 组。
- 结果：
  - `eval_best_lag` 160 组全部为 `-1`（无 `0`）。
  - 在 `lambda_rank>0` 的 128 组中也全部为 `-1`。
  - 最优解（valid/test）均退化到 `lambda_rank=0, margin=0`。
- 结论：当前线性点预测结构对该 ranking 约束不可有效吸收，继续在同类损失上微调意义很小，需转向结构改造。

### 2.15 step24 结构内生相位状态（主方程耦合）
- 方程：
  - `delta_base_t = f_amp(z_t)`
  - `d_t = rho_d*d_(t-1) + g_phase(z_t, delta_base_t)`
  - `y_hat_(t+1)=y_t + delta_base_t*(1 + gamma*d_t)`
- 扫描规模：36 组。
- 结果：
  - `eval_best_lag` 36 组全部 `-1`（无 `0`）。
  - 按 valid 最优：`rho=0.3, phase_ridge=0.1, gamma=1.0`
    - test：`x_r2=0.7056`, `x_rmse=0.03536`, `best_lag=-1`
    - `turn_mean_delay=0.1818`（改善）
  - 按 test 最优也仍为 `best_lag=-1`，且精度低于主版本。
- 结论：结构方向正确，但当前线性化相位状态表达力不足，需引入非线性状态方程或门控转移。

### 2.16 step25 结构内生相位状态（非线性门控）
- 在 step24 上新增：
  - 非线性相位驱动 `f_phase_nl([x, x^2, tanh(x)])`
  - 门控递推 `rho_eff = rho_low*(1-gate)+rho_high*gate`
- 扫描规模：48 组。
- 结果：
  - `eval_best_lag` 48 组全部 `-1`（无 `0`）。
  - 状态活跃度明显上升（`mean_abs_d_state` 约 0.65），但出现过补偿。
  - 按 valid 最优：test `x_r2=0.6073`, `x_rmse=0.04084`, `best_lag=-1`（精度显著劣化）。
  - 按 test 最优：`x_r2=0.6617`, `x_rmse=0.03791`, 仍低于主版本。
- 结论：非线性门控并未修复相位，且精度损失大，继续该方向风险较高。

### 2.17 step26 显式分数位移层（幅相解耦）
- 结构：
  - `y_shift_t = y_raw_t + tau * (y_raw_t - y_raw_(t-1))`
  - 可选步长裁剪 `step_clip_std` 抑制过补偿。
- 扫描规模：204 组（`tau=-0.5..2.0`，步长 `0.05`，`step_clip_std∈{0,2,3,4}`）。
- 结果：
  - `eval_best_lag` 分布：`-1`(126), `0`(78)。
  - 按 valid 最优：`tau=-0.45, clip_std=3`
    - test：`x_r2=0.7134`, `x_rmse=0.03604`, `best_lag=-1`。
  - 按 test 精度最优：`tau=-0.2, clip_std=2`
    - test：`x_r2=0.7596`, `x_rmse=0.03301`, `best_lag=-1`（精度优于主版本但相位不变）。
  - 在 `best_lag=0` 子集的最优为：`tau=1.0, clip_std=0`
    - test：`x_r2=0.0910`, `x_rmse=0.06418`（精度显著崩塌）。
- 结论：显式分数位移层存在“相位-精度”硬冲突，能拉到 `lag=0` 但代价极高；在可用精度区间仍稳定 `lag=-1`，说明核心问题不是单一常数时间平移。

## 3. 本次清理动作
已删除历史试验目录/文件：
- `data/us/ar1_correction`
- `data/us/distribution_analysis`
- `data/us/phase_grid`
- `data/us/plot`
- `data/us/plot_test`
- `data/us/plot_valid`
- `data/us/time_varying_compare`
- `data/us/y_next_target`
- `data/us/phase_test_no_lag1_estimation.json`
- `data/us/y_next_target_estimation.json`

## 4. 当前保留结果（用于后续）
- 主数据：
  - `data/us/us_monthly_panel.csv`
  - `data/us/us_regression_panel.csv`
  - `data/us/us_regression_panel_transform_meta.json`
- 当前估计：
  - `data/us/us_ols_estimation.json`
- 非线性阻尼对比：
  - `data/us/nonlinear_trial/linear_estimation.json`
  - `data/us/nonlinear_trial/nonlinear_absv_estimation.json`
  - `data/us/nonlinear_trial/plot_test_linear/fit_metrics.json`
  - `data/us/nonlinear_trial/plot_test_nonlinear_absv/fit_metrics.json`

## 5. 当前建议
- 下一步可将 `nonlinear_absv` 方案切换为主版本，并在 valid/test 进行固定流程复现。
- 需同时监控条件数（当前非线性版条件数较高），建议后续做正则强度与特征裁剪联合稳健性检查。

