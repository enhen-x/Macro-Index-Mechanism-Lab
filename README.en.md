# MacroDynamics-System-ID

Modeling stock market indices as a forced damped dynamical system.

[Chinese](./README.zh-CN.md) | English

## 1. Project Overview

`MacroDynamics-System-ID` is an experimental quantitative research project that maps market index behavior to a second-order control system, then uses system identification and state-space modeling to estimate interpretable parameters.

Core idea:
- Treat index movement as a dynamic response to macro signals.
- Use identifiable parameters to represent market behavior:
  - `m` (inertia / mass)
  - `c` (damping)
  - `k` (stiffness / mean-reversion strength)
- Build a framework that can be tested across markets (e.g., CSI 300, S&P 500, Nikkei 225).

## 2. Theoretical Model

Second-order forced damped system:

```math
m\frac{d^2x}{dt^2} + c\frac{dx}{dt} + k(x - x_0) = F_{macro}(t) + \eta(t)
```

Definitions:
- `x(t)`: index level (or transformed series such as log-price)
- `m`: inertia, representing resistance to rapid trend changes
- `c`: damping, representing friction from policy/risk-control/market microstructure
- `k`: stiffness, representing pullback toward an equilibrium level
- `F_macro(t)`: external forcing from macro variables
- `\eta(t)`: unobserved noise / residual disturbance

## 3. Methodology

### 3.1 System Identification

Estimate `[m, c, k]` and external forcing effects from historical input-output data:
- Inputs: macro signals (e.g., LPR, M2 growth, FX index, policy proxies)
- Output: index return / log-price dynamics
- Candidate techniques:
  - Least Squares / Regularized Regression
  - Kalman Filter / Extended Kalman Filter
  - Rolling or regime-aware estimation

### 3.2 State-Space Representation

Transform the model for filtering, forecasting, and control:

```math
\begin{bmatrix}
\dot{x}_1 \\
\dot{x}_2
\end{bmatrix}
=
\begin{bmatrix}
0 & 1 \\
-\frac{k}{m} & -\frac{c}{m}
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
+
\begin{bmatrix}
0 \\
\frac{1}{m}
\end{bmatrix}
u(t)
```

Where:
- `x_1`: level state
- `x_2`: velocity/momentum state
- `u(t)`: macro forcing input

### 3.3 Cross-Market Validation

Validate robustness by comparing parameter stability and predictive quality across:
- China A-share representative index
- US large-cap index
- Japan representative index

Evaluation dimensions:
- out-of-sample fit
- turning-point sensitivity
- regime shift adaptability

## 4. Roadmap

- [ ] Phase 1: Data Pipeline
  - Build standardized loaders for price + macro series
  - Handle frequency alignment, missing values, and revisions
- [ ] Phase 2: Baseline Identification
  - Implement static and rolling parameter estimation
  - Build baseline diagnostics and residual analysis
- [ ] Phase 3: Adaptive Modeling
  - Introduce time-varying parameters (e.g., EKF)
  - Support policy/regime-switching sensitivity
- [ ] Phase 4: Physics-Informed ML Integration
  - Explore PINNs or hybrid model constraints
  - Compare with purely data-driven baselines

## 5. Suggested Repository Structure

```text
core/                 # dynamic system equations and simulation engine
data_loader/          # market and macro data ingestion + preprocessing
identification/       # parameter estimation and filtering
backtest/             # strategy evaluation and diagnostics
notebooks/            # research experiments and visualization
tests/                # unit/integration tests
```

## 6. Getting Started (Draft)

Since the project is in an early stage, a minimal setup is:

1. Create and activate a Python environment.
2. Install dependencies (to be finalized in `requirements.txt` or `pyproject.toml`).
3. Prepare sample data in a unified format.
4. Run baseline identification notebook/scripts.

## 7. Contribution

Contributions are welcome via:
- Issue reports (bugs, ideas, validation concerns)
- Pull requests (modeling, data engineering, testing, documentation)

Recommended PR checklist:
- clear problem statement
- reproducible experiment settings
- before/after metrics or visual diagnostics

## 8. License

MIT License.

