# MacroDynamics-System-ID

An interpretable macro-finance research workspace focused on explaining how macro variables affect broad equity indices.

[Chinese](./README.zh-CN.md) | English

## 1. Project Scope

`MacroDynamics-System-ID` is centered on **mechanism discovery**, not on a single modeling paradigm.

Core objectives:
- identify transmission channels from macro variables to broad indices (e.g., S&P500);
- quantify how those channels change across macro/market regimes;
- maintain a reproducible, testable, and extensible research workflow.

## 2. Method Scope (Not Single-Model Bound)

The project supports multiple explainable approaches, including:
- dynamical-system formulations (e.g., vibration-equation / forced-damped models),
- state-dependent local projections (SD-LP),
- discount-rate vs cash-flow decomposition,
- other structured econometric methods (regime switching, event windows, robust regressions).

Notes:
- vibration-equation modeling is preserved as an archived exploration stage;
- current direction is mechanism-first, not equation-first.

## 3. Active and Archived Stages

Current active stage:
- `stages/us_discount-rate vs cash-flow/`

Archived stage example:
- `stages/us_vibration_equation_exploration_2026-03-15/`

## 4. Repository Structure (Recommended Entry Order)

- `stages/`: stage-based scripts, data, and docs (primary entry point).
- `data_loader/`: raw data fetching and preprocessing utilities.
- `identification/`: parameter estimation and identification modules.
- `core/`: shared modeling and computation components.
- `tests/`: test suites.
- `docs/`: global documentation and stage index notes.

## 5. Quick Start

1. Create and activate a Python environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Enter a target stage (prefer `stages/us_discount-rate vs cash-flow/`).
4. Follow that stage's `docs/` and `scripts/README.md`.

## 6. Contribution

Contributions are welcome via:
- Issues: problems, hypotheses, mechanism questions;
- PRs: scripts, replications, docs, and tests.

Recommended PR checklist:
- clear objective and hypothesis,
- reproducible data/parameter setup,
- before/after comparison or figure-based evidence.

## 7. License

MIT License.
