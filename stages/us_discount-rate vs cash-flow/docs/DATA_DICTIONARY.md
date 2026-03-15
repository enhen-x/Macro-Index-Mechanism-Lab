# Data Dictionary (v1)

Panel file: `data/us_macro_mechanism_panel.csv`

## Core target fields
- `log_sp500`: log level of S&P500.
- `ret_1m`: one-month log return.
- `excess_ret_1m`: one-month excess return (ret - fed_funds/12).

## Shock fields (z-scored on train sample)
- `shock_policy`: `-(fed_funds_t - fed_funds_{t-1})`.
- `shock_inflation`: change of YoY inflation proxy.
- `shock_growth`: monthly log-change of industrial production.
- `shock_labor`: `-(unrate_t - unrate_{t-1})`.

## State fields (binary)
- `state_high_infl`: 1 if inflation proxy >= train q60.
- `state_low_growth`: 1 if growth proxy <= train q40.
- `state_high_vol`: 1 if log(VIX) >= train q60.

## Control fields (z-scored on train sample)
- `control_term_spread`: `ust10y - fed_funds`.
- `control_dxy_mom`: monthly log-change of DXY broad.
- `control_vix_log`: log(VIX).
- `control_infl_yoy`: YoY inflation proxy.
- `control_ip_mom`: industrial production monthly change.
- `control_walcl_mom`: monthly log-change of WALCL.
- `control_credit_mom`: monthly log-change of business loans.

## Split flags
- `is_train`, `is_valid`, `is_test`.

Metadata file: `data/us_macro_mechanism_panel_meta.json`
