# Scripts

This folder stores executable scripts for the US discount-rate vs cash-flow stage.

## Implemented
- `build_us_macro_mechanism_panel.py`
  - Build a monthly mechanism panel from raw macro-market data.
  - Creates shocks, states, controls, and train/valid/test flags.
- `run_us_sd_lp.py`
  - Run state-dependent local projections (baseline + interaction model).
  - Uses Newey-West robust standard errors.

## Quick Run
```bash
python "stages/us_discount-rate vs cash-flow/scripts/build_us_macro_mechanism_panel.py"
python "stages/us_discount-rate vs cash-flow/scripts/run_us_sd_lp.py" --split all --h-min 1 --h-max 12 --nw-lag 3 --min-obs 80
```
