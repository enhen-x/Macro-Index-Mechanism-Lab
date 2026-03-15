# US Data Loader Spec

This folder defines the US data intake spec for the two-level model
(global layer + country layer).

## Primary APIs

1. FRED API (main source)
2. BLS API (supplementary labor/inflation stats)
3. BEA API (supplementary GDP/NIPA)
4. Alpha Vantage (backup market price source)

## MVP Field Set

1. `sp500` (market state base series)
2. `fed_funds` (policy rate)
3. `ust10y` (long-end yield)
4. `walcl` (Fed balance sheet proxy)
5. `vix` (global risk shock)
6. `dxy_broad` (USD condition)
7. `bus_loans` (credit transmission proxy)
8. `cpi` (inflation)
9. `indpro` (growth proxy)
10. `unrate` (labor slack)

Notes on long history:

- `sp500`: primary from FRED `SP500`; auto-fallback to Stooq if FRED history starts later than requested `start_date`.
- `dxy_broad`: auto-append `DTWEXAFEGS` after `DTWEXM` end date (2019-12-31), with overlap scaling for continuity.
- `walcl`: auto-prepend `BOGMBASE` for pre-2003 history, with overlap scaling for continuity.

## Frequency Policy

- Target modeling frequency: monthly
- Daily/weekly series are aggregated to month-end or monthly average
- Macro releases keep official monthly values

## Transformation Policy (baseline)

- Level-to-gap for market state:
  - `Y_t = log(SP500_t) - trend_t`
- Rates and indices:
  - use first difference or log-difference where appropriate
- Standardization:
  - fit scaler on train only, then apply to valid/test

## Output Convention

- Raw pull output: `data/us/raw/*.csv`
- Aligned panel output: `data/us/us_monthly_panel.csv`

## Fetch Script

Run:

```powershell
$env:FRED_API_KEY="your_key"
python data_loader/us/fetch_us_fred.py --start-date 2005-01-01
```

Or save key once in local file (recommended for repeated runs):

1. Create `data_loader/us/fred_api_key.txt`
2. Put config in key=value format:
   - `api_key=YOUR_FRED_API_KEY`
   - `start_date=2005-01-01`
3. Run without passing `--api-key` or `--start-date`

```powershell
python data_loader/us/fetch_us_fred.py
```

