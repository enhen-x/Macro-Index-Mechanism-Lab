"""Build monthly panel for discount-rate vs cash-flow mechanism analysis.

Outputs a regression-ready panel with:
- one-month market return / excess return
- macro shock proxies (growth, inflation, policy, labor)
- regime/state indicators (high inflation, low growth, high volatility)
- control variables for LP estimation
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build US macro mechanism panel.")
    parser.add_argument(
        "--input",
        default="stages/us_discount-rate vs cash-flow/data/raw/us_monthly_panel.csv",
        help="Input monthly raw CSV.",
    )
    parser.add_argument(
        "--output",
        default="stages/us_discount-rate vs cash-flow/data/us_macro_mechanism_panel.csv",
        help="Output panel CSV.",
    )
    parser.add_argument(
        "--meta-output",
        default="stages/us_discount-rate vs cash-flow/data/us_macro_mechanism_panel_meta.json",
        help="Output metadata JSON.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--winsor-q", type=float, default=0.01, help="Two-sided winsor quantile.")
    return parser.parse_args()


def _to_float(v: str) -> float:
    t = (v or "").strip()
    if not t:
        return float("nan")
    try:
        return float(t)
    except ValueError:
        return float("nan")


def _read_csv(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty input: {path}")

    dates = [r["date"] for r in rows]
    cols = [k for k in rows[0].keys() if k != "date"]
    out: dict[str, np.ndarray] = {}
    for c in cols:
        out[c] = np.array([_to_float(r.get(c, "")) for r in rows], dtype=float)
    return dates, out


def _diff_log(x: np.ndarray, lag: int = 1) -> np.ndarray:
    out = np.full(len(x), np.nan, dtype=float)
    idx = np.arange(lag, len(x))
    out[idx] = np.log(x[idx]) - np.log(x[idx - lag])
    return out


def _diff_level(x: np.ndarray, lag: int = 1) -> np.ndarray:
    out = np.full(len(x), np.nan, dtype=float)
    idx = np.arange(lag, len(x))
    out[idx] = x[idx] - x[idx - lag]
    return out


def _zscore_train_full(x: np.ndarray, train_end: int) -> tuple[np.ndarray, float, float]:
    mu = float(np.nanmean(x[:train_end]))
    sd = float(np.nanstd(x[:train_end], ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0
    return (x - mu) / sd, mu, sd


def _winsor_train_full(x: np.ndarray, q: float, train_end: int) -> tuple[np.ndarray, float, float]:
    if q <= 0:
        return x.copy(), float("nan"), float("nan")
    lo = float(np.nanquantile(x[:train_end], q))
    hi = float(np.nanquantile(x[:train_end], 1.0 - q))
    out = np.clip(x, lo, hi)
    return out, lo, hi


def _quantile_flag_train_full(x: np.ndarray, p: float, train_end: int, high: bool) -> tuple[np.ndarray, float]:
    thr = float(np.nanquantile(x[:train_end], p))
    if high:
        flag = (x >= thr).astype(float)
    else:
        flag = (x <= thr).astype(float)
    return flag, thr


def _split_flags(n: int, train_ratio: float, valid_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(np.floor(n * train_ratio))
    valid_end = int(np.floor(n * (train_ratio + valid_ratio)))
    train_end = max(1, min(train_end, n - 2))
    valid_end = max(train_end + 1, min(valid_end, n - 1))
    idx = np.arange(n)
    is_train = (idx < train_end).astype(int)
    is_valid = ((idx >= train_end) & (idx < valid_end)).astype(int)
    is_test = (idx >= valid_end).astype(int)
    return is_train, is_valid, is_test


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[3]
    input_path = (root / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    output_path = (root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    meta_path = (root / args.meta_output).resolve() if not Path(args.meta_output).is_absolute() else Path(args.meta_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    dates, raw = _read_csv(input_path)
    n = len(dates)
    is_train, is_valid, is_test = _split_flags(n, args.train_ratio, args.valid_ratio)
    train_end = int(np.where(is_train == 1)[0][-1]) + 1

    log_spx = np.log(raw["sp500"])
    ret_1m = _diff_log(raw["sp500"], lag=1)
    rf_1m = raw["fed_funds"] / 100.0 / 12.0
    ex_ret_1m = ret_1m - rf_1m

    infl_yoy = _diff_log(raw["cpi"], lag=12)
    ip_mom = _diff_log(raw["indpro"], lag=1)
    unrate_chg = _diff_level(raw["unrate"], lag=1)
    policy_chg = _diff_level(raw["fed_funds"], lag=1) / 100.0

    shock_growth = ip_mom.copy()
    shock_inflation = _diff_level(infl_yoy, lag=1)
    shock_policy = -policy_chg.copy()
    shock_labor = -unrate_chg.copy()

    term_spread = raw["ust10y"] - raw["fed_funds"]
    dxy_mom = _diff_log(raw["dxy_broad"], lag=1)
    vix_log = np.log(raw["vix"])
    walcl_mom = _diff_log(raw["walcl"], lag=1)
    credit_proxy = _diff_log(raw["bus_loans"], lag=1)

    series = {
        "shock_growth": shock_growth,
        "shock_inflation": shock_inflation,
        "shock_policy": shock_policy,
        "shock_labor": shock_labor,
        "control_term_spread": term_spread,
        "control_dxy_mom": dxy_mom,
        "control_vix_log": vix_log,
        "control_infl_yoy": infl_yoy,
        "control_ip_mom": ip_mom,
        "control_walcl_mom": walcl_mom,
        "control_credit_mom": credit_proxy,
    }

    meta_stats: dict[str, dict[str, float]] = {}
    transformed: dict[str, np.ndarray] = {}
    for name, arr in series.items():
        w, lo, hi = _winsor_train_full(arr, q=args.winsor_q, train_end=train_end)
        z, mu, sd = _zscore_train_full(w, train_end=train_end)
        transformed[name] = z
        meta_stats[name] = {
            "winsor_lo": lo,
            "winsor_hi": hi,
            "train_mean": mu,
            "train_std": sd,
        }

    state_high_infl, thr_infl = _quantile_flag_train_full(infl_yoy, p=0.6, train_end=train_end, high=True)
    state_low_growth, thr_growth = _quantile_flag_train_full(ip_mom, p=0.4, train_end=train_end, high=False)
    state_high_vol, thr_vol = _quantile_flag_train_full(vix_log, p=0.6, train_end=train_end, high=True)

    keep = np.ones(n, dtype=bool)
    required = [
        log_spx,
        ret_1m,
        ex_ret_1m,
        transformed["shock_growth"],
        transformed["shock_inflation"],
        transformed["shock_policy"],
        transformed["shock_labor"],
        transformed["control_term_spread"],
        transformed["control_dxy_mom"],
        transformed["control_vix_log"],
        transformed["control_infl_yoy"],
        transformed["control_ip_mom"],
    ]
    for arr in required:
        keep &= np.isfinite(arr)

    fields = [
        "date",
        "log_sp500",
        "ret_1m",
        "excess_ret_1m",
        "shock_growth",
        "shock_inflation",
        "shock_policy",
        "shock_labor",
        "state_high_infl",
        "state_low_growth",
        "state_high_vol",
        "control_term_spread",
        "control_dxy_mom",
        "control_vix_log",
        "control_infl_yoy",
        "control_ip_mom",
        "control_walcl_mom",
        "control_credit_mom",
        "is_train",
        "is_valid",
        "is_test",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(n):
            if not keep[i]:
                continue
            writer.writerow(
                {
                    "date": dates[i],
                    "log_sp500": float(log_spx[i]),
                    "ret_1m": float(ret_1m[i]),
                    "excess_ret_1m": float(ex_ret_1m[i]),
                    "shock_growth": float(transformed["shock_growth"][i]),
                    "shock_inflation": float(transformed["shock_inflation"][i]),
                    "shock_policy": float(transformed["shock_policy"][i]),
                    "shock_labor": float(transformed["shock_labor"][i]),
                    "state_high_infl": int(state_high_infl[i]),
                    "state_low_growth": int(state_low_growth[i]),
                    "state_high_vol": int(state_high_vol[i]),
                    "control_term_spread": float(transformed["control_term_spread"][i]),
                    "control_dxy_mom": float(transformed["control_dxy_mom"][i]),
                    "control_vix_log": float(transformed["control_vix_log"][i]),
                    "control_infl_yoy": float(transformed["control_infl_yoy"][i]),
                    "control_ip_mom": float(transformed["control_ip_mom"][i]),
                    "control_walcl_mom": float(transformed["control_walcl_mom"][i]),
                    "control_credit_mom": float(transformed["control_credit_mom"][i]),
                    "is_train": int(is_train[i]),
                    "is_valid": int(is_valid[i]),
                    "is_test": int(is_test[i]),
                }
            )

    meta = {
        "input": str(input_path),
        "output": str(output_path),
        "rows_total": int(n),
        "rows_kept": int(np.sum(keep)),
        "split_counts_raw": {
            "train": int(np.sum(is_train)),
            "valid": int(np.sum(is_valid)),
            "test": int(np.sum(is_test)),
        },
        "winsor_q": args.winsor_q,
        "state_thresholds": {
            "infl_yoy_high_q60": thr_infl,
            "ip_mom_low_q40": thr_growth,
            "vix_log_high_q60": thr_vol,
        },
        "feature_stats": meta_stats,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[done] output panel: {output_path}")
    print(f"[done] rows kept: {int(np.sum(keep))}/{n}")
    print(
        "[done] state thresholds: "
        f"infl={thr_infl:.6g}, growth={thr_growth:.6g}, vol={thr_vol:.6g}"
    )


if __name__ == "__main__":
    main()

