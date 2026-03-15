"""Build regression-ready monthly panel for US forced-damped estimation.

This script constructs:
  Y_t = log(sp500_t) - trend_t
  v_t = (Y_{t+1} - Y_{t-1}) / (2 * dt)
  a_t = (Y_{t+1} - 2Y_t + Y_{t-1}) / (dt^2)

It supports:
- optional smoothing on Y before derivatives
- optional lag-1 macro features
- optional interaction features
- train-fitted robust transforms (asinh-z / rank-gauss) for heavy-tailed variables
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_TRANSFORM_COLUMNS = [
    "g_vix",
    "c_liquidity_ops",
    "c_credit",
    "c_growth",
    "c_cpi",
    "c_unrate",
]


# Acklam inverse-normal approximation coefficients.
_A = [
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
]
_B = [
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
]
_C = [
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
]
_D = [
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
]


def norm_ppf(p: np.ndarray) -> np.ndarray:
    """Approximate inverse CDF of standard normal for p in (0,1)."""
    p = np.asarray(p, dtype=float)
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)

    out = np.empty_like(p)
    plow = 0.02425
    phigh = 1.0 - plow

    low_mask = p < plow
    high_mask = p > phigh
    mid_mask = (~low_mask) & (~high_mask)

    if np.any(low_mask):
        q = np.sqrt(-2.0 * np.log(p[low_mask]))
        num = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
        den = (((( _D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
        out[low_mask] = num / den

    if np.any(mid_mask):
        q = p[mid_mask] - 0.5
        r = q * q
        num = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q
        den = (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
        out[mid_mask] = num / den

    if np.any(high_mask):
        q = np.sqrt(-2.0 * np.log(1.0 - p[high_mask]))
        num = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
        den = (((( _D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
        out[high_mask] = -(num / den)

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build US regression panel from monthly raw panel.")
    parser.add_argument("--input", default="data/us/us_monthly_panel.csv", help="Input monthly panel CSV.")
    parser.add_argument("--output", default="data/us/us_regression_panel.csv", help="Output regression panel CSV.")
    parser.add_argument("--country", default="US", help="Country label written to output.")
    parser.add_argument("--dt", type=float, default=1.0, help="Time step used in finite differences.")
    parser.add_argument(
        "--diff-mode",
        choices=["central", "causal"],
        default="central",
        help="Derivative discretization mode for v/a.",
    )
    parser.add_argument(
        "--slow-log-window",
        type=int,
        default=3,
        help=(
            "Window for slow macro log-change features (walcl/bus_loans/cpi/indpro). "
            "1 means standard monthly log-diff, 3 means log(x_t)-log(x_{t-3})."
        ),
    )
    parser.add_argument(
        "--trend-method",
        choices=["hp", "linear", "none"],
        default="hp",
        help="How to build trend in Y_t = log(sp500)-trend.",
    )
    parser.add_argument("--hp-lambda", type=float, default=129600.0, help="HP filter lambda (monthly default).")
    parser.add_argument(
        "--smooth-method",
        choices=["none", "ma", "ema"],
        default="none",
        help="Smoothing method for Y before central differences.",
    )
    parser.add_argument("--smooth-window", type=int, default=5, help="Window size for moving average smoothing.")
    parser.add_argument("--ema-alpha", type=float, default=0.3, help="EMA alpha in (0,1] when smooth-method=ema.")
    parser.add_argument(
        "--winsor-quantile",
        type=float,
        default=0.01,
        help="Two-sided winsor quantile for v/a (0 disables).",
    )
    parser.add_argument(
        "--add-lag1-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add lag-1 copy for each g_*/c_* base feature.",
    )
    parser.add_argument(
        "--interaction-mode",
        choices=["none", "with_y", "gxc", "with_y_gxc"],
        default="none",
        help="How to build interaction terms from base features.",
    )
    parser.add_argument(
        "--feature-transform",
        choices=["none", "asinh_z", "rank_gauss"],
        default="rank_gauss",
        help="Feature transform fitted on train split only.",
    )
    parser.add_argument(
        "--transform-columns",
        default=",".join(DEFAULT_TRANSFORM_COLUMNS),
        help="Comma-separated columns to transform (e.g. g_vix,c_credit).",
    )
    parser.add_argument(
        "--transform-tail-clip-q",
        type=float,
        default=0.005,
        help="Train-fitted two-sided clipping quantile before transform (0 disables).",
    )
    parser.add_argument(
        "--transform-meta",
        default="",
        help="Optional transform meta JSON path. Default: <output>_transform_meta.json",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio by time.")
    parser.add_argument("--valid-ratio", type=float, default=0.15, help="Validation split ratio by time.")
    return parser.parse_args()


def parse_column_list(raw: str) -> list[str]:
    cols = [x.strip() for x in raw.split(",") if x.strip()]
    seen: set[str] = set()
    out: list[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def to_float(value: str) -> float:
    text = (value or "").strip()
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def read_panel(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty input file: {path}")

    fields = [k for k in rows[0].keys() if k != "date"]
    dates = [r["date"] for r in rows]
    data: dict[str, np.ndarray] = {}
    for field in fields:
        data[field] = np.array([to_float(r.get(field, "")) for r in rows], dtype=float)
    return dates, data


def first_diff(x: np.ndarray) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    if len(x) < 2:
        return out
    lhs = x[1:]
    rhs = x[:-1]
    valid = np.isfinite(lhs) & np.isfinite(rhs)
    out[1:][valid] = lhs[valid] - rhs[valid]
    return out


def lag1(x: np.ndarray) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    if len(x) < 2:
        return out
    out[1:] = x[:-1]
    return out


def log_diff(x: np.ndarray) -> np.ndarray:
    lx = np.full_like(x, np.nan, dtype=float)
    valid = np.isfinite(x) & (x > 0)
    lx[valid] = np.log(x[valid])
    return first_diff(lx)


def log_change_window(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be >= 1")
    if window == 1:
        return log_diff(x)

    lx = np.full_like(x, np.nan, dtype=float)
    valid = np.isfinite(x) & (x > 0)
    lx[valid] = np.log(x[valid])

    out = np.full_like(x, np.nan, dtype=float)
    if len(x) <= window:
        return out
    lead = lx[window:]
    lag = lx[:-window]
    ok = np.isfinite(lead) & np.isfinite(lag)
    out[window:][ok] = lead[ok] - lag[ok]
    return out


def central_velocity(y: np.ndarray, dt: float) -> np.ndarray:
    out = np.full_like(y, np.nan, dtype=float)
    if len(y) < 3:
        return out
    lead = y[2:]
    lag = y[:-2]
    valid = np.isfinite(lead) & np.isfinite(lag)
    out[1:-1][valid] = (lead[valid] - lag[valid]) / (2.0 * dt)
    return out


def central_acceleration(y: np.ndarray, dt: float) -> np.ndarray:
    out = np.full_like(y, np.nan, dtype=float)
    if len(y) < 3:
        return out
    lead = y[2:]
    mid = y[1:-1]
    lag = y[:-2]
    valid = np.isfinite(lead) & np.isfinite(mid) & np.isfinite(lag)
    out[1:-1][valid] = (lead[valid] - 2.0 * mid[valid] + lag[valid]) / (dt * dt)
    return out


def causal_velocity(y: np.ndarray, dt: float) -> np.ndarray:
    out = np.full_like(y, np.nan, dtype=float)
    if len(y) < 2:
        return out
    cur = y[1:]
    prev = y[:-1]
    valid = np.isfinite(cur) & np.isfinite(prev)
    out[1:][valid] = (cur[valid] - prev[valid]) / dt
    return out


def causal_acceleration(y: np.ndarray, dt: float) -> np.ndarray:
    out = np.full_like(y, np.nan, dtype=float)
    if len(y) < 3:
        return out
    cur = y[2:]
    prev = y[1:-1]
    prev2 = y[:-2]
    valid = np.isfinite(cur) & np.isfinite(prev) & np.isfinite(prev2)
    out[2:][valid] = (cur[valid] - 2.0 * prev[valid] + prev2[valid]) / (dt * dt)
    return out


def hp_filter(y: np.ndarray, lam: float) -> np.ndarray:
    n = len(y)
    if n < 3:
        return y.copy()
    k = np.zeros((n - 2, n), dtype=float)
    idx = np.arange(n - 2)
    k[idx, idx] = 1.0
    k[idx, idx + 1] = -2.0
    k[idx, idx + 2] = 1.0
    a = np.eye(n, dtype=float) + lam * (k.T @ k)
    return np.linalg.solve(a, y)


def linear_trend(y: np.ndarray) -> np.ndarray:
    n = len(y)
    t = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(t, y, deg=1)
    return slope * t + intercept


def build_trend(log_sp500: np.ndarray, method: str, hp_lambda: float) -> np.ndarray:
    if method == "none":
        return np.zeros_like(log_sp500, dtype=float)
    if method == "linear":
        return linear_trend(log_sp500)
    return hp_filter(log_sp500, lam=hp_lambda)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    out = np.convolve(padded, kernel, mode="valid")
    return out.astype(float)


def ema_smooth(x: np.ndarray, alpha: float) -> np.ndarray:
    if not (0 < alpha <= 1):
        raise ValueError("ema alpha must be in (0,1]")
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def smooth_series(y: np.ndarray, method: str, window: int, alpha: float) -> np.ndarray:
    if method == "none":
        return y.copy()
    if method == "ma":
        return moving_average(y, window=window)
    return ema_smooth(y, alpha=alpha)


def winsorize(x: np.ndarray, q: float) -> np.ndarray:
    if q <= 0:
        return x.copy()
    if q >= 0.5:
        raise ValueError("winsor quantile must be < 0.5")
    out = x.copy()
    finite = np.isfinite(out)
    if not np.any(finite):
        return out
    lo, hi = np.quantile(out[finite], [q, 1.0 - q])
    out[finite] = np.clip(out[finite], lo, hi)
    return out


def require_columns(data: dict[str, np.ndarray], required: Iterable[str]) -> None:
    missing = [c for c in required if c not in data]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def robust_loc_scale(x: np.ndarray) -> tuple[float, float]:
    v = x[np.isfinite(x)]
    if len(v) == 0:
        return 0.0, 1.0
    loc = float(np.median(v))
    mad = float(np.median(np.abs(v - loc)))
    scale = 1.4826 * mad
    if (not np.isfinite(scale)) or scale <= 1e-12:
        sd = float(np.std(v))
        scale = sd if np.isfinite(sd) and sd > 1e-12 else 1.0
    return loc, scale


def apply_asinh_z(x: np.ndarray, center: float, scale: float) -> np.ndarray:
    out = x.copy()
    finite = np.isfinite(out)
    out[finite] = np.arcsinh((out[finite] - center) / scale)
    return out


def apply_bounds(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = x.copy()
    finite = np.isfinite(out)
    out[finite] = np.clip(out[finite], lo, hi)
    return out


def fit_clip_bounds(x_train: np.ndarray, q: float) -> tuple[float, float]:
    v = x_train[np.isfinite(x_train)]
    if len(v) == 0:
        return -np.inf, np.inf
    if q <= 0:
        return float(np.min(v)), float(np.max(v))
    if q >= 0.5:
        raise ValueError("transform-tail-clip-q must be < 0.5")
    lo, hi = np.quantile(v, [q, 1.0 - q])
    return float(lo), float(hi)


def fit_rank_gauss_reference(x_train: np.ndarray) -> np.ndarray:
    v = x_train[np.isfinite(x_train)]
    if len(v) == 0:
        return np.array([0.0], dtype=float)
    return np.sort(v)


def apply_rank_gauss(x: np.ndarray, sorted_train: np.ndarray) -> np.ndarray:
    out = x.copy()
    finite = np.isfinite(out)
    if not np.any(finite):
        return out

    n = len(sorted_train)
    if n == 1:
        out[finite] = 0.0
        return out

    probs = np.linspace(1.0 / (n + 1.0), n / (n + 1.0), n)
    p = np.interp(out[finite], sorted_train, probs, left=probs[0], right=probs[-1])
    out[finite] = norm_ppf(p)
    return out


def build_interaction_features(base_features: dict[str, np.ndarray], y: np.ndarray, mode: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if mode == "none":
        return out

    if mode in {"with_y", "with_y_gxc"}:
        for name, arr in base_features.items():
            out[f"i_{name}_x_Y"] = arr * y

    if mode in {"gxc", "with_y_gxc"}:
        g_names = [k for k in base_features if k.startswith("g_")]
        c_names = [k for k in base_features if k.startswith("c_")]
        for g in g_names:
            for c in c_names:
                out[f"i_{g}_x_{c}"] = base_features[g] * base_features[c]

    return out


def default_transform_meta_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_transform_meta.json")


def main() -> None:
    args = parse_args()
    if args.dt <= 0:
        raise ValueError("--dt must be > 0")
    if args.slow_log_window < 1:
        raise ValueError("--slow-log-window must be >= 1")
    if args.smooth_window < 1:
        raise ValueError("--smooth-window must be >= 1")
    if not (0 < args.train_ratio < 1):
        raise ValueError("--train-ratio must be in (0,1)")
    if not (0 <= args.valid_ratio < 1):
        raise ValueError("--valid-ratio must be in [0,1)")
    if args.train_ratio + args.valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio must be < 1")

    input_path = Path(args.input)
    output_path = Path(args.output)
    transform_meta_path = Path(args.transform_meta) if args.transform_meta.strip() else default_transform_meta_path(output_path)

    dates, data = read_panel(input_path)
    require_columns(
        data,
        [
            "sp500",
            "ust10y",
            "dxy_broad",
            "vix",
            "fed_funds",
            "walcl",
            "bus_loans",
            "cpi",
            "indpro",
            "unrate",
        ],
    )

    sp500 = data["sp500"]
    if np.any(~np.isfinite(sp500) | (sp500 <= 0)):
        raise ValueError("sp500 contains non-positive or missing values; cannot compute log(sp500)")

    log_sp500 = np.log(sp500)
    trend = build_trend(log_sp500, method=args.trend_method, hp_lambda=args.hp_lambda)
    y_raw = log_sp500 - trend
    y = smooth_series(y_raw, method=args.smooth_method, window=args.smooth_window, alpha=args.ema_alpha)

    if args.diff_mode == "central":
        v = central_velocity(y, dt=args.dt)
        a = central_acceleration(y, dt=args.dt)
    else:
        v = causal_velocity(y, dt=args.dt)
        a = causal_acceleration(y, dt=args.dt)
    v = winsorize(v, q=args.winsor_quantile)
    a = winsorize(a, q=args.winsor_quantile)

    base_feature_map: dict[str, np.ndarray] = {
        "g_us10y": first_diff(data["ust10y"]),
        "g_dxy": log_diff(data["dxy_broad"]),
        "g_vix": log_diff(data["vix"]),
        "c_policy_rate": first_diff(data["fed_funds"]),
        "c_liquidity_ops": log_change_window(data["walcl"], window=args.slow_log_window),
        "c_credit": log_change_window(data["bus_loans"], window=args.slow_log_window),
        "c_cpi": log_change_window(data["cpi"], window=args.slow_log_window),
        "c_growth": log_change_window(data["indpro"], window=args.slow_log_window),
        "c_unrate": first_diff(data["unrate"]),
    }

    series_map: dict[str, np.ndarray] = {
        "Y": y.copy(),
        "v": v.copy(),
        "a": a.copy(),
    }
    series_map.update({k: arr.copy() for k, arr in base_feature_map.items()})

    prelim_valid = np.isfinite(y_raw) & np.isfinite(series_map["Y"]) & np.isfinite(series_map["v"]) & np.isfinite(series_map["a"])
    for arr in base_feature_map.values():
        prelim_valid &= np.isfinite(arr)
    prelim_idx = np.flatnonzero(prelim_valid)
    if len(prelim_idx) == 0:
        raise ValueError("no valid rows before transform fitting")

    prelim_train_end = int(len(prelim_idx) * args.train_ratio)
    if prelim_train_end <= 1:
        raise ValueError("too few train rows for transform fitting")
    train_idx_for_fit = prelim_idx[:prelim_train_end]

    transform_columns = parse_column_list(args.transform_columns)
    transform_meta: dict[str, object] = {
        "method": args.feature_transform,
        "columns": transform_columns,
        "tail_clip_q": args.transform_tail_clip_q,
        "interaction_mode": args.interaction_mode,
        "train_rows_for_fit": int(len(train_idx_for_fit)),
        "params": {},
    }

    if args.feature_transform != "none":
        for col in transform_columns:
            if col not in series_map:
                raise ValueError(f"transform column not found: {col}")

            train_vals = series_map[col][train_idx_for_fit]
            lo, hi = fit_clip_bounds(train_vals, q=args.transform_tail_clip_q)
            clipped = apply_bounds(series_map[col], lo=lo, hi=hi)

            if args.feature_transform == "asinh_z":
                center, scale = robust_loc_scale(clipped[train_idx_for_fit])
                series_map[col] = apply_asinh_z(clipped, center=center, scale=scale)
                transform_meta["params"][col] = {
                    "clip_lo": lo,
                    "clip_hi": hi,
                    "center": center,
                    "scale": scale,
                }
            elif args.feature_transform == "rank_gauss":
                ref = fit_rank_gauss_reference(clipped[train_idx_for_fit])
                series_map[col] = apply_rank_gauss(clipped, sorted_train=ref)
                transform_meta["params"][col] = {
                    "clip_lo": lo,
                    "clip_hi": hi,
                    "ref_n": int(len(ref)),
                    "ref_min": float(ref[0]),
                    "ref_max": float(ref[-1]),
                }

    base_keys = list(base_feature_map.keys())
    feature_map: dict[str, np.ndarray] = {k: series_map[k] for k in base_keys}

    interaction_map = build_interaction_features(feature_map, series_map["Y"], mode=args.interaction_mode)
    feature_map.update(interaction_map)

    if args.add_lag1_features:
        feature_map.update({f"{k}_lag1": lag1(series_map[k]) for k in base_keys})

    final_valid = np.isfinite(y_raw) & np.isfinite(series_map["Y"]) & np.isfinite(series_map["v"]) & np.isfinite(series_map["a"])
    for arr in feature_map.values():
        final_valid &= np.isfinite(arr)

    kept_idx = np.flatnonzero(final_valid)
    if len(kept_idx) == 0:
        raise ValueError("no valid samples after alignment; check missing values and transforms")

    n = len(kept_idx)
    train_end = int(n * args.train_ratio)
    valid_end = int(n * (args.train_ratio + args.valid_ratio))

    output_fields = [
        "date",
        "country",
        "index_level",
        "Y_raw",
        "v_raw",
        "a_raw",
        "Y",
        "v",
        "a",
        *feature_map.keys(),
        "is_train",
        "is_valid",
        "is_test",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()

        for rank, i in enumerate(kept_idx):
            row = {
                "date": dates[i],
                "country": args.country,
                "index_level": f"{sp500[i]:.10g}",
                "Y_raw": f"{y_raw[i]:.10g}",
                "v_raw": f"{v[i]:.10g}",
                "a_raw": f"{a[i]:.10g}",
                "Y": f"{series_map['Y'][i]:.10g}",
                "v": f"{series_map['v'][i]:.10g}",
                "a": f"{series_map['a'][i]:.10g}",
                "is_train": "1" if rank < train_end else "0",
                "is_valid": "1" if train_end <= rank < valid_end else "0",
                "is_test": "1" if rank >= valid_end else "0",
            }
            for k, arr in feature_map.items():
                row[k] = f"{arr[i]:.10g}"
            writer.writerow(row)

    transform_meta_path.parent.mkdir(parents=True, exist_ok=True)
    transform_meta_path.write_text(json.dumps(transform_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[done] input rows: {len(dates)}")
    print(f"[done] output rows: {n}")
    print(
        "[done] options: "
        f"trend={args.trend_method}, smooth={args.smooth_method}, diff_mode={args.diff_mode}, slow_log_window={args.slow_log_window}, "
        f"winsor_q={args.winsor_quantile}, lag1={args.add_lag1_features}, "
        f"interaction={args.interaction_mode}, transform={args.feature_transform}, clip_q={args.transform_tail_clip_q}"
    )
    print(f"[done] panel: {output_path}")
    print(f"[done] transform meta: {transform_meta_path}")


if __name__ == "__main__":
    main()
