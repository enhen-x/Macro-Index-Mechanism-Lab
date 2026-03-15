"""Analyze distribution effects of log/diff transforms and fit residual tails.

Outputs all artifacts to a dedicated folder under data/us.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze transformed feature distributions and extreme-fit behavior."
    )
    parser.add_argument(
        "--monthly-panel",
        default="data/us/us_monthly_panel.csv",
        help="Input monthly raw panel CSV.",
    )
    parser.add_argument(
        "--regression-panel",
        default="data/us/us_regression_panel.csv",
        help="Input regression panel CSV.",
    )
    parser.add_argument(
        "--estimation",
        default="data/us/us_ols_estimation.json",
        help="Estimated parameter JSON path.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test", "all"],
        default="train",
        help="Split used for residual diagnostics.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/us/distribution_analysis",
        help="Output folder for analysis artifacts.",
    )
    parser.add_argument("--bins", type=int, default=36, help="Histogram bins.")
    parser.add_argument("--dpi", type=int, default=140, help="Figure DPI.")
    return parser.parse_args()


def _to_float(value: str) -> float:
    txt = (value or "").strip()
    if not txt:
        return np.nan
    try:
        return float(txt)
    except ValueError:
        return np.nan


def _first_diff(x: np.ndarray) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    if len(x) < 2:
        return out
    lhs = x[1:]
    rhs = x[:-1]
    ok = np.isfinite(lhs) & np.isfinite(rhs)
    out[1:][ok] = lhs[ok] - rhs[ok]
    return out


def _log_diff(x: np.ndarray) -> np.ndarray:
    lx = np.full_like(x, np.nan, dtype=float)
    ok = np.isfinite(x) & (x > 0)
    lx[ok] = np.log(x[ok])
    return _first_diff(lx)


def _series_stats(x: np.ndarray) -> dict[str, float]:
    v = x[np.isfinite(x)]
    if len(v) == 0:
        return {"n": 0}
    n = len(v)
    mu = float(np.mean(v))
    sd = float(np.std(v))
    centered = v - mu
    if sd > 0:
        skew = float(np.mean((centered / sd) ** 3))
        ex_kurt = float(np.mean((centered / sd) ** 4) - 3.0)
        jb = float((n / 6.0) * (skew**2 + 0.25 * (ex_kurt**2)))
        # For chi-square(df=2), survival function is exp(-x/2)
        jb_p_approx = float(np.exp(-jb / 2.0))
    else:
        skew = float("nan")
        ex_kurt = float("nan")
        jb = float("nan")
        jb_p_approx = float("nan")

    q = np.quantile(v, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "n": int(n),
        "mean": mu,
        "std": sd,
        "skew": skew,
        "excess_kurtosis": ex_kurt,
        "jb": jb,
        "jb_p_approx": jb_p_approx,
        "q01": float(q[0]),
        "q05": float(q[1]),
        "q50": float(q[2]),
        "q95": float(q[3]),
        "q99": float(q[4]),
    }


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty csv file: {path}")
    return rows


def _select_rows_by_split(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    if split == "all":
        return rows
    col = f"is_{split}"
    if col not in rows[0]:
        raise ValueError(f"split column not found: {col}")
    out = [r for r in rows if (r.get(col) or "").strip() == "1"]
    if not out:
        raise ValueError(f"no rows selected for split={split}")
    return out


def _orthogonalize_country(u: np.ndarray, names: list[str]) -> np.ndarray:
    gidx = [i for i, c in enumerate(names) if c.startswith("g_")]
    cidx = [i for i, c in enumerate(names) if c.startswith("c_")]
    if not gidx or not cidx:
        return u

    out = u.copy()
    g = out[:, gidx]
    z = np.column_stack([np.ones(len(out)), g])
    for j in cidx:
        t = out[:, j]
        coef, *_ = np.linalg.lstsq(z, t, rcond=None)
        out[:, j] = t - z @ coef
    return out


def _predict_accel(rows: list[dict[str, str]], est: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feat_names = list(est.get("betas", {}).keys())
    if not feat_names:
        raise ValueError("estimation json has empty betas")

    a = np.array([_to_float(r["a"]) for r in rows], dtype=float)
    v = np.array([_to_float(r["v"]) for r in rows], dtype=float)
    y = np.array([_to_float(r["Y"]) for r in rows], dtype=float)
    u = np.array([[_to_float(r[c]) for c in feat_names] for r in rows], dtype=float)

    finite = np.isfinite(a) & np.isfinite(v) & np.isfinite(y) & np.all(np.isfinite(u), axis=1)
    a = a[finite]
    v = v[finite]
    y = y[finite]
    u = u[finite]

    if bool(est.get("orthogonalize_country", False)):
        u = _orthogonalize_country(u, feat_names)

    m = float(est["m"])
    c = float(est["c"])
    k = float(est["k"])
    intercept = float(est.get("intercept", 0.0))
    b_v = -c / m
    b_y = -k / m
    beta = np.array([float(est["betas"][n]) / m for n in feat_names], dtype=float)

    a_hat = intercept + b_v * v + b_y * y + u @ beta
    resid = a - a_hat
    return a, a_hat, resid


def _tail_error_report(a: np.ndarray, a_hat: np.ndarray) -> dict[str, dict[str, float]]:
    abs_a = np.abs(a)
    out: dict[str, dict[str, float]] = {}
    for q in [0.8, 0.9, 0.95, 0.98]:
        th = float(np.quantile(abs_a, q))
        mask = abs_a >= th
        err = a[mask] - a_hat[mask]
        ratio = float(np.mean(np.abs(a_hat[mask]) / np.abs(a[mask])))
        out[f"q{int(q*100)}"] = {
            "threshold_abs_a": th,
            "n": int(np.sum(mask)),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mae": float(np.mean(np.abs(err))),
            "sign_accuracy": float(np.mean(np.sign(a[mask]) == np.sign(a_hat[mask]))),
            "mean_abs_fitted_over_actual": ratio,
        }
    return out


def _plot_feature_hists(output_dir: Path, bins: int, dpi: int, transformed: dict[str, np.ndarray]) -> None:
    show_cols = [
        "dxy_log_diff",
        "vix_log_diff",
        "walcl_log_diff",
        "bus_loans_log_diff",
        "cpi_log_diff",
        "indpro_log_diff",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), dpi=dpi)
    for ax, name in zip(axes.ravel(), show_cols):
        x = transformed[name]
        v = x[np.isfinite(x)]
        if len(v) == 0:
            ax.set_title(f"{name} (empty)")
            continue
        ax.hist(v, bins=bins, alpha=0.85)
        ax.set_title(name)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "transformed_feature_histograms.png")
    plt.close(fig)


def _plot_tail_ratio(output_dir: Path, dpi: int, stats: dict[str, dict[str, float]]) -> None:
    names = []
    vals = []
    for k, v in stats.items():
        sd = v.get("std", float("nan"))
        q99 = v.get("q99", float("nan"))
        if np.isfinite(sd) and sd > 0 and np.isfinite(q99):
            names.append(k)
            vals.append(abs(q99) / sd)

    order = np.argsort(vals)[::-1]
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=dpi)
    ax.bar(range(len(names)), vals)
    ax.axhline(2.326, linestyle="--", linewidth=1.0)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right")
    ax.set_ylabel("|q99| / std")
    ax.set_title("Tail Heaviness Proxy (Normal q99/std=2.326)")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "tail_heaviness_proxy.png")
    plt.close(fig)


def _plot_residual_diagnostics(output_dir: Path, bins: int, dpi: int, a: np.ndarray, a_hat: np.ndarray, resid: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=dpi)
    axes[0].scatter(a, a_hat, s=16, alpha=0.7)
    lo = float(min(np.min(a), np.min(a_hat)))
    hi = float(max(np.max(a), np.max(a_hat)))
    axes[0].plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
    axes[0].set_title("Actual vs Fitted (a)")
    axes[0].set_xlabel("actual a")
    axes[0].set_ylabel("fitted a")
    axes[0].grid(alpha=0.25)

    axes[1].hist(resid, bins=bins, alpha=0.85)
    axes[1].set_title("Residual Histogram")
    axes[1].set_xlabel("residual")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_dir / "residual_fit_diagnostics.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    monthly_path = Path(args.monthly_panel)
    reg_path = Path(args.regression_panel)
    est_path = Path(args.estimation)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    monthly_rows = _load_csv_rows(monthly_path)
    mdata = {k: np.array([_to_float(r.get(k, "")) for r in monthly_rows], dtype=float) for k in monthly_rows[0].keys() if k != "date"}

    raw_series = {
        "sp500_level": mdata["sp500"],
        "dxy_level": mdata["dxy_broad"],
        "vix_level": mdata["vix"],
        "walcl_level": mdata["walcl"],
        "bus_loans_level": mdata["bus_loans"],
        "cpi_level": mdata["cpi"],
        "indpro_level": mdata["indpro"],
        "ust10y_level": mdata["ust10y"],
        "fed_funds_level": mdata["fed_funds"],
        "unrate_level": mdata["unrate"],
    }

    transformed_series = {
        "sp500_log_diff": _log_diff(mdata["sp500"]),
        "dxy_log_diff": _log_diff(mdata["dxy_broad"]),
        "vix_log_diff": _log_diff(mdata["vix"]),
        "walcl_log_diff": _log_diff(mdata["walcl"]),
        "bus_loans_log_diff": _log_diff(mdata["bus_loans"]),
        "cpi_log_diff": _log_diff(mdata["cpi"]),
        "indpro_log_diff": _log_diff(mdata["indpro"]),
        "ust10y_diff": _first_diff(mdata["ust10y"]),
        "fed_funds_diff": _first_diff(mdata["fed_funds"]),
        "unrate_diff": _first_diff(mdata["unrate"]),
    }

    reg_rows = _load_csv_rows(reg_path)
    sel_rows = _select_rows_by_split(reg_rows, split=args.split)
    est = json.loads(est_path.read_text(encoding="utf-8"))
    a, a_hat, resid = _predict_accel(sel_rows, est)

    raw_stats = {k: _series_stats(v) for k, v in raw_series.items()}
    transformed_stats = {k: _series_stats(v) for k, v in transformed_series.items()}
    residual_stats = {
        "actual_a": _series_stats(a),
        "fitted_a": _series_stats(a_hat),
        "residual": _series_stats(resid),
    }

    rmse = float(np.sqrt(np.mean((a - a_hat) ** 2)))
    mae = float(np.mean(np.abs(a - a_hat)))

    summary = {
        "split": args.split,
        "monthly_panel": str(monthly_path),
        "regression_panel": str(reg_path),
        "estimation": str(est_path),
        "normality_notes": {
            "jb_p_approx_interpretation": "approximate p-value using exp(-JB/2) for df=2",
            "heuristic": "jb_p_approx < 0.05 suggests non-normal",
        },
        "raw_stats": raw_stats,
        "transformed_stats": transformed_stats,
        "fit_summary": {
            "n_obs": int(len(a)),
            "rmse": rmse,
            "mae": mae,
            "r2": float(1.0 - np.sum((a - a_hat) ** 2) / np.sum((a - np.mean(a)) ** 2)),
        },
        "tail_error": _tail_error_report(a, a_hat),
        "residual_stats": residual_stats,
    }

    (out_dir / "distribution_fit_analysis.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _plot_feature_hists(out_dir, bins=args.bins, dpi=args.dpi, transformed=transformed_series)
    combined_tail_stats = {**transformed_stats, "residual": residual_stats["residual"]}
    _plot_tail_ratio(out_dir, dpi=args.dpi, stats=combined_tail_stats)
    _plot_residual_diagnostics(out_dir, bins=args.bins, dpi=args.dpi, a=a, a_hat=a_hat, resid=resid)

    print(f"[done] output dir: {out_dir}")
    print(f"[done] report: {out_dir / 'distribution_fit_analysis.json'}")
    print(f"[done] fit rmse={rmse:.6g}, mae={mae:.6g}, n={len(a)}")


if __name__ == "__main__":
    main()
