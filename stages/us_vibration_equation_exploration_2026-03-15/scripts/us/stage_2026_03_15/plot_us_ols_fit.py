"""Plot OLS fit diagnostics for US regression panel.

Outputs figures to data/us/plot by default:
- actual_vs_fitted_timeseries.png
- actual_vs_fitted_scatter.png
- residual_diagnostics.png
- x_actual_vs_fitted_timeseries.png
- x_actual_vs_fitted_scatter.png
- fit_metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot fit diagnostics from regression panel and estimation JSON.")
    parser.add_argument("--panel", default="data/us/us_regression_panel.csv", help="Regression panel CSV path.")
    parser.add_argument("--estimation", default="data/us/us_ols_estimation.json", help="Estimation JSON path.")
    parser.add_argument("--split", choices=["train", "valid", "test", "all"], default="train")
    parser.add_argument("--output-dir", default="data/us/plot", help="Output directory for plots.")
    parser.add_argument("--dpi", type=int, default=140, help="Figure DPI.")
    parser.add_argument("--dt", type=float, default=1.0, help="Time step used in x reconstruction from a_hat.")
    parser.add_argument(
        "--damping-mode",
        choices=["auto", "linear", "nonlinear_absv"],
        default="auto",
        help="Damping form used in one-step forecast. auto=read from estimation JSON.",
    )
    parser.add_argument(
        "--x-time-align",
        choices=["asof_t", "outcome_t1"],
        default="outcome_t1",
        help="Time axis for one-step x plot: decision time t or realized outcome time t+1.",
    )
    parser.add_argument(
        "--x-error-correction",
        choices=["none", "ar1"],
        default="none",
        help="Optional one-step prediction error correction for x/Y.",
    )
    parser.add_argument(
        "--x-phi",
        type=float,
        default=float("nan"),
        help="Fixed AR(1) correction coefficient. If NaN and x-error-correction=ar1, estimate from source split.",
    )
    parser.add_argument(
        "--x-phi-source-split",
        choices=["train", "valid", "test", "all"],
        default="train",
        help="Split used to estimate AR(1) correction coefficient when x-phi is not provided.",
    )
    return parser.parse_args()


def _to_float(value: str) -> float:
    text = (value or "").strip()
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    mean_y = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - mean_y) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def _predict_y_next_closed_form(
    y: np.ndarray,
    u: np.ndarray,
    beta_vec: np.ndarray,
    *,
    c: float,
    c_nl: float,
    k: float,
    m: float,
    x0: float,
    intercept: float,
    dt: float = 1.0,
    damping_mode: str = "linear",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-step-ahead prediction for Y_{t+1|t} from the fitted discrete equation."""
    n = len(y)
    if n < 3:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)

    idx = np.arange(1, n - 1, dtype=int)
    dt2 = dt * dt
    forcing = u[idx] @ beta_vec

    if damping_mode == "nonlinear_absv":
        v_causal = (y[idx] - y[idx - 1]) / dt
        y_next_hat = (
            2.0 * y[idx]
            - y[idx - 1]
            + dt2
            * (
                intercept
                - (c / m) * v_causal
                - (c_nl / m) * np.abs(v_causal) * v_causal
                - (k / m) * y[idx]
                + forcing
            )
        )
    else:
        lam = (c * dt) / (2.0 * m)
        num = (
            2.0 * y[idx]
            - (1.0 - lam) * y[idx - 1]
            - (k / m) * dt2 * (y[idx] - x0)
            + dt2 * forcing
        )
        den = 1.0 + lam
        y_next_hat = num / den

    y_next_actual = y[idx + 1]
    return idx, y_next_actual, y_next_hat


def _estimate_ar1_phi(errors: np.ndarray) -> float:
    if len(errors) < 3:
        return 0.0
    e_prev = errors[:-1]
    e_cur = errors[1:]
    den = float(np.sum(e_prev * e_prev))
    if den <= 1e-12:
        return 0.0
    phi = float(np.sum(e_cur * e_prev) / den)
    return float(np.clip(phi, -0.99, 0.99))


def _apply_ar1_correction(y_true: np.ndarray, y_hat_raw: np.ndarray, phi: float) -> np.ndarray:
    if len(y_true) != len(y_hat_raw):
        raise ValueError("y_true and y_hat_raw must have same length")
    if len(y_true) == 0:
        return y_hat_raw.copy()

    out = y_hat_raw.copy()
    e_prev = 0.0
    for i in range(len(out)):
        out[i] = out[i] + phi * e_prev
        e_prev = float(y_true[i] - out[i])
    return out


def _orthogonalize_country_features(u: np.ndarray, feature_names: list[str]) -> np.ndarray:
    global_idx = [i for i, c in enumerate(feature_names) if c.startswith("g_")]
    country_idx = [i for i, c in enumerate(feature_names) if c.startswith("c_")]
    if not global_idx or not country_idx:
        return u

    out = u.copy()
    g = out[:, global_idx]
    z = np.column_stack([np.ones(len(out)), g])
    for j in country_idx:
        target = out[:, j]
        coef, *_ = np.linalg.lstsq(z, target, rcond=None)
        out[:, j] = target - z @ coef
    return out


def load_panel_rows(panel_path: Path, split: str) -> list[dict[str, str]]:
    with panel_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty panel file: {panel_path}")

    if split == "all":
        return rows

    col = f"is_{split}"
    if col not in rows[0]:
        raise ValueError(f"split column not found: {col}")
    selected = [r for r in rows if (r.get(col) or "").strip() == "1"]
    if not selected:
        raise ValueError(f"no rows selected for split={split}")
    return selected


def main() -> None:
    args = parse_args()

    panel_path = Path(args.panel)
    est_path = Path(args.estimation)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    est = json.loads(est_path.read_text(encoding="utf-8"))
    m = float(est["m"])
    c = float(est["c"])
    c_nl = float(est.get("c_nl", 0.0))
    k = float(est["k"])
    x0 = float(est.get("x0", 0.0))
    intercept = float(est.get("intercept", 0.0))
    betas: dict[str, float] = est.get("betas", {})
    damping_mode = str(est.get("damping_mode", "linear")) if args.damping_mode == "auto" else args.damping_mode
    feature_cols = list(betas.keys())
    if not feature_cols:
        raise ValueError("estimation JSON has empty betas")

    rows = load_panel_rows(panel_path, split=args.split)

    dates = [datetime.strptime(r["date"], "%Y-%m-%d") for r in rows]
    a = np.array([_to_float(r["a"]) for r in rows], dtype=float)
    v = np.array([_to_float(r["v"]) for r in rows], dtype=float)
    y = np.array([_to_float(r["Y"]) for r in rows], dtype=float)
    u = np.array([[_to_float(r[cname]) for cname in feature_cols] for r in rows], dtype=float)

    finite = np.isfinite(a) & np.isfinite(v) & np.isfinite(y) & np.all(np.isfinite(u), axis=1)
    if not np.any(finite):
        raise ValueError("no finite rows found for plotting")

    if bool(est.get("orthogonalize_country", False)):
        u = _orthogonalize_country_features(u, feature_cols)

    dates_f = [d for d, ok in zip(dates, finite) if ok]
    a = a[finite]
    v = v[finite]
    y = y[finite]
    u = u[finite]

    b_v = -c / m
    b_y = -k / m
    beta_vec = np.array([float(betas[cname]) / m for cname in feature_cols], dtype=float)

    if damping_mode == "nonlinear_absv":
        v_used = v.copy()
        if len(v_used) > 1:
            v_used[1:] = (y[1:] - y[:-1]) / args.dt
            v_used[0] = v_used[1]
        a_hat = intercept + b_v * v_used + (-c_nl / m) * np.abs(v_used) * v_used + b_y * y + u @ beta_vec
    else:
        a_hat = intercept + b_v * v + b_y * y + u @ beta_vec
    resid = a - a_hat

    def build_one_step_from_rows(rows_input: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        y_in = np.array([_to_float(r["Y"]) for r in rows_input], dtype=float)
        u_in = np.array([[_to_float(r[cname]) for cname in feature_cols] for r in rows_input], dtype=float)
        finite_in = np.isfinite(y_in) & np.all(np.isfinite(u_in), axis=1)
        y_in = y_in[finite_in]
        u_in = u_in[finite_in]
        if bool(est.get("orthogonalize_country", False)):
            u_in = _orthogonalize_country_features(u_in, feature_cols)
        _, y_next_act, y_next_hat = _predict_y_next_closed_form(
            y_in,
            u_in,
            beta_vec,
            c=c,
            c_nl=c_nl,
            k=k,
            m=m,
            x0=x0,
            intercept=intercept,
            dt=args.dt,
            damping_mode=damping_mode,
        )
        return y_next_act, y_next_hat

    idx_step, x_next_actual, x_next_hat = _predict_y_next_closed_form(
        y,
        u,
        beta_vec,
        c=c,
        c_nl=c_nl,
        k=k,
        m=m,
        x0=x0,
        intercept=intercept,
        dt=args.dt,
        damping_mode=damping_mode,
    )

    x_error_phi = 0.0
    if args.x_error_correction == "ar1":
        if np.isfinite(args.x_phi):
            x_error_phi = float(args.x_phi)
        else:
            src_rows = load_panel_rows(panel_path, split=args.x_phi_source_split)
            src_actual, src_hat = build_one_step_from_rows(src_rows)
            src_err = src_actual - src_hat
            x_error_phi = _estimate_ar1_phi(src_err)
        x_next_hat = _apply_ar1_correction(x_next_actual, x_next_hat, phi=x_error_phi)

    mse = float(np.mean((a - a_hat) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(a - a_hat)))
    r2 = _r2(a, a_hat)
    corr = float(np.corrcoef(a, a_hat)[0, 1]) if len(a) > 1 else float("nan")
    direction_accuracy = _direction_accuracy(a, a_hat)
    finite_x = np.isfinite(x_next_actual) & np.isfinite(x_next_hat)
    y_eval = x_next_actual[finite_x]
    x_hat_eval = x_next_hat[finite_x]
    x_mse = float(np.mean((y_eval - x_hat_eval) ** 2))
    x_rmse = float(np.sqrt(x_mse))
    x_mae = float(np.mean(np.abs(y_eval - x_hat_eval)))
    x_r2 = _r2(y_eval, x_hat_eval)
    x_corr = float(np.corrcoef(y_eval, x_hat_eval)[0, 1]) if len(y_eval) > 1 else float("nan")

    metrics = {
        "split": args.split,
        "n_obs": int(len(a)),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "corr": corr,
        "direction_accuracy": direction_accuracy,
        "x_mse": x_mse,
        "x_rmse": x_rmse,
        "x_mae": x_mae,
        "x_r2": x_r2,
        "x_corr": x_corr,
        "x_eval_mode": "one_step_t_to_t1",
        "x_eval_n_obs": int(len(y_eval)),
        "x_time_align": args.x_time_align,
        "x_error_correction": args.x_error_correction,
        "x_error_phi": x_error_phi,
        "x_error_phi_source_split": args.x_phi_source_split,
        "c": c,
        "c_nl": c_nl,
        "k": k,
        "damping_mode": damping_mode,
    }
    (out_dir / "fit_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # 1) Time series: actual vs fitted acceleration
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=args.dpi)
    ax.plot(dates_f, a, label="actual a", linewidth=1.2)
    ax.plot(dates_f, a_hat, label="fitted a", linewidth=1.2)
    ax.set_title(f"Acceleration Fit ({args.split})")
    ax.set_xlabel("Date")
    ax.set_ylabel("a")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "actual_vs_fitted_timeseries.png")
    plt.close(fig)

    # 2) Scatter: fitted vs actual
    fig, ax = plt.subplots(figsize=(5.8, 5.8), dpi=args.dpi)
    ax.scatter(a, a_hat, s=14, alpha=0.7)
    lo = float(min(np.min(a), np.min(a_hat)))
    hi = float(max(np.max(a), np.max(a_hat)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
    ax.set_title(f"Actual vs Fitted ({args.split})")
    ax.set_xlabel("actual a")
    ax.set_ylabel("fitted a")
    ax.grid(alpha=0.25)
    txt = f"R2={r2:.4f}\\nRMSE={rmse:.4g}"
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(out_dir / "actual_vs_fitted_scatter.png")
    plt.close(fig)

    # 3) Residual diagnostics: over time + histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=args.dpi)
    axes[0].plot(dates_f, resid, linewidth=1.0)
    axes[0].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[0].set_title("Residual Over Time")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("residual")
    axes[0].grid(alpha=0.25)

    axes[1].hist(resid, bins=25, alpha=0.85)
    axes[1].set_title("Residual Histogram")
    axes[1].set_xlabel("residual")
    axes[1].set_ylabel("count")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_dir / "residual_diagnostics.png")
    plt.close(fig)

    # 4) x (Y) one-step diagnostics from predicted acceleration
    if args.x_time_align == "asof_t":
        dates_x_all = [dates_f[i] for i in idx_step]
        x_title = f"X (Y) One-Step Ahead As-Of t ({args.split})"
        x_actual_label = "actual x(t+1)"
        x_pred_label = "predicted x(t+1|t)"
        x_xlabel = "actual x(t+1)"
        x_ylabel = "predicted x(t+1|t)"
    else:
        dates_x_all = [dates_f[i + 1] for i in idx_step]
        x_title = f"X (Y) One-Step Ahead Outcome t+1 ({args.split})"
        x_actual_label = "actual x(t+1)"
        x_pred_label = "predicted x(t+1|t)"
        x_xlabel = "actual x(t+1)"
        x_ylabel = "predicted x(t+1|t)"
    dates_x = [d for d, ok in zip(dates_x_all, finite_x) if ok]
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=args.dpi)
    ax.plot(dates_x, y_eval, label=x_actual_label, linewidth=1.2)
    ax.plot(dates_x, x_hat_eval, label=x_pred_label, linewidth=1.2)
    ax.set_title(x_title)
    ax.set_xlabel("Date")
    ax.set_ylabel("x (Y)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "x_actual_vs_fitted_timeseries.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 5.8), dpi=args.dpi)
    ax.scatter(y_eval, x_hat_eval, s=14, alpha=0.7)
    lo = float(min(np.min(y_eval), np.min(x_hat_eval)))
    hi = float(max(np.max(y_eval), np.max(x_hat_eval)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
    ax.set_title(f"X (Y): Actual vs One-Step Pred ({args.split})")
    ax.set_xlabel(x_xlabel)
    ax.set_ylabel(x_ylabel)
    ax.grid(alpha=0.25)
    txt = f"R2={x_r2:.4f}\\nRMSE={x_rmse:.4g}"
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(out_dir / "x_actual_vs_fitted_scatter.png")
    plt.close(fig)

    print(f"[done] split={args.split}, n={len(a)}")
    print(
        f"[done] r2={r2:.6g}, rmse={rmse:.6g}, mae={mae:.6g}, "
        f"direction_acc={direction_accuracy:.6g}"
    )
    print(f"[done] x_r2={x_r2:.6g}, x_rmse={x_rmse:.6g}, x_mae={x_mae:.6g}")
    if args.x_error_correction == "ar1":
        print(f"[done] x_error_correction=ar1, phi={x_error_phi:.6g}, source_split={args.x_phi_source_split}")
    print(f"[done] saved metrics: {out_dir / 'fit_metrics.json'}")
    print(f"[done] saved plots in: {out_dir}")


if __name__ == "__main__":
    main()
