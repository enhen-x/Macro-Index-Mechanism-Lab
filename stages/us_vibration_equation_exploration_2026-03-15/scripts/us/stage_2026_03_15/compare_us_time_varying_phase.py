"""Compare static vs time-varying one-step Y(t+1|t) forecasts on US regression panel."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from identification.ols_identifier import identify_ols_y_next_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare static and time-varying one-step forecasts.")
    parser.add_argument("--panel", default="data/us/us_regression_panel.csv", help="Regression panel CSV path.")
    parser.add_argument(
        "--static-estimation",
        default="data/us/us_ols_estimation.json",
        help="Static estimation JSON path. If missing, static comparison is skipped.",
    )
    parser.add_argument("--split", choices=["train", "valid", "test", "all"], default="test")
    parser.add_argument("--mode", choices=["rolling", "expanding"], default="rolling")
    parser.add_argument("--window", type=int, default=120, help="Rolling window length in center-index samples.")
    parser.add_argument("--min-train-obs", type=int, default=72, help="Minimum history size for each dynamic fit.")
    parser.add_argument(
        "--fit-scope",
        choices=["history", "train_only"],
        default="history",
        help="Use all past centers or only centers in train split.",
    )
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--m", type=float, default=1.0)
    parser.add_argument("--fit-intercept", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ridge-alpha", type=float, default=0.05)
    parser.add_argument("--enforce-physical", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--c-min", type=float, default=1e-4)
    parser.add_argument("--k-min", type=float, default=0.0)
    parser.add_argument("--orthogonalize-country", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tail-weight-mode", choices=["none", "abs_power"], default="none")
    parser.add_argument("--tail-weight-q", type=float, default=0.9)
    parser.add_argument("--tail-weight-scale", type=float, default=3.0)
    parser.add_argument("--tail-weight-power", type=float, default=1.0)
    parser.add_argument("--robust-mode", choices=["none", "huber"], default="none")
    parser.add_argument("--robust-tuning", type=float, default=1.345)
    parser.add_argument("--robust-max-iter", type=int, default=20)
    parser.add_argument("--robust-tol", type=float, default=1e-6)
    parser.add_argument("--max-lag", type=int, default=4)
    parser.add_argument("--output-dir", default="data/us/time_varying_compare")
    parser.add_argument("--dpi", type=int, default=140)
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


def _predict_y_next(
    *,
    y_prev: float,
    y_cur: float,
    u_t: np.ndarray,
    c: float,
    k: float,
    m: float,
    x0: float,
    beta_vec: np.ndarray,
    dt: float,
) -> float:
    lam = (c * dt) / (2.0 * m)
    dt2 = dt * dt
    num = 2.0 * y_cur - (1.0 - lam) * y_prev - (k / m) * dt2 * (y_cur - x0) + dt2 * float(u_t @ beta_vec)
    den = 1.0 + lam
    return float(num / den)


def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_anchor: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    r2 = _r2(y_true, y_pred)
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")
    dir_acc = float(np.mean(np.sign(y_true - y_anchor) == np.sign(y_pred - y_anchor)))
    return {
        "n_obs": int(len(y_true)),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "corr": corr,
        "direction_accuracy": dir_acc,
    }


def _lag_scan(y_true: np.ndarray, y_pred: np.ndarray, max_lag: int) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            pred = y_pred[-lag:]
            act = y_true[: len(pred)]
        elif lag > 0:
            pred = y_pred[:-lag]
            act = y_true[lag:]
        else:
            pred = y_pred
            act = y_true
        if len(act) < 3:
            continue
        rmse = float(np.sqrt(np.mean((act - pred) ** 2)))
        corr = float(np.corrcoef(act, pred)[0, 1]) if len(act) > 1 else float("nan")
        out.append({"lag": int(lag), "corr": corr, "rmse": rmse, "n": int(len(act))})
    return out


def _best_lag(scan: list[dict[str, float]]) -> tuple[int, float, float]:
    if not scan:
        return 0, float("nan"), float("nan")
    best = max(scan, key=lambda r: (float(r["corr"]), -float(r["rmse"])))
    return int(best["lag"]), float(best["corr"]), float(best["rmse"])


def main() -> None:
    args = parse_args()

    panel_path = Path(args.panel)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with panel_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty panel file: {panel_path}")

    static_est = None
    static_path = Path(args.static_estimation)
    if static_path.exists():
        static_est = json.loads(static_path.read_text(encoding="utf-8"))

    if static_est and static_est.get("betas"):
        feature_cols = list(static_est["betas"].keys())
    else:
        feature_cols = [c for c in rows[0].keys() if c.startswith("g_") or c.startswith("c_") or c.startswith("i_")]

    required = {"Y", "a", *feature_cols, "date", "is_train", "is_valid", "is_test"}
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise ValueError(f"missing required columns in panel: {missing}")

    dates = [datetime.strptime(r["date"], "%Y-%m-%d") for r in rows]
    y = np.array([_to_float(r["Y"]) for r in rows], dtype=float)
    a = np.array([_to_float(r["a"]) for r in rows], dtype=float)
    u = np.array([[_to_float(r[c]) for c in feature_cols] for r in rows], dtype=float)
    is_train = np.array([(r.get("is_train") or "").strip() == "1" for r in rows], dtype=bool)
    is_valid = np.array([(r.get("is_valid") or "").strip() == "1" for r in rows], dtype=bool)
    is_test = np.array([(r.get("is_test") or "").strip() == "1" for r in rows], dtype=bool)

    if args.orthogonalize_country:
        u = _orthogonalize_country_features(u, feature_cols)

    n = len(y)
    idx = np.arange(1, n - 1, dtype=int)
    center_finite = (
        np.isfinite(y[idx - 1])
        & np.isfinite(y[idx])
        & np.isfinite(y[idx + 1])
        & np.isfinite(a[idx])
        & np.all(np.isfinite(u[idx]), axis=1)
    )
    center_idx = idx[center_finite]

    if args.fit_scope == "train_only":
        train_pool_mask = is_train[center_idx]
    else:
        train_pool_mask = np.ones_like(center_idx, dtype=bool)
    train_pool = center_idx[train_pool_mask]

    if args.split == "train":
        eval_mask = is_train[center_idx + 1]
    elif args.split == "valid":
        eval_mask = is_valid[center_idx + 1]
    elif args.split == "test":
        eval_mask = is_test[center_idx + 1]
    else:
        eval_mask = np.ones_like(center_idx, dtype=bool)
    eval_centers = center_idx[eval_mask]
    if len(eval_centers) == 0:
        raise ValueError(f"no evaluation centers for split={args.split}")

    static_beta = None
    static_c = static_k = static_x0 = static_m = None
    if static_est and static_est.get("betas"):
        static_c = float(static_est["c"])
        static_k = float(static_est["k"])
        static_x0 = float(static_est.get("x0", 0.0))
        static_m = float(static_est.get("m", args.m))
        static_beta = np.array([float(static_est["betas"][c]) / static_m for c in feature_cols], dtype=float)

    records: list[dict[str, object]] = []
    skipped = 0

    for center in eval_centers:
        hist = train_pool[train_pool < center]
        if args.mode == "rolling" and args.window > 0 and len(hist) > args.window:
            hist = hist[-args.window :]
        if len(hist) < args.min_train_obs:
            skipped += 1
            continue

        try:
            est_dyn = identify_ols_y_next_panel(
                y_prev=y[hist - 1],
                y_cur=y[hist],
                y_next=y[hist + 1],
                u_cur=u[hist],
                feature_names=feature_cols,
                a_for_weight=a[hist],
                m=args.m,
                dt=args.dt,
                fit_intercept=args.fit_intercept,
                ridge_alpha=args.ridge_alpha,
                enforce_physical=args.enforce_physical,
                c_min=args.c_min,
                k_min=args.k_min,
                tail_weight_mode=args.tail_weight_mode,
                tail_weight_q=args.tail_weight_q,
                tail_weight_scale=args.tail_weight_scale,
                tail_weight_power=args.tail_weight_power,
                robust_mode=args.robust_mode,
                robust_tuning=args.robust_tuning,
                robust_max_iter=args.robust_max_iter,
                robust_tol=args.robust_tol,
            )
        except ValueError:
            skipped += 1
            continue

        beta_dyn = np.array([float(est_dyn.betas[c]) / est_dyn.m for c in feature_cols], dtype=float)
        y_pred_dyn = _predict_y_next(
            y_prev=float(y[center - 1]),
            y_cur=float(y[center]),
            u_t=u[center],
            c=est_dyn.c,
            k=est_dyn.k,
            m=est_dyn.m,
            x0=est_dyn.x0,
            beta_vec=beta_dyn,
            dt=args.dt,
        )

        y_pred_static = float("nan")
        if static_beta is not None:
            y_pred_static = _predict_y_next(
                y_prev=float(y[center - 1]),
                y_cur=float(y[center]),
                u_t=u[center],
                c=float(static_c),
                k=float(static_k),
                m=float(static_m),
                x0=float(static_x0),
                beta_vec=static_beta,
                dt=args.dt,
            )

        records.append(
            {
                "center_idx": int(center),
                "date_t": dates[center].strftime("%Y-%m-%d"),
                "date_t1": dates[center + 1].strftime("%Y-%m-%d"),
                "y_t": float(y[center]),
                "y_t1_actual": float(y[center + 1]),
                "y_t1_pred_dynamic": y_pred_dyn,
                "y_t1_pred_static": y_pred_static,
                "c_dynamic": float(est_dyn.c),
                "k_dynamic": float(est_dyn.k),
                "n_train": int(len(hist)),
            }
        )

    if not records:
        raise ValueError("no predictions generated; try reducing --min-train-obs")

    rec_dates = np.array([datetime.strptime(str(r["date_t1"]), "%Y-%m-%d") for r in records])
    y_true = np.array([float(r["y_t1_actual"]) for r in records], dtype=float)
    y_anchor = np.array([float(r["y_t"]) for r in records], dtype=float)
    y_dyn = np.array([float(r["y_t1_pred_dynamic"]) for r in records], dtype=float)
    y_sta = np.array([float(r["y_t1_pred_static"]) for r in records], dtype=float)

    metrics_dynamic = _calc_metrics(y_true, y_dyn, y_anchor)
    lag_dynamic = _lag_scan(y_true, y_dyn, args.max_lag)
    b_lag_dyn, b_corr_dyn, b_rmse_dyn = _best_lag(lag_dynamic)
    metrics_dynamic["best_lag"] = int(b_lag_dyn)
    metrics_dynamic["best_lag_corr"] = b_corr_dyn
    metrics_dynamic["best_lag_rmse"] = b_rmse_dyn

    metrics_static = None
    lag_static = None
    valid_sta = np.isfinite(y_sta)
    if np.any(valid_sta):
        metrics_static = _calc_metrics(y_true[valid_sta], y_sta[valid_sta], y_anchor[valid_sta])
        lag_static = _lag_scan(y_true[valid_sta], y_sta[valid_sta], args.max_lag)
        b_lag_sta, b_corr_sta, b_rmse_sta = _best_lag(lag_static)
        metrics_static["best_lag"] = int(b_lag_sta)
        metrics_static["best_lag_corr"] = b_corr_sta
        metrics_static["best_lag_rmse"] = b_rmse_sta

    pred_csv = out_dir / "predictions.csv"
    with pred_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "center_idx",
                "date_t",
                "date_t1",
                "y_t",
                "y_t1_actual",
                "y_t1_pred_dynamic",
                "y_t1_pred_static",
                "c_dynamic",
                "k_dynamic",
                "n_train",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    payload = {
        "panel": str(panel_path),
        "static_estimation": str(static_path),
        "split": args.split,
        "mode": args.mode,
        "window": int(args.window),
        "min_train_obs": int(args.min_train_obs),
        "fit_scope": args.fit_scope,
        "n_eval_candidates": int(len(eval_centers)),
        "n_predictions": int(len(records)),
        "n_skipped": int(skipped),
        "dynamic_metrics": metrics_dynamic,
        "dynamic_lag_scan": lag_dynamic,
        "static_metrics": metrics_static,
        "static_lag_scan": lag_static,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=args.dpi)
    ax.plot(rec_dates, y_true, label="actual y(t+1)", linewidth=1.25)
    ax.plot(rec_dates, y_dyn, label="dynamic y_hat(t+1|t)", linewidth=1.2)
    if np.any(np.isfinite(y_sta)):
        ax.plot(rec_dates, y_sta, label="static y_hat(t+1|t)", linewidth=1.0, alpha=0.85)
    ax.set_title(f"One-Step Y Forecast ({args.split}, {args.mode})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "y_next_timeseries_compare.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=args.dpi)
    axes[0].plot(rec_dates, [float(r["c_dynamic"]) for r in records], linewidth=1.0)
    axes[0].set_title("Dynamic c(t)")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("c")
    axes[0].grid(alpha=0.25)

    axes[1].plot(rec_dates, [float(r["k_dynamic"]) for r in records], linewidth=1.0)
    axes[1].set_title("Dynamic k(t)")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("k")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "dynamic_ck_timeseries.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 5.8), dpi=args.dpi)
    ax.scatter(y_true, y_dyn, s=14, alpha=0.7, label="dynamic")
    if np.any(np.isfinite(y_sta)):
        ax.scatter(y_true[valid_sta], y_sta[valid_sta], s=14, alpha=0.6, label="static")
    lo = float(min(np.nanmin(y_true), np.nanmin(y_dyn), np.nanmin(y_sta) if np.any(np.isfinite(y_sta)) else np.nanmin(y_dyn)))
    hi = float(max(np.nanmax(y_true), np.nanmax(y_dyn), np.nanmax(y_sta) if np.any(np.isfinite(y_sta)) else np.nanmax(y_dyn)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
    ax.set_title("Actual vs Predicted Y(t+1)")
    ax.set_xlabel("actual y(t+1)")
    ax.set_ylabel("predicted y(t+1|t)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "y_next_scatter_compare.png")
    plt.close(fig)

    print(f"[done] split={args.split}, mode={args.mode}, preds={len(records)}, skipped={skipped}")
    print(
        f"[done] dynamic: r2={metrics_dynamic['r2']:.6g}, rmse={metrics_dynamic['rmse']:.6g}, "
        f"dir_acc={metrics_dynamic['direction_accuracy']:.6g}, best_lag={metrics_dynamic['best_lag']}"
    )
    if metrics_static is not None:
        print(
            f"[done] static: r2={metrics_static['r2']:.6g}, rmse={metrics_static['rmse']:.6g}, "
            f"dir_acc={metrics_static['direction_accuracy']:.6g}, best_lag={metrics_static['best_lag']}"
        )
    print(f"[done] outputs: {out_dir}")


if __name__ == "__main__":
    main()
