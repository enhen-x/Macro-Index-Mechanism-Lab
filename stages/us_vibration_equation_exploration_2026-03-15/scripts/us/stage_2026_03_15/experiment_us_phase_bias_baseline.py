"""Step 5 experiment: quantify phase bias of current main model on one-step Y(t+1|t)."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantify phase bias on one-step Y(t+1|t) forecasts.")
    parser.add_argument("--panel", default="data/us/us_regression_panel.csv")
    parser.add_argument("--estimation", default="data/us/us_ols_estimation.json")
    parser.add_argument("--split", choices=["train", "valid", "test", "all"], default="test")
    parser.add_argument("--output-dir", default="data/us/experiments/step5_phase_bias_baseline")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--turn-eps", type=float, default=1e-6)
    parser.add_argument("--turn-tol", type=int, default=1)
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


def _corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


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


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty panel file: {path}")
    return rows


def _split_rows(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    if split == "all":
        return rows
    col = f"is_{split}"
    out = [r for r in rows if (r.get(col) or "").strip() == "1"]
    if not out:
        raise ValueError(f"no rows selected for split={split}")
    return out


def _align_for_lag(y_true: np.ndarray, y_pred: np.ndarray, y_anchor: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if lag < 0:
        pred = y_pred[-lag:]
        true = y_true[: len(pred)]
        anchor = y_anchor[: len(pred)]
    elif lag > 0:
        pred = y_pred[:-lag]
        true = y_true[lag:]
        anchor = y_anchor[lag:]
    else:
        pred = y_pred
        true = y_true
        anchor = y_anchor
    return true, pred, anchor


def _lag_scan(y_true: np.ndarray, y_pred: np.ndarray, y_anchor: np.ndarray, max_lag: int) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for lag in range(-max_lag, max_lag + 1):
        yt, yp, ya = _align_for_lag(y_true, y_pred, y_anchor, lag)
        if len(yt) < 3:
            continue
        out.append(
            {
                "lag": int(lag),
                "n": int(len(yt)),
                "r2": _r2(yt, yp),
                "corr": _corr(yt, yp),
                "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
                "direction_accuracy": float(np.mean(np.sign(yt - ya) == np.sign(yp - ya))),
            }
        )
    return out


def _best_lag_by_corr(scan: list[dict[str, float]]) -> dict[str, float]:
    if not scan:
        return {"lag": 0, "corr": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    best = max(scan, key=lambda r: (float(r["corr"]), float(r["r2"]), -float(r["rmse"])))
    return best


def _best_lag_by_rmse(scan: list[dict[str, float]]) -> dict[str, float]:
    if not scan:
        return {"lag": 0, "corr": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    best = min(scan, key=lambda r: (float(r["rmse"]), -float(r["corr"]), -float(r["r2"])))
    return best


def _stable_sign(x: np.ndarray, eps: float) -> np.ndarray:
    s = np.zeros(len(x), dtype=int)
    for i, v in enumerate(x):
        if v > eps:
            s[i] = 1
        elif v < -eps:
            s[i] = -1
        elif i > 0:
            s[i] = s[i - 1]
        else:
            s[i] = 0
    return s


def _turning_indices(delta: np.ndarray, eps: float) -> np.ndarray:
    s = _stable_sign(delta, eps=eps)
    out: list[int] = []
    for i in range(1, len(s)):
        if s[i - 1] == 0 or s[i] == 0:
            continue
        if s[i] != s[i - 1]:
            out.append(i)
    return np.array(out, dtype=int)


def _match_turns(true_idx: np.ndarray, pred_idx: np.ndarray, tol: int) -> tuple[int, int, list[int]]:
    used_pred: set[int] = set()
    matched = 0
    delays: list[int] = []
    for t in true_idx.tolist():
        candidates = [p for p in pred_idx.tolist() if p not in used_pred and abs(p - t) <= tol]
        if not candidates:
            continue
        p_best = sorted(candidates, key=lambda p: (abs(p - t), p))[0]
        used_pred.add(p_best)
        matched += 1
        delays.append(int(p_best - t))
    return matched, len(used_pred), delays


def _turn_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_anchor: np.ndarray, eps: float, tol: int) -> dict[str, float]:
    d_true = y_true - y_anchor
    d_pred = y_pred - y_anchor
    idx_true = _turning_indices(d_true, eps=eps)
    idx_pred = _turning_indices(d_pred, eps=eps)
    matched_true, matched_pred_unique, delays = _match_turns(idx_true, idx_pred, tol=tol)

    n_true = int(len(idx_true))
    n_pred = int(len(idx_pred))
    recall = float(matched_true / n_true) if n_true > 0 else float("nan")
    precision = float(matched_pred_unique / n_pred) if n_pred > 0 else float("nan")
    if np.isfinite(recall) and np.isfinite(precision) and (recall + precision) > 0:
        f1 = float(2.0 * recall * precision / (recall + precision))
    else:
        f1 = float("nan")

    return {
        "n_turn_true": n_true,
        "n_turn_pred": n_pred,
        "turn_hit_rate_recall": recall,
        "turn_precision": precision,
        "turn_f1": f1,
        "mean_delay": float(np.mean(delays)) if delays else float("nan"),
        "median_delay": float(np.median(delays)) if delays else float("nan"),
        "matched_count": int(matched_true),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _split_rows(_load_rows(Path(args.panel)), split=args.split)
    est = json.loads(Path(args.estimation).read_text(encoding="utf-8"))

    m = float(est["m"])
    c = float(est["c"])
    c_nl = float(est.get("c_nl", 0.0))
    k = float(est["k"])
    intercept = float(est.get("intercept", 0.0))
    feature_cols = list(est.get("betas", {}).keys())
    if not feature_cols:
        feature_cols = [cname for cname in rows[0].keys() if cname.startswith("g_") or cname.startswith("c_") or cname.startswith("i_")]
    beta_vec = np.array([float(est["betas"][cname]) / m for cname in feature_cols], dtype=float)

    dates = np.array([datetime.strptime(r["date"], "%Y-%m-%d") for r in rows])
    y = np.array([_to_float(r["Y"]) for r in rows], dtype=float)
    u = np.array([[_to_float(r[cname]) for cname in feature_cols] for r in rows], dtype=float)

    finite = np.isfinite(y) & np.all(np.isfinite(u), axis=1)
    y = y[finite]
    u = u[finite]
    dates = dates[finite]
    if bool(est.get("orthogonalize_country", False)):
        u = _orthogonalize_country_features(u, feature_cols)

    if len(y) < 4:
        raise ValueError("not enough finite rows after filtering")

    idx = np.arange(1, len(y) - 1, dtype=int)
    dt2 = args.dt * args.dt
    v = (y[idx] - y[idx - 1]) / args.dt
    y_hat = (
        2.0 * y[idx]
        - y[idx - 1]
        + dt2
        * (
            intercept
            - (c / m) * v
            - (c_nl / m) * np.abs(v) * v
            - (k / m) * y[idx]
            + (u[idx] @ beta_vec)
        )
    )
    y_true = y[idx + 1]
    y_anchor = y[idx]
    dates_eval = dates[idx + 1]

    lag_scan = _lag_scan(y_true, y_hat, y_anchor, max_lag=args.max_lag)
    best_corr = _best_lag_by_corr(lag_scan)
    best_rmse = _best_lag_by_rmse(lag_scan)

    lag0_true, lag0_pred, lag0_anchor = _align_for_lag(y_true, y_hat, y_anchor, lag=0)
    bc_true, bc_pred, bc_anchor = _align_for_lag(y_true, y_hat, y_anchor, lag=int(best_corr["lag"]))
    br_true, br_pred, br_anchor = _align_for_lag(y_true, y_hat, y_anchor, lag=int(best_rmse["lag"]))

    metrics_lag0 = {
        "n_obs": int(len(lag0_true)),
        "r2": _r2(lag0_true, lag0_pred),
        "corr": _corr(lag0_true, lag0_pred),
        "rmse": float(np.sqrt(np.mean((lag0_true - lag0_pred) ** 2))),
        "mae": float(np.mean(np.abs(lag0_true - lag0_pred))),
        "direction_accuracy": float(np.mean(np.sign(lag0_true - lag0_anchor) == np.sign(lag0_pred - lag0_anchor))),
    }
    metrics_best_corr_lag = {
        "lag": int(best_corr["lag"]),
        "n_obs": int(len(bc_true)),
        "r2": _r2(bc_true, bc_pred),
        "corr": _corr(bc_true, bc_pred),
        "rmse": float(np.sqrt(np.mean((bc_true - bc_pred) ** 2))),
        "direction_accuracy": float(np.mean(np.sign(bc_true - bc_anchor) == np.sign(bc_pred - bc_anchor))),
    }
    metrics_best_rmse_lag = {
        "lag": int(best_rmse["lag"]),
        "n_obs": int(len(br_true)),
        "r2": _r2(br_true, br_pred),
        "corr": _corr(br_true, br_pred),
        "rmse": float(np.sqrt(np.mean((br_true - br_pred) ** 2))),
        "direction_accuracy": float(np.mean(np.sign(br_true - br_anchor) == np.sign(br_pred - br_anchor))),
    }

    turn_lag0 = _turn_metrics(lag0_true, lag0_pred, lag0_anchor, eps=args.turn_eps, tol=args.turn_tol)
    turn_best_corr = _turn_metrics(bc_true, bc_pred, bc_anchor, eps=args.turn_eps, tol=args.turn_tol)

    summary = {
        "panel": str(args.panel),
        "estimation": str(args.estimation),
        "split": args.split,
        "dt": args.dt,
        "max_lag": args.max_lag,
        "turn_eps": args.turn_eps,
        "turn_tol": args.turn_tol,
        "lag0_metrics": metrics_lag0,
        "best_lag_by_corr": best_corr,
        "best_lag_by_rmse": best_rmse,
        "best_corr_lag_metrics": metrics_best_corr_lag,
        "best_rmse_lag_metrics": metrics_best_rmse_lag,
        "turn_metrics_lag0": turn_lag0,
        "turn_metrics_best_corr_lag": turn_best_corr,
    }
    (out_dir / "phase_bias_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with (out_dir / "lag_scan.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lag", "n", "r2", "corr", "rmse", "direction_accuracy"])
        writer.writeheader()
        for row in lag_scan:
            writer.writerow(row)

    with (out_dir / "phase_eval_series.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["date_t1", "y_t1_actual", "y_t1_pred", "y_t", "delta_actual", "delta_pred"],
        )
        writer.writeheader()
        for i in range(len(y_true)):
            writer.writerow(
                {
                    "date_t1": dates_eval[i].strftime("%Y-%m-%d"),
                    "y_t1_actual": float(y_true[i]),
                    "y_t1_pred": float(y_hat[i]),
                    "y_t": float(y_anchor[i]),
                    "delta_actual": float(y_true[i] - y_anchor[i]),
                    "delta_pred": float(y_hat[i] - y_anchor[i]),
                }
            )

    lags = [int(r["lag"]) for r in lag_scan]
    corrs = [float(r["corr"]) for r in lag_scan]
    rmses = [float(r["rmse"]) for r in lag_scan]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=args.dpi)
    axes[0].plot(lags, corrs, marker="o")
    axes[0].axvline(int(best_corr["lag"]), linestyle="--", linewidth=1.0)
    axes[0].set_title("Lag Scan: Correlation")
    axes[0].set_xlabel("lag")
    axes[0].set_ylabel("corr")
    axes[0].grid(alpha=0.25)

    axes[1].plot(lags, rmses, marker="o")
    axes[1].axvline(int(best_rmse["lag"]), linestyle="--", linewidth=1.0)
    axes[1].set_title("Lag Scan: RMSE")
    axes[1].set_xlabel("lag")
    axes[1].set_ylabel("rmse")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "phase_lag_scan.png")
    plt.close(fig)

    d_true = y_true - y_anchor
    d_pred = y_hat - y_anchor
    t_true = _turning_indices(d_true, eps=args.turn_eps)
    t_pred = _turning_indices(d_pred, eps=args.turn_eps)
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=args.dpi)
    ax.plot(dates_eval, d_true, label="delta actual", linewidth=1.2)
    ax.plot(dates_eval, d_pred, label="delta pred", linewidth=1.2)
    if len(t_true) > 0:
        ax.scatter(dates_eval[t_true], d_true[t_true], s=30, marker="o", label="turn true")
    if len(t_pred) > 0:
        ax.scatter(dates_eval[t_pred], d_pred[t_pred], s=30, marker="x", label="turn pred")
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_title("Turning-Point Markers on One-Step Delta")
    ax.set_xlabel("Date")
    ax.set_ylabel("delta = y(t+1)-y(t)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "turning_points_overlay.png")
    plt.close(fig)

    print(f"[done] split={args.split}, n_eval={len(y_true)}")
    print(
        "[done] lag0: "
        f"r2={metrics_lag0['r2']:.6g}, corr={metrics_lag0['corr']:.6g}, "
        f"rmse={metrics_lag0['rmse']:.6g}, dir={metrics_lag0['direction_accuracy']:.6g}"
    )
    print(
        "[done] best_lag_by_corr: "
        f"lag={int(best_corr['lag'])}, corr={float(best_corr['corr']):.6g}, rmse={float(best_corr['rmse']):.6g}"
    )
    print(
        "[done] turning(lag0): "
        f"recall={float(turn_lag0['turn_hit_rate_recall']):.6g}, "
        f"precision={float(turn_lag0['turn_precision']):.6g}, "
        f"mean_delay={float(turn_lag0['mean_delay']):.6g}"
    )
    print(f"[done] output: {out_dir}")


if __name__ == "__main__":
    main()

