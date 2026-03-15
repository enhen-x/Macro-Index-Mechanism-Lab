"""Step 8 experiment: ARX residual correction layer for one-step Y(t+1|t)."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply ARX residual correction on top of main one-step forecasts.")
    parser.add_argument("--panel", default="data/us/us_regression_panel.csv")
    parser.add_argument("--estimation", default="data/us/us_ols_estimation.json")
    parser.add_argument("--baseline-metrics", default="data/us/plot/fit_metrics.json")
    parser.add_argument("--baseline-phase", default="data/us/experiments/step5_phase_bias_baseline/phase_bias_summary.json")
    parser.add_argument("--output-dir", default="data/us/experiments/step8_phase_arx_correction")
    parser.add_argument("--train-split", choices=["train", "valid", "test", "all"], default="train")
    parser.add_argument("--test-split", choices=["train", "valid", "test", "all"], default="test")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--turn-eps", type=float, default=1e-6)
    parser.add_argument("--turn-tol", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=1e-8, help="Tiny ridge for ARX OLS stability.")
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


def _direction_acc(y_true: np.ndarray, y_pred: np.ndarray, y_anchor: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.sign(y_true - y_anchor) == np.sign(y_pred - y_anchor)))


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


def _one_step_series(
    rows: list[dict[str, str]],
    feature_cols: list[str],
    est: dict,
    dt: float,
) -> dict[str, np.ndarray]:
    dates = np.array([datetime.strptime(r["date"], "%Y-%m-%d") for r in rows])
    y = np.array([_to_float(r["Y"]) for r in rows], dtype=float)
    u = np.array([[_to_float(r[cname]) for cname in feature_cols] for r in rows], dtype=float)

    finite = np.isfinite(y) & np.all(np.isfinite(u), axis=1)
    y = y[finite]
    u = u[finite]
    dates = dates[finite]

    if bool(est.get("orthogonalize_country", False)):
        u = _orthogonalize_country_features(u, feature_cols)

    if len(y) < 3:
        raise ValueError("not enough finite rows for one-step evaluation")

    m = float(est["m"])
    c = float(est["c"])
    c_nl = float(est.get("c_nl", 0.0))
    k = float(est["k"])
    intercept = float(est.get("intercept", 0.0))
    beta = np.array([float(est["betas"][cname]) / m for cname in feature_cols], dtype=float)

    idx = np.arange(1, len(y) - 1, dtype=int)
    dt2 = dt * dt
    v = (y[idx] - y[idx - 1]) / dt
    y_hat = (
        2.0 * y[idx]
        - y[idx - 1]
        + dt2
        * (
            intercept
            - (c / m) * v
            - (c_nl / m) * np.abs(v) * v
            - (k / m) * y[idx]
            + (u[idx] @ beta)
        )
    )

    return {
        "dates_t1": dates[idx + 1],
        "y_true": y[idx + 1],
        "y_hat_raw": y_hat,
        "y_anchor": y[idx],
    }


def _fit_arx_residual(
    y_true: np.ndarray,
    y_hat_raw: np.ndarray,
    y_anchor: np.ndarray,
    ridge: float,
) -> dict[str, float]:
    e = y_true - y_hat_raw
    if len(e) < 4:
        raise ValueError("not enough samples to fit ARX residual model")

    d_anchor = np.zeros(len(y_anchor), dtype=float)
    d_anchor[1:] = y_anchor[1:] - y_anchor[:-1]

    target = e[1:]
    x_prev_e = e[:-1]
    x_d_anchor = d_anchor[1:]
    x = np.column_stack([np.ones(len(target)), x_prev_e, x_d_anchor])

    xtx = x.T @ x
    xtx += ridge * np.eye(xtx.shape[0], dtype=float)
    coef = np.linalg.solve(xtx, x.T @ target)
    y_fit = x @ coef

    return {
        "alpha": float(coef[0]),
        "phi": float(coef[1]),
        "theta": float(coef[2]),
        "train_r2": _r2(target, y_fit),
        "train_rmse": float(np.sqrt(np.mean((target - y_fit) ** 2))),
        "last_train_error": float(e[-1]),
        "last_train_anchor": float(y_anchor[-1]),
    }


def _apply_arx_residual(
    y_true: np.ndarray,
    y_hat_raw: np.ndarray,
    y_anchor: np.ndarray,
    arx: dict[str, float],
    *,
    prev_error_seed: float,
    prev_anchor_seed: float,
) -> dict[str, np.ndarray]:
    alpha = float(arx["alpha"])
    phi = float(arx["phi"])
    theta = float(arx["theta"])

    corr = np.zeros(len(y_hat_raw), dtype=float)
    y_hat_corr = np.zeros(len(y_hat_raw), dtype=float)
    e_prev_used = np.zeros(len(y_hat_raw), dtype=float)
    d_anchor_used = np.zeros(len(y_hat_raw), dtype=float)

    e_prev = float(prev_error_seed)
    prev_anchor = float(prev_anchor_seed)

    for i in range(len(y_hat_raw)):
        d_anchor = float(y_anchor[i] - prev_anchor)
        c_i = alpha + phi * e_prev + theta * d_anchor
        y_hat_corr[i] = y_hat_raw[i] + c_i
        corr[i] = c_i
        e_prev_used[i] = e_prev
        d_anchor_used[i] = d_anchor

        e_prev = float(y_true[i] - y_hat_corr[i])
        prev_anchor = float(y_anchor[i])

    return {
        "y_hat_corr": y_hat_corr,
        "corr_term": corr,
        "e_prev_used": e_prev_used,
        "d_anchor_used": d_anchor_used,
    }


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
                "direction_accuracy": _direction_acc(yt, yp, ya),
            }
        )
    return out


def _best_lag(scan: list[dict[str, float]]) -> dict[str, float]:
    if not scan:
        return {"lag": 0, "corr": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    return max(scan, key=lambda r: (float(r["corr"]), float(r["r2"]), -float(r["rmse"])))


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


def _turn_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_anchor: np.ndarray, eps: float, tol: int) -> dict[str, float]:
    d_true = y_true - y_anchor
    d_pred = y_pred - y_anchor
    idx_true = _turning_indices(d_true, eps=eps)
    idx_pred = _turning_indices(d_pred, eps=eps)

    used_pred: set[int] = set()
    matched = 0
    delays: list[int] = []
    pred_list = idx_pred.tolist()
    for t in idx_true.tolist():
        candidates = [p for p in pred_list if p not in used_pred and abs(p - t) <= tol]
        if not candidates:
            continue
        p_best = sorted(candidates, key=lambda p: (abs(p - t), p))[0]
        used_pred.add(p_best)
        matched += 1
        delays.append(int(p_best - t))

    n_true = int(len(idx_true))
    n_pred = int(len(idx_pred))
    recall = float(matched / n_true) if n_true > 0 else float("nan")
    precision = float(len(used_pred) / n_pred) if n_pred > 0 else float("nan")
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
        "matched_count": int(matched),
    }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_anchor: np.ndarray) -> dict[str, float]:
    return {
        "n_obs": int(len(y_true)),
        "x_r2": _r2(y_true, y_pred),
        "x_rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "x_mae": float(np.mean(np.abs(y_true - y_pred))),
        "x_corr": _corr(y_true, y_pred),
        "x_direction": _direction_acc(y_true, y_pred, y_anchor),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_all = _load_rows(Path(args.panel))
    est = json.loads(Path(args.estimation).read_text(encoding="utf-8"))
    baseline_metrics = json.loads(Path(args.baseline_metrics).read_text(encoding="utf-8"))
    baseline_phase = json.loads(Path(args.baseline_phase).read_text(encoding="utf-8"))

    feature_cols = list(est.get("betas", {}).keys())
    if not feature_cols:
        feature_cols = [c for c in rows_all[0].keys() if c.startswith("g_") or c.startswith("c_") or c.startswith("i_")]

    rows_train = _split_rows(rows_all, split=args.train_split)
    rows_test = _split_rows(rows_all, split=args.test_split)

    tr = _one_step_series(rows_train, feature_cols=feature_cols, est=est, dt=args.dt)
    te = _one_step_series(rows_test, feature_cols=feature_cols, est=est, dt=args.dt)

    arx = _fit_arx_residual(
        y_true=tr["y_true"],
        y_hat_raw=tr["y_hat_raw"],
        y_anchor=tr["y_anchor"],
        ridge=args.ridge,
    )

    ap = _apply_arx_residual(
        y_true=te["y_true"],
        y_hat_raw=te["y_hat_raw"],
        y_anchor=te["y_anchor"],
        arx=arx,
        prev_error_seed=float(arx["last_train_error"]),
        prev_anchor_seed=float(arx["last_train_anchor"]),
    )

    y_true = te["y_true"]
    y_anchor = te["y_anchor"]
    y_hat_raw = te["y_hat_raw"]
    y_hat_corr = ap["y_hat_corr"]

    m_raw = _metrics(y_true, y_hat_raw, y_anchor)
    m_corr = _metrics(y_true, y_hat_corr, y_anchor)

    lag_raw = _lag_scan(y_true, y_hat_raw, y_anchor, max_lag=args.max_lag)
    lag_corr = _lag_scan(y_true, y_hat_corr, y_anchor, max_lag=args.max_lag)
    best_raw = _best_lag(lag_raw)
    best_corr = _best_lag(lag_corr)

    turn_raw = _turn_metrics(y_true, y_hat_raw, y_anchor, eps=args.turn_eps, tol=args.turn_tol)
    turn_corr = _turn_metrics(y_true, y_hat_corr, y_anchor, eps=args.turn_eps, tol=args.turn_tol)

    summary = {
        "config": {
            "panel": str(args.panel),
            "estimation": str(args.estimation),
            "train_split": args.train_split,
            "test_split": args.test_split,
            "dt": args.dt,
            "max_lag": args.max_lag,
            "turn_eps": args.turn_eps,
            "turn_tol": args.turn_tol,
            "ridge": args.ridge,
        },
        "baseline_refs": {
            "metrics_path": str(args.baseline_metrics),
            "phase_path": str(args.baseline_phase),
            "main_x_r2": float(baseline_metrics["x_r2"]),
            "main_x_rmse": float(baseline_metrics["x_rmse"]),
            "main_best_lag": int(baseline_phase["best_lag_by_corr"]["lag"]),
            "main_turn_recall": float(baseline_phase["turn_metrics_lag0"]["turn_hit_rate_recall"]),
            "main_turn_precision": float(baseline_phase["turn_metrics_lag0"]["turn_precision"]),
            "main_turn_mean_delay": float(baseline_phase["turn_metrics_lag0"]["mean_delay"]),
        },
        "arx_model": arx,
        "test_raw_metrics": m_raw,
        "test_corr_metrics": m_corr,
        "test_raw_best_lag": best_raw,
        "test_corr_best_lag": best_corr,
        "test_raw_turn_metrics": turn_raw,
        "test_corr_turn_metrics": turn_corr,
        "gap_corr_minus_raw": {
            "x_r2": float(m_corr["x_r2"] - m_raw["x_r2"]),
            "x_rmse": float(m_corr["x_rmse"] - m_raw["x_rmse"]),
            "x_direction": float(m_corr["x_direction"] - m_raw["x_direction"]),
            "best_lag_corr": float(best_corr["corr"] - best_raw["corr"]),
            "turn_recall": float(turn_corr["turn_hit_rate_recall"] - turn_raw["turn_hit_rate_recall"]),
            "turn_precision": float(turn_corr["turn_precision"] - turn_raw["turn_precision"]),
            "turn_mean_delay": float(turn_corr["mean_delay"] - turn_raw["mean_delay"]),
        },
        "gap_corr_minus_main": {
            "x_r2": float(m_corr["x_r2"] - float(baseline_metrics["x_r2"])),
            "x_rmse": float(m_corr["x_rmse"] - float(baseline_metrics["x_rmse"])),
            "best_lag": int(best_corr["lag"] - int(baseline_phase["best_lag_by_corr"]["lag"])),
            "turn_recall": float(turn_corr["turn_hit_rate_recall"] - float(baseline_phase["turn_metrics_lag0"]["turn_hit_rate_recall"])),
            "turn_precision": float(turn_corr["turn_precision"] - float(baseline_phase["turn_metrics_lag0"]["turn_precision"])),
            "turn_mean_delay": float(turn_corr["mean_delay"] - float(baseline_phase["turn_metrics_lag0"]["mean_delay"])),
        },
    }
    (out_dir / "arx_phase_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with (out_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "date_t1",
                "y_t1_actual",
                "y_t1_pred_raw",
                "y_t1_pred_corr",
                "y_t",
                "error_raw",
                "error_corr",
                "corr_term",
                "e_prev_used",
                "d_anchor_used",
            ],
        )
        writer.writeheader()
        for i in range(len(y_true)):
            writer.writerow(
                {
                    "date_t1": te["dates_t1"][i].strftime("%Y-%m-%d"),
                    "y_t1_actual": float(y_true[i]),
                    "y_t1_pred_raw": float(y_hat_raw[i]),
                    "y_t1_pred_corr": float(y_hat_corr[i]),
                    "y_t": float(y_anchor[i]),
                    "error_raw": float(y_true[i] - y_hat_raw[i]),
                    "error_corr": float(y_true[i] - y_hat_corr[i]),
                    "corr_term": float(ap["corr_term"][i]),
                    "e_prev_used": float(ap["e_prev_used"][i]),
                    "d_anchor_used": float(ap["d_anchor_used"][i]),
                }
            )

    with (out_dir / "lag_scan_raw.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lag", "n", "r2", "corr", "rmse", "direction_accuracy"])
        writer.writeheader()
        for row in lag_raw:
            writer.writerow(row)
    with (out_dir / "lag_scan_corr.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lag", "n", "r2", "corr", "rmse", "direction_accuracy"])
        writer.writeheader()
        for row in lag_corr:
            writer.writerow(row)

    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=args.dpi)
    ax.plot(te["dates_t1"], y_true, label="actual y(t+1)", linewidth=1.25)
    ax.plot(te["dates_t1"], y_hat_raw, label="raw y_hat", linewidth=1.1)
    ax.plot(te["dates_t1"], y_hat_corr, label="ARX-corr y_hat", linewidth=1.1)
    ax.set_title("ARX Residual Correction: One-Step Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "timeseries_raw_vs_corr.png")
    plt.close(fig)

    lags_raw = [int(r["lag"]) for r in lag_raw]
    corr_raw = [float(r["corr"]) for r in lag_raw]
    rmse_raw = [float(r["rmse"]) for r in lag_raw]
    lags_corr = [int(r["lag"]) for r in lag_corr]
    corr_corr = [float(r["corr"]) for r in lag_corr]
    rmse_corr = [float(r["rmse"]) for r in lag_corr]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=args.dpi)
    axes[0].plot(lags_raw, corr_raw, marker="o", label="raw")
    axes[0].plot(lags_corr, corr_corr, marker="o", label="corr")
    axes[0].axvline(int(best_corr["lag"]), linestyle="--", linewidth=1.0)
    axes[0].set_title("Lag Scan: Correlation")
    axes[0].set_xlabel("lag")
    axes[0].set_ylabel("corr")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(lags_raw, rmse_raw, marker="o", label="raw")
    axes[1].plot(lags_corr, rmse_corr, marker="o", label="corr")
    axes[1].axvline(int(best_corr["lag"]), linestyle="--", linewidth=1.0)
    axes[1].set_title("Lag Scan: RMSE")
    axes[1].set_xlabel("lag")
    axes[1].set_ylabel("rmse")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "lag_scan_raw_vs_corr.png")
    plt.close(fig)

    print(f"[done] train_split={args.train_split}, test_split={args.test_split}")
    print(
        "[done] raw: "
        f"x_r2={m_raw['x_r2']:.6g}, x_rmse={m_raw['x_rmse']:.6g}, "
        f"dir={m_raw['x_direction']:.6g}, best_lag={int(best_raw['lag'])}"
    )
    print(
        "[done] corr: "
        f"x_r2={m_corr['x_r2']:.6g}, x_rmse={m_corr['x_rmse']:.6g}, "
        f"dir={m_corr['x_direction']:.6g}, best_lag={int(best_corr['lag'])}"
    )
    print(
        "[done] turn raw->corr: "
        f"recall {turn_raw['turn_hit_rate_recall']:.6g}->{turn_corr['turn_hit_rate_recall']:.6g}, "
        f"precision {turn_raw['turn_precision']:.6g}->{turn_corr['turn_precision']:.6g}, "
        f"mean_delay {turn_raw['mean_delay']:.6g}->{turn_corr['mean_delay']:.6g}"
    )
    print(f"[done] output: {out_dir}")


if __name__ == "__main__":
    main()

