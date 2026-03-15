"""Step 16 experiment: observable error-state (Kalman-style) phase correction.

Causal update for phase state:
    z_pred_t = rho * z_{t-1}
    obs_t = y_t - y_hat_raw_{t-1}         (available at time t)
    z_t = z_pred_t + K * (obs_t - z_pred_t)
    y_hat_corr_{t+1|t} = y_hat_raw_{t+1|t} + gamma * z_t

This keeps delay as an explicit dynamic state rather than hiding it in static betas.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan observable error-state phase correction.")
    parser.add_argument("--panel", default="data/us/us_regression_panel.csv")
    parser.add_argument("--estimation", default="data/us/us_ols_estimation.json")
    parser.add_argument("--baseline-metrics", default="data/us/plot/fit_metrics.json")
    parser.add_argument("--baseline-phase", default="data/us/experiments/step5_phase_bias_baseline/phase_bias_summary.json")
    parser.add_argument("--output-dir", default="data/us/experiments/step16_error_state_kalman")
    parser.add_argument("--selection-split", choices=["train", "valid", "test", "all"], default="valid")
    parser.add_argument("--eval-split", choices=["train", "valid", "test", "all"], default="test")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--turn-eps", type=float, default=1e-6)
    parser.add_argument("--turn-tol", type=int, default=1)
    parser.add_argument("--rho-grid", default="0.0:0.95:0.05")
    parser.add_argument("--k-grid", default="0.0:1.0:0.1")
    parser.add_argument("--gamma-grid", default="0.0:1.5:0.1")
    parser.add_argument("--z-clip", type=float, default=0.2, help="Clip latent state to [-z_clip, z_clip]. <=0 means no clip.")
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
        raise ValueError(f"empty panel: {path}")
    return rows


def _split_rows(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    if split == "all":
        return rows
    col = f"is_{split}"
    out = [r for r in rows if (r.get(col) or "").strip() == "1"]
    if not out:
        raise ValueError(f"no rows for split={split}")
    return out


def _one_step_raw(rows: list[dict[str, str]], feature_cols: list[str], est: dict, dt: float) -> dict[str, np.ndarray]:
    dates = np.array([datetime.strptime(r["date"], "%Y-%m-%d") for r in rows])
    y = np.array([_to_float(r["Y"]) for r in rows], dtype=float)
    u = np.array([[_to_float(r[c]) for c in feature_cols] for r in rows], dtype=float)
    finite = np.isfinite(y) & np.all(np.isfinite(u), axis=1)
    y = y[finite]
    u = u[finite]
    dates = dates[finite]
    if bool(est.get("orthogonalize_country", False)):
        u = _orthogonalize_country_features(u, feature_cols)

    m = float(est["m"])
    c = float(est["c"])
    c_nl = float(est.get("c_nl", 0.0))
    k = float(est["k"])
    intercept = float(est.get("intercept", 0.0))
    beta_vec = np.array([float(est["betas"][name]) / m for name in feature_cols], dtype=float)

    idx = np.arange(1, len(y) - 1, dtype=int)
    v = (y[idx] - y[idx - 1]) / dt
    dt2 = dt * dt
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
    return {
        "date_t1": dates[idx + 1],
        "y_true": y[idx + 1],
        "y_hat_raw": y_hat,
        "y_anchor": y[idx],
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
        "x_r2": _r2(y_true, y_pred),
        "x_rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "x_mae": float(np.mean(np.abs(y_true - y_pred))),
        "x_corr": _corr(y_true, y_pred),
        "x_direction": _direction_acc(y_true, y_pred, y_anchor),
    }


def _parse_grid(raw: str) -> np.ndarray:
    parts = [p.strip() for p in raw.split(":")]
    if len(parts) != 3:
        raise ValueError(f"invalid grid spec: {raw}")
    start = float(parts[0])
    end = float(parts[1])
    step = float(parts[2])
    if step <= 0:
        raise ValueError("grid step must be > 0")
    n = int(np.floor((end - start) / step + 1e-12)) + 1
    vals = start + np.arange(n + 1, dtype=float) * step
    vals = vals[vals <= end + 1e-12]
    return np.round(vals, 10)


def _apply_error_state(
    y_true: np.ndarray,
    y_hat_raw: np.ndarray,
    y_anchor: np.ndarray,
    *,
    rho: float,
    k_gain: float,
    gamma: float,
    z_clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(y_hat_raw)
    z_series = np.zeros(n, dtype=float)
    obs_series = np.zeros(n, dtype=float)
    y_corr = np.zeros(n, dtype=float)
    do_clip = z_clip > 0
    z_prev = 0.0
    prev_raw_error = 0.0
    for i in range(n):
        if i == 0:
            obs = 0.0
        else:
            obs = float(y_anchor[i] - y_hat_raw[i - 1])
        z_pred = rho * z_prev
        z_t = z_pred + k_gain * (obs - z_pred)
        if do_clip:
            z_t = float(np.clip(z_t, -z_clip, z_clip))
        y_corr[i] = y_hat_raw[i] + gamma * z_t
        z_series[i] = z_t
        obs_series[i] = obs
        z_prev = z_t
        prev_raw_error = float(y_true[i] - y_hat_raw[i])
        _ = prev_raw_error
    return y_corr, z_series, obs_series


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_est = json.loads(Path(args.estimation).read_text(encoding="utf-8"))
    baseline_metrics = json.loads(Path(args.baseline_metrics).read_text(encoding="utf-8"))
    baseline_phase = json.loads(Path(args.baseline_phase).read_text(encoding="utf-8"))
    rows_all = _load_rows(Path(args.panel))

    feature_cols = list(baseline_est.get("betas", {}).keys())
    if not feature_cols:
        feature_cols = [c for c in rows_all[0].keys() if c.startswith("g_") or c.startswith("c_") or c.startswith("i_")]

    sel = _one_step_raw(_split_rows(rows_all, args.selection_split), feature_cols, baseline_est, dt=args.dt)
    eva = _one_step_raw(_split_rows(rows_all, args.eval_split), feature_cols, baseline_est, dt=args.dt)

    rho_grid = _parse_grid(args.rho_grid)
    k_grid = _parse_grid(args.k_grid)
    gamma_grid = _parse_grid(args.gamma_grid)

    rows_scan: list[dict[str, float]] = []
    for rho in rho_grid:
        for k_gain in k_grid:
            for gamma in gamma_grid:
                y_corr_sel, _, _ = _apply_error_state(
                    sel["y_true"],
                    sel["y_hat_raw"],
                    sel["y_anchor"],
                    rho=float(rho),
                    k_gain=float(k_gain),
                    gamma=float(gamma),
                    z_clip=float(args.z_clip),
                )
                m = _metrics(sel["y_true"], y_corr_sel, sel["y_anchor"])
                best = _best_lag(_lag_scan(sel["y_true"], y_corr_sel, sel["y_anchor"], max_lag=args.max_lag))
                rows_scan.append(
                    {
                        "rho": float(rho),
                        "k_gain": float(k_gain),
                        "gamma": float(gamma),
                        "sel_x_r2": float(m["x_r2"]),
                        "sel_x_rmse": float(m["x_rmse"]),
                        "sel_x_corr": float(m["x_corr"]),
                        "sel_x_direction": float(m["x_direction"]),
                        "sel_best_lag": int(best["lag"]),
                        "sel_best_lag_corr": float(best["corr"]),
                    }
                )

    def _rank_key(r: dict[str, float]) -> tuple[float, float, float, float]:
        return (
            abs(int(r["sel_best_lag"])),
            float(r["sel_x_rmse"]),
            -float(r["sel_x_corr"]),
            -float(r["sel_x_r2"]),
        )

    best_row = sorted(rows_scan, key=_rank_key)[0]
    rho_best = float(best_row["rho"])
    k_best = float(best_row["k_gain"])
    gamma_best = float(best_row["gamma"])

    y_raw = eva["y_hat_raw"]
    y_corr, z_corr, obs_corr = _apply_error_state(
        eva["y_true"],
        y_raw,
        eva["y_anchor"],
        rho=rho_best,
        k_gain=k_best,
        gamma=gamma_best,
        z_clip=float(args.z_clip),
    )

    m_raw = _metrics(eva["y_true"], y_raw, eva["y_anchor"])
    m_corr = _metrics(eva["y_true"], y_corr, eva["y_anchor"])
    lag_raw = _lag_scan(eva["y_true"], y_raw, eva["y_anchor"], max_lag=args.max_lag)
    lag_corr = _lag_scan(eva["y_true"], y_corr, eva["y_anchor"], max_lag=args.max_lag)
    best_raw = _best_lag(lag_raw)
    best_corr = _best_lag(lag_corr)
    turn_raw = _turn_metrics(eva["y_true"], y_raw, eva["y_anchor"], eps=args.turn_eps, tol=args.turn_tol)
    turn_corr = _turn_metrics(eva["y_true"], y_corr, eva["y_anchor"], eps=args.turn_eps, tol=args.turn_tol)

    summary = {
        "config": {
            "panel": str(args.panel),
            "estimation": str(args.estimation),
            "selection_split": args.selection_split,
            "eval_split": args.eval_split,
            "dt": args.dt,
            "max_lag": args.max_lag,
            "turn_eps": args.turn_eps,
            "turn_tol": args.turn_tol,
            "rho_grid": [float(x) for x in rho_grid.tolist()],
            "k_grid": [float(x) for x in k_grid.tolist()],
            "gamma_grid": [float(x) for x in gamma_grid.tolist()],
            "z_clip": float(args.z_clip),
        },
        "chosen": best_row,
        "test_raw_metrics": m_raw,
        "test_state_metrics": m_corr,
        "test_raw_best_lag": best_raw,
        "test_state_best_lag": best_corr,
        "test_raw_turn_metrics": turn_raw,
        "test_state_turn_metrics": turn_corr,
        "gap_state_minus_raw": {
            "x_r2": float(m_corr["x_r2"] - m_raw["x_r2"]),
            "x_rmse": float(m_corr["x_rmse"] - m_raw["x_rmse"]),
            "best_lag": int(best_corr["lag"] - best_raw["lag"]),
            "turn_recall": float(turn_corr["turn_hit_rate_recall"] - turn_raw["turn_hit_rate_recall"]),
            "turn_precision": float(turn_corr["turn_precision"] - turn_raw["turn_precision"]),
            "turn_mean_delay": float(turn_corr["mean_delay"] - turn_raw["mean_delay"]),
        },
        "gap_state_minus_main": {
            "x_r2": float(m_corr["x_r2"] - float(baseline_metrics["x_r2"])),
            "x_rmse": float(m_corr["x_rmse"] - float(baseline_metrics["x_rmse"])),
            "best_lag": int(best_corr["lag"] - int(baseline_phase["best_lag_by_corr"]["lag"])),
            "turn_recall": float(turn_corr["turn_hit_rate_recall"] - float(baseline_phase["turn_metrics_lag0"]["turn_hit_rate_recall"])),
            "turn_precision": float(turn_corr["turn_precision"] - float(baseline_phase["turn_metrics_lag0"]["turn_precision"])),
            "turn_mean_delay": float(turn_corr["mean_delay"] - float(baseline_phase["turn_metrics_lag0"]["mean_delay"])),
        },
    }
    (out_dir / "error_state_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with (out_dir / "error_state_grid_scan.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rho",
                "k_gain",
                "gamma",
                "sel_x_r2",
                "sel_x_rmse",
                "sel_x_corr",
                "sel_x_direction",
                "sel_best_lag",
                "sel_best_lag_corr",
            ],
        )
        writer.writeheader()
        for r in rows_scan:
            writer.writerow(r)

    with (out_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "date_t1",
                "y_t1_actual",
                "y_t1_pred_raw",
                "y_t1_pred_state",
                "z_state",
                "obs_prev_error",
                "y_t",
            ],
        )
        writer.writeheader()
        for i in range(len(eva["y_true"])):
            writer.writerow(
                {
                    "date_t1": eva["date_t1"][i].strftime("%Y-%m-%d"),
                    "y_t1_actual": float(eva["y_true"][i]),
                    "y_t1_pred_raw": float(y_raw[i]),
                    "y_t1_pred_state": float(y_corr[i]),
                    "z_state": float(z_corr[i]),
                    "obs_prev_error": float(obs_corr[i]),
                    "y_t": float(eva["y_anchor"][i]),
                }
            )

    with (out_dir / "lag_scan_raw.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lag", "n", "r2", "corr", "rmse", "direction_accuracy"])
        writer.writeheader()
        for row in lag_raw:
            writer.writerow(row)

    with (out_dir / "lag_scan_state.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lag", "n", "r2", "corr", "rmse", "direction_accuracy"])
        writer.writeheader()
        for row in lag_corr:
            writer.writerow(row)

    print("[done] step16 error-state scan complete")
    print(
        "[done] chosen: "
        f"rho={rho_best:.4g}, K={k_best:.4g}, gamma={gamma_best:.4g}, "
        f"sel_best_lag={int(best_row['sel_best_lag'])}, sel_x_r2={float(best_row['sel_x_r2']):.6g}"
    )
    print(
        f"[done] raw->state ({args.eval_split}): "
        f"x_r2 {m_raw['x_r2']:.6g}->{m_corr['x_r2']:.6g}, "
        f"x_rmse {m_raw['x_rmse']:.6g}->{m_corr['x_rmse']:.6g}, "
        f"best_lag {int(best_raw['lag'])}->{int(best_corr['lag'])}"
    )
    print(f"[done] output: {out_dir}")


if __name__ == "__main__":
    main()

