"""Step 18 experiment: recursive state dynamics with multi-step consistency.

Model form (causal):
    delta_{t+1} = rho * delta_t + r_theta(s_t)
    y_hat_{t+1} = y_t + delta_{t+1} + phase_gain * delta_t
where state s_t = [y_t, delta_t, accel_t, u_t].

Selection is based on:
1) phase criterion (|best_lag| close to 0),
2) multi-step consistency RMSE on horizons (default h=1,2,3).
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan recursive state dynamics with multi-step consistency.")
    parser.add_argument("--panel", default="data/us/us_regression_panel.csv")
    parser.add_argument("--baseline-est", default="data/us/us_ols_estimation.json")
    parser.add_argument("--baseline-metrics", default="data/us/plot/fit_metrics.json")
    parser.add_argument("--baseline-phase", default="data/us/experiments/step5_phase_bias_baseline/phase_bias_summary.json")
    parser.add_argument("--output-dir", default="data/us/experiments/step18_multistep_consistency_state")
    parser.add_argument("--fit-split", choices=["train", "valid", "test", "all"], default="train")
    parser.add_argument("--selection-split", choices=["train", "valid", "test", "all"], default="valid")
    parser.add_argument("--eval-split", choices=["train", "valid", "test", "all"], default="test")
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--turn-eps", type=float, default=1e-6)
    parser.add_argument("--turn-tol", type=int, default=1)
    parser.add_argument("--alphas", default="0.001,0.005,0.01,0.05,0.1,0.2,0.5,1,2,5,10")
    parser.add_argument("--rho-grid", default="0.0:1.0:0.1")
    parser.add_argument("--phase-gain-grid", default="-0.4:0.4:0.1")
    parser.add_argument("--multi-horizons", default="1,2,3")
    parser.add_argument("--multi-weights", default="1.0,0.7,0.5")
    parser.add_argument("--orthogonalize-country", action="store_true")
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


def _parse_float_list(raw: str) -> list[float]:
    vals = []
    for p in raw.split(","):
        t = p.strip()
        if not t:
            continue
        vals.append(float(t))
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("empty float list")
    return vals


def _parse_int_list(raw: str) -> list[int]:
    vals = []
    for p in raw.split(","):
        t = p.strip()
        if not t:
            continue
        vals.append(int(t))
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("empty int list")
    return vals


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


def _prepare_split_series(
    rows: list[dict[str, str]],
    feature_cols: list[str],
    *,
    orthogonalize_country: bool,
) -> dict[str, np.ndarray]:
    dates = np.array([datetime.strptime(r["date"], "%Y-%m-%d") for r in rows])
    y = np.array([_to_float(r["Y"]) for r in rows], dtype=float)
    u = np.array([[_to_float(r[c]) for c in feature_cols] for r in rows], dtype=float)

    finite = np.isfinite(y) & np.all(np.isfinite(u), axis=1)
    y = y[finite]
    u = u[finite]
    dates = dates[finite]

    if orthogonalize_country:
        u = _orthogonalize_country_features(u, feature_cols)

    return {"dates": dates, "y": y, "u": u}


def _build_supervised(series: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    y = series["y"]
    u = series["u"]
    dates = series["dates"]
    idx = np.arange(2, len(y) - 1, dtype=int)  # t index with target at t+1
    if len(idx) < 5:
        raise ValueError("not enough rows for supervised set")

    y_t = y[idx]
    y_tm1 = y[idx - 1]
    y_tm2 = y[idx - 2]
    y_tp1 = y[idx + 1]

    delta_t = y_t - y_tm1
    delta_tm1 = y_tm1 - y_tm2
    accel_t = delta_t - delta_tm1
    delta_tp1 = y_tp1 - y_t
    x = np.column_stack([y_t, delta_t, accel_t, u[idx]])

    return {
        "idx_t": idx,
        "date_t1": dates[idx + 1],
        "x": x,
        "y_anchor": y_t,
        "delta_t": delta_t,
        "delta_target": delta_tp1,
        "y_true": y_tp1,
    }


def _fit_residual_ridge(
    sup: dict[str, np.ndarray],
    *,
    rho: float,
    alpha: float,
) -> dict[str, np.ndarray | float]:
    x = sup["x"]
    delta_t = sup["delta_t"]
    target = sup["delta_target"] - float(rho) * delta_t

    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    sigma[sigma < 1e-12] = 1.0
    xs = (x - mu) / sigma
    z = np.column_stack([np.ones(len(xs)), xs])
    p = z.shape[1]
    reg = np.eye(p, dtype=float)
    reg[0, 0] = 0.0
    coef = np.linalg.solve(z.T @ z + float(alpha) * reg, z.T @ target)
    fit_hat = z @ coef
    return {
        "rho": float(rho),
        "alpha": float(alpha),
        "mu": mu,
        "sigma": sigma,
        "coef": coef,
        "fit_resid_r2": _r2(target, fit_hat),
    }


def _predict_residual(x: np.ndarray, model: dict[str, np.ndarray | float]) -> np.ndarray:
    mu = np.asarray(model["mu"], dtype=float)
    sigma = np.asarray(model["sigma"], dtype=float)
    coef = np.asarray(model["coef"], dtype=float)
    xs = (x - mu) / sigma
    z = np.column_stack([np.ones(len(xs)), xs])
    return z @ coef


def _predict_one_step(
    sup: dict[str, np.ndarray],
    model: dict[str, np.ndarray | float],
    *,
    phase_gain: float,
) -> np.ndarray:
    r_hat = _predict_residual(sup["x"], model)
    rho = float(model["rho"])
    delta_hat = rho * sup["delta_t"] + r_hat
    return sup["y_anchor"] + delta_hat + float(phase_gain) * sup["delta_t"]


def _rollout_multistep_rmse(
    series: dict[str, np.ndarray],
    model: dict[str, np.ndarray | float],
    *,
    phase_gain: float,
    horizons: list[int],
) -> dict[str, float]:
    y = series["y"]
    u = series["u"]
    rho = float(model["rho"])
    rmse_by_h: dict[str, float] = {}
    for h in horizons:
        errs: list[float] = []
        for i in range(2, len(y) - h):
            y_tm2 = float(y[i - 2])
            y_tm1 = float(y[i - 1])
            y_t = float(y[i])
            delta_prev = y_tm1 - y_tm2
            delta_curr = y_t - y_tm1
            y_curr = y_t
            for s in range(1, h + 1):
                j = i + s - 1
                accel_curr = delta_curr - delta_prev
                x_step = np.concatenate(([y_curr, delta_curr, accel_curr], u[j]), axis=0).reshape(1, -1)
                r_hat = float(_predict_residual(x_step, model)[0])
                delta_next = rho * delta_curr + r_hat
                y_next = y_curr + delta_next + float(phase_gain) * delta_curr
                delta_prev, delta_curr = delta_curr, delta_next
                y_curr = y_next
            true_h = float(y[i + h])
            errs.append(y_curr - true_h)
        if errs:
            rmse_by_h[f"h{h}"] = float(np.sqrt(np.mean(np.square(np.array(errs, dtype=float)))))
        else:
            rmse_by_h[f"h{h}"] = float("nan")
    return rmse_by_h


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


def _rmse_combo(rmse_map: dict[str, float], horizons: list[int], weights: list[float]) -> float:
    total = 0.0
    wsum = 0.0
    for h, w in zip(horizons, weights):
        v = float(rmse_map.get(f"h{h}", float("nan")))
        if np.isfinite(v):
            total += float(w) * v
            wsum += float(w)
    if wsum <= 0:
        return float("nan")
    return total / wsum


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_est = json.loads(Path(args.baseline_est).read_text(encoding="utf-8"))
    baseline_metrics = json.loads(Path(args.baseline_metrics).read_text(encoding="utf-8"))
    baseline_phase = json.loads(Path(args.baseline_phase).read_text(encoding="utf-8"))
    rows_all = _load_rows(Path(args.panel))

    feature_cols = list(baseline_est.get("betas", {}).keys())
    if not feature_cols:
        feature_cols = [c for c in rows_all[0].keys() if c.startswith("g_") or c.startswith("c_") or c.startswith("i_")]

    orth_flag = bool(args.orthogonalize_country or baseline_est.get("orthogonalize_country", False))
    horizons = _parse_int_list(args.multi_horizons)
    weights = _parse_float_list(args.multi_weights)
    if len(weights) != len(horizons):
        raise ValueError("multi-weights length must equal multi-horizons length")
    alphas = _parse_float_list(args.alphas)
    rho_grid = _parse_grid(args.rho_grid)
    phase_grid = _parse_grid(args.phase_gain_grid)

    fit_series = _prepare_split_series(_split_rows(rows_all, args.fit_split), feature_cols, orthogonalize_country=orth_flag)
    sel_series = _prepare_split_series(_split_rows(rows_all, args.selection_split), feature_cols, orthogonalize_country=orth_flag)
    eval_series = _prepare_split_series(_split_rows(rows_all, args.eval_split), feature_cols, orthogonalize_country=orth_flag)

    fit_sup = _build_supervised(fit_series)
    sel_sup = _build_supervised(sel_series)
    eval_sup = _build_supervised(eval_series)

    candidates: list[dict[str, object]] = []
    for alpha in alphas:
        for rho in rho_grid:
            model = _fit_residual_ridge(fit_sup, rho=float(rho), alpha=float(alpha))
            for phase_gain in phase_grid:
                sel_y_hat = _predict_one_step(sel_sup, model, phase_gain=float(phase_gain))
                sel_m = _metrics(sel_sup["y_true"], sel_y_hat, sel_sup["y_anchor"])
                sel_lag = _best_lag(_lag_scan(sel_sup["y_true"], sel_y_hat, sel_sup["y_anchor"], max_lag=args.max_lag))
                sel_rmse_h = _rollout_multistep_rmse(
                    sel_series,
                    model,
                    phase_gain=float(phase_gain),
                    horizons=horizons,
                )
                sel_combo = _rmse_combo(sel_rmse_h, horizons=horizons, weights=weights)

                eval_y_hat = _predict_one_step(eval_sup, model, phase_gain=float(phase_gain))
                eval_m = _metrics(eval_sup["y_true"], eval_y_hat, eval_sup["y_anchor"])
                eval_lag = _best_lag(_lag_scan(eval_sup["y_true"], eval_y_hat, eval_sup["y_anchor"], max_lag=args.max_lag))
                eval_turn = _turn_metrics(eval_sup["y_true"], eval_y_hat, eval_sup["y_anchor"], eps=args.turn_eps, tol=args.turn_tol)
                eval_rmse_h = _rollout_multistep_rmse(
                    eval_series,
                    model,
                    phase_gain=float(phase_gain),
                    horizons=horizons,
                )
                eval_combo = _rmse_combo(eval_rmse_h, horizons=horizons, weights=weights)

                rec = {
                    "alpha": float(alpha),
                    "rho": float(rho),
                    "phase_gain": float(phase_gain),
                    "fit_resid_r2": float(model["fit_resid_r2"]),
                    "selection": {
                        "metrics": sel_m,
                        "best_lag": int(sel_lag["lag"]),
                        "best_lag_corr": float(sel_lag["corr"]),
                        "multistep_rmse": sel_rmse_h,
                        "multistep_combo_rmse": float(sel_combo),
                    },
                    "eval": {
                        "metrics": eval_m,
                        "best_lag": int(eval_lag["lag"]),
                        "best_lag_corr": float(eval_lag["corr"]),
                        "turn": eval_turn,
                        "multistep_rmse": eval_rmse_h,
                        "multistep_combo_rmse": float(eval_combo),
                    },
                }
                rec["gap_eval_vs_main"] = {
                    "x_r2": float(eval_m["x_r2"] - float(baseline_metrics["x_r2"])),
                    "x_rmse": float(eval_m["x_rmse"] - float(baseline_metrics["x_rmse"])),
                    "best_lag": int(int(eval_lag["lag"]) - int(baseline_phase["best_lag_by_corr"]["lag"])),
                    "turn_recall": float(eval_turn["turn_hit_rate_recall"] - float(baseline_phase["turn_metrics_lag0"]["turn_hit_rate_recall"])),
                    "turn_precision": float(eval_turn["turn_precision"] - float(baseline_phase["turn_metrics_lag0"]["turn_precision"])),
                    "turn_mean_delay": float(eval_turn["mean_delay"] - float(baseline_phase["turn_metrics_lag0"]["mean_delay"])),
                }
                candidates.append(rec)

    best_by_selection = sorted(
        candidates,
        key=lambda r: (
            abs(int(r["selection"]["best_lag"])),
            float(r["selection"]["multistep_combo_rmse"]),
            float(r["selection"]["metrics"]["x_rmse"]),
            -float(r["selection"]["metrics"]["x_corr"]),
        ),
    )[0]
    best_by_eval = sorted(
        candidates,
        key=lambda r: (
            abs(int(r["eval"]["best_lag"])),
            float(r["eval"]["multistep_combo_rmse"]),
            float(r["eval"]["metrics"]["x_rmse"]),
            -float(r["eval"]["metrics"]["x_corr"]),
        ),
    )[0]

    chosen_model = _fit_residual_ridge(
        fit_sup,
        rho=float(best_by_selection["rho"]),
        alpha=float(best_by_selection["alpha"]),
    )
    chosen_eval_hat = _predict_one_step(eval_sup, chosen_model, phase_gain=float(best_by_selection["phase_gain"]))

    summary = {
        "config": {
            "panel": str(args.panel),
            "fit_split": args.fit_split,
            "selection_split": args.selection_split,
            "eval_split": args.eval_split,
            "alphas": alphas,
            "rho_grid": [float(x) for x in rho_grid.tolist()],
            "phase_gain_grid": [float(x) for x in phase_grid.tolist()],
            "multi_horizons": horizons,
            "multi_weights": weights,
            "orthogonalize_country": orth_flag,
            "max_lag": args.max_lag,
            "turn_eps": args.turn_eps,
            "turn_tol": args.turn_tol,
        },
        "baseline": {
            "x_r2": float(baseline_metrics["x_r2"]),
            "x_rmse": float(baseline_metrics["x_rmse"]),
            "best_lag": int(baseline_phase["best_lag_by_corr"]["lag"]),
            "turn_recall": float(baseline_phase["turn_metrics_lag0"]["turn_hit_rate_recall"]),
            "turn_precision": float(baseline_phase["turn_metrics_lag0"]["turn_precision"]),
            "turn_mean_delay": float(baseline_phase["turn_metrics_lag0"]["mean_delay"]),
        },
        "best_by_selection": best_by_selection,
        "best_by_eval": best_by_eval,
        "candidates": candidates,
    }
    (out_dir / "multistep_state_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with (out_dir / "multistep_state_scan.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "alpha",
                "rho",
                "phase_gain",
                "fit_resid_r2",
                "sel_x_r2",
                "sel_x_rmse",
                "sel_x_corr",
                "sel_best_lag",
                "sel_combo_rmse",
                "eval_x_r2",
                "eval_x_rmse",
                "eval_x_corr",
                "eval_best_lag",
                "eval_combo_rmse",
                "eval_turn_recall",
                "eval_turn_precision",
                "eval_turn_mean_delay",
                "gap_eval_x_r2_vs_main",
                "gap_eval_x_rmse_vs_main",
                "gap_eval_best_lag_vs_main",
            ]
            + [f"sel_rmse_h{h}" for h in horizons]
            + [f"eval_rmse_h{h}" for h in horizons],
        )
        writer.writeheader()
        for r in candidates:
            row = {
                "alpha": r["alpha"],
                "rho": r["rho"],
                "phase_gain": r["phase_gain"],
                "fit_resid_r2": r["fit_resid_r2"],
                "sel_x_r2": r["selection"]["metrics"]["x_r2"],
                "sel_x_rmse": r["selection"]["metrics"]["x_rmse"],
                "sel_x_corr": r["selection"]["metrics"]["x_corr"],
                "sel_best_lag": r["selection"]["best_lag"],
                "sel_combo_rmse": r["selection"]["multistep_combo_rmse"],
                "eval_x_r2": r["eval"]["metrics"]["x_r2"],
                "eval_x_rmse": r["eval"]["metrics"]["x_rmse"],
                "eval_x_corr": r["eval"]["metrics"]["x_corr"],
                "eval_best_lag": r["eval"]["best_lag"],
                "eval_combo_rmse": r["eval"]["multistep_combo_rmse"],
                "eval_turn_recall": r["eval"]["turn"]["turn_hit_rate_recall"],
                "eval_turn_precision": r["eval"]["turn"]["turn_precision"],
                "eval_turn_mean_delay": r["eval"]["turn"]["mean_delay"],
                "gap_eval_x_r2_vs_main": r["gap_eval_vs_main"]["x_r2"],
                "gap_eval_x_rmse_vs_main": r["gap_eval_vs_main"]["x_rmse"],
                "gap_eval_best_lag_vs_main": r["gap_eval_vs_main"]["best_lag"],
            }
            for h in horizons:
                row[f"sel_rmse_h{h}"] = r["selection"]["multistep_rmse"].get(f"h{h}", float("nan"))
                row[f"eval_rmse_h{h}"] = r["eval"]["multistep_rmse"].get(f"h{h}", float("nan"))
            writer.writerow(row)

    with (out_dir / "test_predictions_best_by_selection.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["date_t1", "y_t1_actual", "y_t1_pred", "y_t", "alpha", "rho", "phase_gain"],
        )
        writer.writeheader()
        for i in range(len(eval_sup["y_true"])):
            writer.writerow(
                {
                    "date_t1": eval_sup["date_t1"][i].strftime("%Y-%m-%d"),
                    "y_t1_actual": float(eval_sup["y_true"][i]),
                    "y_t1_pred": float(chosen_eval_hat[i]),
                    "y_t": float(eval_sup["y_anchor"][i]),
                    "alpha": float(best_by_selection["alpha"]),
                    "rho": float(best_by_selection["rho"]),
                    "phase_gain": float(best_by_selection["phase_gain"]),
                }
            )

    print("[done] step18 multistep-consistency-state scan complete")
    print(
        "[done] best_by_selection: "
        f"alpha={best_by_selection['alpha']:.6g}, rho={best_by_selection['rho']:.6g}, "
        f"phase_gain={best_by_selection['phase_gain']:.6g}, sel_best_lag={int(best_by_selection['selection']['best_lag'])}, "
        f"eval_best_lag={int(best_by_selection['eval']['best_lag'])}, eval_x_r2={float(best_by_selection['eval']['metrics']['x_r2']):.6g}"
    )
    print(
        "[done] best_by_eval: "
        f"alpha={best_by_eval['alpha']:.6g}, rho={best_by_eval['rho']:.6g}, phase_gain={best_by_eval['phase_gain']:.6g}, "
        f"eval_best_lag={int(best_by_eval['eval']['best_lag'])}, eval_x_r2={float(best_by_eval['eval']['metrics']['x_r2']):.6g}, "
        f"eval_combo_rmse={float(best_by_eval['eval']['multistep_combo_rmse']):.6g}"
    )
    print(f"[done] output: {out_dir}")


if __name__ == "__main__":
    main()

