"""Step 22 experiment: phase-aware training loss (in-training, not post-hoc).

Delta model:
    delta_hat = X * theta
    y_hat = y_anchor + delta_hat

Training objective:
    L = L_amp + lambda_phase * L_phase + lambda_turn * L_turn + L2(theta)
where
    L_amp   = MSE(y_hat - y_true)
    L_phase = MSE(diff(y_hat) - diff(y_true))   # phase timing term
    L_turn  = MSE(sigmoid(k*delta_hat) - sigmoid(k*delta_true))
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan phase-aware in-training loss.")
    parser.add_argument("--panel", default="data/us/us_regression_panel.csv")
    parser.add_argument("--baseline-est", default="data/us/us_ols_estimation.json")
    parser.add_argument("--baseline-metrics", default="data/us/plot/fit_metrics.json")
    parser.add_argument("--baseline-phase", default="data/us/experiments/step5_phase_bias_baseline/phase_bias_summary.json")
    parser.add_argument("--output-dir", default="data/us/experiments/step22_phase_aware_training_loss")
    parser.add_argument("--fit-split", choices=["train", "valid", "test", "all"], default="train")
    parser.add_argument("--selection-split", choices=["train", "valid", "test", "all"], default="valid")
    parser.add_argument("--eval-split", choices=["train", "valid", "test", "all"], default="test")
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--turn-eps", type=float, default=1e-6)
    parser.add_argument("--turn-tol", type=int, default=1)
    parser.add_argument("--phase-lambda-grid", default="0,0.2,0.5,1,2,5")
    parser.add_argument("--turn-lambda-grid", default="0,0.1,0.2,0.5,1")
    parser.add_argument("--ridge-grid", default="1e-4,5e-4,1e-3,5e-3,1e-2,5e-2")
    parser.add_argument("--turn-k", type=float, default=8.0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--iters", type=int, default=2000)
    return parser.parse_args()


def _to_float(value: str) -> float:
    text = (value or "").strip()
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


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


def _prepare_dataset(
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

    idx = np.arange(2, len(y) - 1, dtype=int)
    if len(idx) < 6:
        raise ValueError("not enough rows in split")

    y_t = y[idx]
    y_tm1 = y[idx - 1]
    y_tm2 = y[idx - 2]
    y_tp1 = y[idx + 1]
    delta_t = y_t - y_tm1
    delta_tm1 = y_tm1 - y_tm2
    accel_t = delta_t - delta_tm1
    delta_true = y_tp1 - y_t

    x_no_bias = np.column_stack([y_t, delta_t, accel_t, u[idx]])
    return {
        "date_t1": dates[idx + 1],
        "y_true": y_tp1,
        "y_anchor": y_t,
        "delta_true": delta_true,
        "x_no_bias": x_no_bias,
    }


def _build_standardized_train(x_no_bias: np.ndarray) -> dict[str, np.ndarray]:
    mu = np.mean(x_no_bias, axis=0)
    sigma = np.std(x_no_bias, axis=0)
    sigma[sigma < 1e-12] = 1.0
    xs = (x_no_bias - mu) / sigma
    x = np.column_stack([np.ones(len(xs)), xs])
    return {"x": x, "mu": mu, "sigma": sigma}


def _apply_standardize(x_no_bias: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    xs = (x_no_bias - mu) / sigma
    return np.column_stack([np.ones(len(xs)), xs])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    zc = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-zc))


def _phase_grad_from_diff(res: np.ndarray) -> np.ndarray:
    # res length n-1 for diff residual; return grad wrt y_hat length n
    n = len(res) + 1
    g = np.zeros(n, dtype=float)
    c = 2.0 / max(len(res), 1)
    g[0] = -c * res[0]
    g[-1] = c * res[-1]
    if n > 2:
        g[1:-1] = c * (res[:-1] - res[1:])
    return g


def _train_phase_aware(
    x: np.ndarray,
    y_anchor: np.ndarray,
    y_true: np.ndarray,
    delta_true: np.ndarray,
    *,
    lambda_phase: float,
    lambda_turn: float,
    ridge_alpha: float,
    turn_k: float,
    lr: float,
    iters: int,
) -> dict[str, np.ndarray | float]:
    n, p = x.shape
    theta = np.zeros(p, dtype=float)

    for it in range(iters):
        delta_hat = x @ theta
        y_hat = y_anchor + delta_hat

        err_amp = y_hat - y_true
        loss_amp = float(np.mean(err_amp * err_amp))
        grad_amp = (2.0 / n) * (x.T @ err_amp)

        d_hat = y_hat[1:] - y_hat[:-1]
        d_true = y_true[1:] - y_true[:-1]
        err_phase = d_hat - d_true
        loss_phase = float(np.mean(err_phase * err_phase))
        g_y_phase = _phase_grad_from_diff(err_phase)
        grad_phase = x.T @ g_y_phase

        p_hat = _sigmoid(float(turn_k) * delta_hat)
        p_true = _sigmoid(float(turn_k) * delta_true)
        err_turn = p_hat - p_true
        loss_turn = float(np.mean(err_turn * err_turn))
        g_delta_turn = (2.0 / n) * err_turn * (float(turn_k) * p_hat * (1.0 - p_hat))
        grad_turn = x.T @ g_delta_turn

        reg = theta.copy()
        reg[0] = 0.0
        grad_reg = 2.0 * float(ridge_alpha) * reg

        grad = grad_amp + float(lambda_phase) * grad_phase + float(lambda_turn) * grad_turn + grad_reg
        step = float(lr) / np.sqrt(1.0 + 0.005 * it)
        theta -= step * grad

    delta_hat = x @ theta
    y_hat = y_anchor + delta_hat
    fit_err = y_hat - y_true
    return {
        "theta": theta,
        "fit_x_r2": _r2(y_true, y_hat),
        "fit_x_rmse": float(np.sqrt(np.mean(fit_err * fit_err))),
    }


def _predict_y(x: np.ndarray, y_anchor: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return y_anchor + x @ theta


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
    orth_flag = bool(baseline_est.get("orthogonalize_country", False))

    fit_ds = _prepare_dataset(_split_rows(rows_all, args.fit_split), feature_cols, orthogonalize_country=orth_flag)
    sel_ds = _prepare_dataset(_split_rows(rows_all, args.selection_split), feature_cols, orthogonalize_country=orth_flag)
    eval_ds = _prepare_dataset(_split_rows(rows_all, args.eval_split), feature_cols, orthogonalize_country=orth_flag)

    std_fit = _build_standardized_train(fit_ds["x_no_bias"])
    x_fit = std_fit["x"]
    mu = std_fit["mu"]
    sigma = std_fit["sigma"]
    x_sel = _apply_standardize(sel_ds["x_no_bias"], mu, sigma)
    x_eval = _apply_standardize(eval_ds["x_no_bias"], mu, sigma)

    phase_lambdas = _parse_float_list(args.phase_lambda_grid)
    turn_lambdas = _parse_float_list(args.turn_lambda_grid)
    ridge_grid = _parse_float_list(args.ridge_grid)

    candidates: list[dict[str, object]] = []
    for lp in phase_lambdas:
        for lt in turn_lambdas:
            for ridge in ridge_grid:
                fit_obj = _train_phase_aware(
                    x_fit,
                    fit_ds["y_anchor"],
                    fit_ds["y_true"],
                    fit_ds["delta_true"],
                    lambda_phase=float(lp),
                    lambda_turn=float(lt),
                    ridge_alpha=float(ridge),
                    turn_k=float(args.turn_k),
                    lr=float(args.lr),
                    iters=int(args.iters),
                )
                theta = np.asarray(fit_obj["theta"], dtype=float)
                y_sel_hat = _predict_y(x_sel, sel_ds["y_anchor"], theta)
                sel_m = _metrics(sel_ds["y_true"], y_sel_hat, sel_ds["y_anchor"])
                sel_lag = _best_lag(_lag_scan(sel_ds["y_true"], y_sel_hat, sel_ds["y_anchor"], max_lag=args.max_lag))
                sel_turn = _turn_metrics(sel_ds["y_true"], y_sel_hat, sel_ds["y_anchor"], eps=args.turn_eps, tol=args.turn_tol)

                y_eval_hat = _predict_y(x_eval, eval_ds["y_anchor"], theta)
                eval_m = _metrics(eval_ds["y_true"], y_eval_hat, eval_ds["y_anchor"])
                eval_lag = _best_lag(_lag_scan(eval_ds["y_true"], y_eval_hat, eval_ds["y_anchor"], max_lag=args.max_lag))
                eval_turn = _turn_metrics(eval_ds["y_true"], y_eval_hat, eval_ds["y_anchor"], eps=args.turn_eps, tol=args.turn_tol)

                rec = {
                    "lambda_phase": float(lp),
                    "lambda_turn": float(lt),
                    "ridge": float(ridge),
                    "fit": {
                        "x_r2": float(fit_obj["fit_x_r2"]),
                        "x_rmse": float(fit_obj["fit_x_rmse"]),
                    },
                    "selection": {
                        "metrics": sel_m,
                        "best_lag": int(sel_lag["lag"]),
                        "best_lag_corr": float(sel_lag["corr"]),
                        "turn": sel_turn,
                    },
                    "eval": {
                        "metrics": eval_m,
                        "best_lag": int(eval_lag["lag"]),
                        "best_lag_corr": float(eval_lag["corr"]),
                        "turn": eval_turn,
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

    def _delay_abs(v: float) -> float:
        return float(abs(v)) if np.isfinite(v) else 1e9

    best_by_selection = sorted(
        candidates,
        key=lambda r: (
            abs(int(r["selection"]["best_lag"])),
            _delay_abs(float(r["selection"]["turn"]["mean_delay"])),
            float(r["selection"]["metrics"]["x_rmse"]),
            -float(r["selection"]["metrics"]["x_corr"]),
        ),
    )[0]
    best_by_eval = sorted(
        candidates,
        key=lambda r: (
            abs(int(r["eval"]["best_lag"])),
            _delay_abs(float(r["eval"]["turn"]["mean_delay"])),
            float(r["eval"]["metrics"]["x_rmse"]),
            -float(r["eval"]["metrics"]["x_corr"]),
        ),
    )[0]

    chosen = best_by_selection
    fit_obj = _train_phase_aware(
        x_fit,
        fit_ds["y_anchor"],
        fit_ds["y_true"],
        fit_ds["delta_true"],
        lambda_phase=float(chosen["lambda_phase"]),
        lambda_turn=float(chosen["lambda_turn"]),
        ridge_alpha=float(chosen["ridge"]),
        turn_k=float(args.turn_k),
        lr=float(args.lr),
        iters=int(args.iters),
    )
    theta = np.asarray(fit_obj["theta"], dtype=float)
    y_eval_hat = _predict_y(x_eval, eval_ds["y_anchor"], theta)

    summary = {
        "config": {
            "panel": str(args.panel),
            "fit_split": args.fit_split,
            "selection_split": args.selection_split,
            "eval_split": args.eval_split,
            "phase_lambda_grid": phase_lambdas,
            "turn_lambda_grid": turn_lambdas,
            "ridge_grid": ridge_grid,
            "turn_k": float(args.turn_k),
            "lr": float(args.lr),
            "iters": int(args.iters),
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
    (out_dir / "phase_aware_loss_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with (out_dir / "phase_aware_loss_scan.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lambda_phase",
                "lambda_turn",
                "ridge",
                "fit_x_r2",
                "fit_x_rmse",
                "sel_x_r2",
                "sel_x_rmse",
                "sel_x_corr",
                "sel_best_lag",
                "sel_turn_recall",
                "sel_turn_precision",
                "sel_turn_mean_delay",
                "eval_x_r2",
                "eval_x_rmse",
                "eval_x_corr",
                "eval_best_lag",
                "eval_turn_recall",
                "eval_turn_precision",
                "eval_turn_mean_delay",
                "gap_eval_x_r2_vs_main",
                "gap_eval_x_rmse_vs_main",
                "gap_eval_best_lag_vs_main",
            ],
        )
        writer.writeheader()
        for r in candidates:
            writer.writerow(
                {
                    "lambda_phase": r["lambda_phase"],
                    "lambda_turn": r["lambda_turn"],
                    "ridge": r["ridge"],
                    "fit_x_r2": r["fit"]["x_r2"],
                    "fit_x_rmse": r["fit"]["x_rmse"],
                    "sel_x_r2": r["selection"]["metrics"]["x_r2"],
                    "sel_x_rmse": r["selection"]["metrics"]["x_rmse"],
                    "sel_x_corr": r["selection"]["metrics"]["x_corr"],
                    "sel_best_lag": r["selection"]["best_lag"],
                    "sel_turn_recall": r["selection"]["turn"]["turn_hit_rate_recall"],
                    "sel_turn_precision": r["selection"]["turn"]["turn_precision"],
                    "sel_turn_mean_delay": r["selection"]["turn"]["mean_delay"],
                    "eval_x_r2": r["eval"]["metrics"]["x_r2"],
                    "eval_x_rmse": r["eval"]["metrics"]["x_rmse"],
                    "eval_x_corr": r["eval"]["metrics"]["x_corr"],
                    "eval_best_lag": r["eval"]["best_lag"],
                    "eval_turn_recall": r["eval"]["turn"]["turn_hit_rate_recall"],
                    "eval_turn_precision": r["eval"]["turn"]["turn_precision"],
                    "eval_turn_mean_delay": r["eval"]["turn"]["mean_delay"],
                    "gap_eval_x_r2_vs_main": r["gap_eval_vs_main"]["x_r2"],
                    "gap_eval_x_rmse_vs_main": r["gap_eval_vs_main"]["x_rmse"],
                    "gap_eval_best_lag_vs_main": r["gap_eval_vs_main"]["best_lag"],
                }
            )

    with (out_dir / "test_predictions_best_by_selection.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["date_t1", "y_t1_actual", "y_t1_pred_phase_loss", "y_t", "lambda_phase", "lambda_turn", "ridge"],
        )
        writer.writeheader()
        for i in range(len(eval_ds["y_true"])):
            writer.writerow(
                {
                    "date_t1": eval_ds["date_t1"][i].strftime("%Y-%m-%d"),
                    "y_t1_actual": float(eval_ds["y_true"][i]),
                    "y_t1_pred_phase_loss": float(y_eval_hat[i]),
                    "y_t": float(eval_ds["y_anchor"][i]),
                    "lambda_phase": float(chosen["lambda_phase"]),
                    "lambda_turn": float(chosen["lambda_turn"]),
                    "ridge": float(chosen["ridge"]),
                }
            )

    print("[done] step22 phase-aware-training-loss scan complete")
    print(
        "[done] best_by_selection: "
        f"lambda_phase={best_by_selection['lambda_phase']:.6g}, lambda_turn={best_by_selection['lambda_turn']:.6g}, "
        f"ridge={best_by_selection['ridge']:.6g}, sel_best_lag={int(best_by_selection['selection']['best_lag'])}, "
        f"eval_best_lag={int(best_by_selection['eval']['best_lag'])}, eval_x_r2={float(best_by_selection['eval']['metrics']['x_r2']):.6g}"
    )
    print(
        "[done] best_by_eval: "
        f"lambda_phase={best_by_eval['lambda_phase']:.6g}, lambda_turn={best_by_eval['lambda_turn']:.6g}, "
        f"ridge={best_by_eval['ridge']:.6g}, eval_best_lag={int(best_by_eval['eval']['best_lag'])}, "
        f"eval_x_r2={float(best_by_eval['eval']['metrics']['x_r2']):.6g}, eval_x_rmse={float(best_by_eval['eval']['metrics']['x_rmse']):.6g}"
    )
    print(f"[done] output: {out_dir}")


if __name__ == "__main__":
    main()

