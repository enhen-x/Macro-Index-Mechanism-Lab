"""Step 26 experiment: explicit fractional shift layer decoupled from amplitude model.

Given amplitude prediction series x_t (one-step raw y_hat), apply causal fractional
advance layer:
    y_shift_t = x_t + tau * (x_t - x_{t-1})

tau is explicit phase parameter. This separates amplitude modeling from phase shift.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan explicit fractional shift layer.")
    parser.add_argument("--panel", default="data/us/us_regression_panel.csv")
    parser.add_argument("--estimation", default="data/us/us_ols_estimation.json")
    parser.add_argument("--baseline-metrics", default="data/us/plot/fit_metrics.json")
    parser.add_argument("--baseline-phase", default="data/us/experiments/step5_phase_bias_baseline/phase_bias_summary.json")
    parser.add_argument("--output-dir", default="data/us/experiments/step26_fractional_shift_layer")
    parser.add_argument("--selection-split", choices=["train", "valid", "test", "all"], default="valid")
    parser.add_argument("--eval-split", choices=["train", "valid", "test", "all"], default="test")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--turn-eps", type=float, default=1e-6)
    parser.add_argument("--turn-tol", type=int, default=1)
    parser.add_argument("--tau-grid", default="-0.5:2.0:0.05")
    parser.add_argument("--step-clip-std-grid", default="0,2,3,4", help="0 means no clipping.")
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


def _apply_fractional_shift(y_raw: np.ndarray, tau: float, step_clip_std: float) -> np.ndarray:
    if len(y_raw) == 0:
        return y_raw.copy()
    d = np.zeros(len(y_raw), dtype=float)
    if len(y_raw) > 1:
        d[1:] = y_raw[1:] - y_raw[:-1]
    if step_clip_std > 0:
        s = float(np.std(d))
        lim = float(step_clip_std) * s
        if lim > 0:
            d = np.clip(d, -lim, lim)
    return y_raw + float(tau) * d


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
        cands = [p for p in pred_list if p not in used_pred and abs(p - t) <= tol]
        if not cands:
            continue
        p_best = sorted(cands, key=lambda p: (abs(p - t), p))[0]
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

    baseline_est = json.loads(Path(args.estimation).read_text(encoding="utf-8"))
    baseline_metrics = json.loads(Path(args.baseline_metrics).read_text(encoding="utf-8"))
    baseline_phase = json.loads(Path(args.baseline_phase).read_text(encoding="utf-8"))
    rows_all = _load_rows(Path(args.panel))

    feature_cols = list(baseline_est.get("betas", {}).keys())
    if not feature_cols:
        feature_cols = [c for c in rows_all[0].keys() if c.startswith("g_") or c.startswith("c_") or c.startswith("i_")]

    sel = _one_step_raw(_split_rows(rows_all, args.selection_split), feature_cols, baseline_est, dt=args.dt)
    eva = _one_step_raw(_split_rows(rows_all, args.eval_split), feature_cols, baseline_est, dt=args.dt)

    tau_grid = _parse_grid(args.tau_grid)
    step_clip_grid = _parse_float_list(args.step_clip_std_grid)

    candidates: list[dict[str, object]] = []
    for clip_std in step_clip_grid:
        for tau in tau_grid:
            y_sel = _apply_fractional_shift(sel["y_hat_raw"], tau=float(tau), step_clip_std=float(clip_std))
            sel_m = _metrics(sel["y_true"], y_sel, sel["y_anchor"])
            sel_lag = _best_lag(_lag_scan(sel["y_true"], y_sel, sel["y_anchor"], max_lag=args.max_lag))
            sel_turn = _turn_metrics(sel["y_true"], y_sel, sel["y_anchor"], eps=args.turn_eps, tol=args.turn_tol)

            y_eval = _apply_fractional_shift(eva["y_hat_raw"], tau=float(tau), step_clip_std=float(clip_std))
            eval_m = _metrics(eva["y_true"], y_eval, eva["y_anchor"])
            eval_lag = _best_lag(_lag_scan(eva["y_true"], y_eval, eva["y_anchor"], max_lag=args.max_lag))
            eval_turn = _turn_metrics(eva["y_true"], y_eval, eva["y_anchor"], eps=args.turn_eps, tol=args.turn_tol)

            rec = {
                "tau": float(tau),
                "step_clip_std": float(clip_std),
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

    y_eval_best = _apply_fractional_shift(
        eva["y_hat_raw"],
        tau=float(best_by_selection["tau"]),
        step_clip_std=float(best_by_selection["step_clip_std"]),
    )

    summary = {
        "config": {
            "panel": str(args.panel),
            "selection_split": args.selection_split,
            "eval_split": args.eval_split,
            "tau_grid": [float(x) for x in tau_grid.tolist()],
            "step_clip_std_grid": step_clip_grid,
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
    (out_dir / "fractional_shift_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with (out_dir / "fractional_shift_scan.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tau",
                "step_clip_std",
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
                    "tau": r["tau"],
                    "step_clip_std": r["step_clip_std"],
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
            fieldnames=["date_t1", "y_t1_actual", "y_t1_pred_raw", "y_t1_pred_shift", "y_t", "tau", "step_clip_std"],
        )
        writer.writeheader()
        for i in range(len(eva["y_true"])):
            writer.writerow(
                {
                    "date_t1": eva["date_t1"][i].strftime("%Y-%m-%d"),
                    "y_t1_actual": float(eva["y_true"][i]),
                    "y_t1_pred_raw": float(eva["y_hat_raw"][i]),
                    "y_t1_pred_shift": float(y_eval_best[i]),
                    "y_t": float(eva["y_anchor"][i]),
                    "tau": float(best_by_selection["tau"]),
                    "step_clip_std": float(best_by_selection["step_clip_std"]),
                }
            )

    print("[done] step26 fractional-shift-layer scan complete")
    print(
        "[done] best_by_selection: "
        f"tau={best_by_selection['tau']:.6g}, clip_std={best_by_selection['step_clip_std']:.6g}, "
        f"sel_best_lag={int(best_by_selection['selection']['best_lag'])}, eval_best_lag={int(best_by_selection['eval']['best_lag'])}, "
        f"eval_x_r2={float(best_by_selection['eval']['metrics']['x_r2']):.6g}"
    )
    print(
        "[done] best_by_eval: "
        f"tau={best_by_eval['tau']:.6g}, clip_std={best_by_eval['step_clip_std']:.6g}, "
        f"eval_best_lag={int(best_by_eval['eval']['best_lag'])}, eval_x_r2={float(best_by_eval['eval']['metrics']['x_r2']):.6g}, "
        f"eval_x_rmse={float(best_by_eval['eval']['metrics']['x_rmse']):.6g}"
    )
    print(f"[done] output: {out_dir}")


if __name__ == "__main__":
    main()

