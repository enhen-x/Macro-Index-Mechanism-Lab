"""Step 14 experiment: explicit delay parameters for feature groups.

Model keeps the same dynamic form but uses delayed features:
    u_j(t - d_group(j))
where delays are searched on validation split and evaluated on test split.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from identification.ols_identifier import identify_ols_y_next_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan explicit feature delays by groups.")
    parser.add_argument("--panel", default="data/us/us_regression_panel.csv")
    parser.add_argument("--baseline-est", default="data/us/us_ols_estimation.json")
    parser.add_argument("--baseline-metrics", default="data/us/plot/fit_metrics.json")
    parser.add_argument("--baseline-phase", default="data/us/experiments/step5_phase_bias_baseline/phase_bias_summary.json")
    parser.add_argument("--output-dir", default="data/us/experiments/step14_explicit_delay_params")
    parser.add_argument("--fit-split", choices=["train", "valid", "test", "all"], default="train")
    parser.add_argument("--selection-split", choices=["train", "valid", "test", "all"], default="valid")
    parser.add_argument("--eval-split", choices=["train", "valid", "test", "all"], default="test")
    parser.add_argument("--delay-values", default="0,1,2", help="Comma-separated integer delays to scan, e.g. 0,1,2")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--turn-eps", type=float, default=1e-6)
    parser.add_argument("--turn-tol", type=int, default=1)
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


def _extract(rows: list[dict[str, str]], features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.array([_to_float(r["Y"]) for r in rows], dtype=float)
    a = np.array([_to_float(r["a"]) for r in rows], dtype=float)
    u = np.array([[_to_float(r[c]) for c in features] for r in rows], dtype=float)
    finite = np.isfinite(y) & np.isfinite(a) & np.all(np.isfinite(u), axis=1)
    return y[finite], a[finite], u[finite]


def _feature_group(name: str) -> str:
    if name.startswith("g_"):
        return "market"
    if name.startswith("c_policy_rate"):
        return "policy"
    if name.startswith("c_"):
        return "macro"
    return "other"


def _shift_features_by_group(u: np.ndarray, feature_names: list[str], delays: dict[str, int]) -> np.ndarray:
    out = np.full_like(u, np.nan, dtype=float)
    n = len(u)
    for j, name in enumerate(feature_names):
        grp = _feature_group(name)
        d = int(delays.get(grp, 0))
        if d <= 0:
            out[:, j] = u[:, j]
        elif d < n:
            out[d:, j] = u[:-d, j]
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


def _fit_and_predict(
    y_fit: np.ndarray,
    a_fit: np.ndarray,
    u_fit_shifted: np.ndarray,
    y_eval: np.ndarray,
    u_eval_shifted: np.ndarray,
    feature_cols: list[str],
    baseline_est: dict,
    dt: float,
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    idx_fit = np.arange(1, len(y_fit) - 1)
    est_obj = identify_ols_y_next_panel(
        y_prev=y_fit[idx_fit - 1],
        y_cur=y_fit[idx_fit],
        y_next=y_fit[idx_fit + 1],
        u_cur=u_fit_shifted[idx_fit],
        feature_names=feature_cols,
        a_for_weight=a_fit[idx_fit],
        m=float(baseline_est.get("m", 1.0)),
        dt=dt,
        fit_intercept=True,
        ridge_alpha=float(baseline_est.get("ridge_alpha", 0.05)),
        enforce_physical=bool(baseline_est.get("enforce_physical", True)),
        c_min=float(baseline_est.get("c_min", 1e-4)),
        k_min=float(baseline_est.get("k_min", 0.0)),
        damping_mode=str(baseline_est.get("damping_mode", "nonlinear_absv")),
        c_nl_min=float(baseline_est.get("c_nl_min", 0.0)),
        tail_weight_mode=str(baseline_est.get("tail_weight_mode", "none")),
        tail_weight_q=float(baseline_est.get("tail_weight_q", 0.9)),
        tail_weight_scale=float(baseline_est.get("tail_weight_scale", 3.0)),
        tail_weight_power=float(baseline_est.get("tail_weight_power", 1.0)),
        robust_mode=str(baseline_est.get("robust_mode", "none")),
        robust_tuning=float(baseline_est.get("robust_tuning", 1.345)),
        robust_max_iter=int(baseline_est.get("robust_max_iter", 20)),
        robust_tol=float(baseline_est.get("robust_tol", 1e-6)),
    )

    idx_eval = np.arange(1, len(y_eval) - 1)
    y_prev = y_eval[idx_eval - 1]
    y_cur = y_eval[idx_eval]
    y_true = y_eval[idx_eval + 1]
    m = float(est_obj.m)
    c = float(est_obj.c)
    c_nl = float(est_obj.c_nl)
    k = float(est_obj.k)
    intercept = float(est_obj.intercept)
    beta_vec = np.array([float(est_obj.betas[name]) / m for name in feature_cols], dtype=float)
    v = (y_cur - y_prev) / dt
    dt2 = dt * dt
    y_hat = (
        2.0 * y_cur
        - y_prev
        + dt2
        * (
            intercept
            - (c / m) * v
            - (c_nl / m) * np.abs(v) * v
            - (k / m) * y_cur
            + (u_eval_shifted[idx_eval] @ beta_vec)
        )
    )
    finite = np.isfinite(y_true) & np.isfinite(y_hat) & np.isfinite(y_cur)
    return est_obj, y_true[finite], y_hat[finite], y_cur[finite]


def _parse_delay_values(raw: str) -> list[int]:
    vals = []
    for p in raw.split(","):
        t = p.strip()
        if not t:
            continue
        v = int(t)
        if v < 0:
            raise ValueError("delay values must be >= 0")
        vals.append(v)
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("delay-values cannot be empty")
    return vals


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

    y_fit, a_fit, u_fit = _extract(_split_rows(rows_all, args.fit_split), feature_cols)
    y_sel, _, u_sel = _extract(_split_rows(rows_all, args.selection_split), feature_cols)
    y_eval, _, u_eval = _extract(_split_rows(rows_all, args.eval_split), feature_cols)

    # Orthogonalize on original aligned features first; then apply delay shifts.
    if bool(baseline_est.get("orthogonalize_country", False)):
        u_fit = _orthogonalize_country_features(u_fit, feature_cols)
        u_sel = _orthogonalize_country_features(u_sel, feature_cols)
        u_eval = _orthogonalize_country_features(u_eval, feature_cols)

    delay_values = _parse_delay_values(args.delay_values)
    candidates: list[dict[str, object]] = []

    for d_market, d_policy, d_macro in itertools.product(delay_values, delay_values, delay_values):
        delays = {"market": d_market, "policy": d_policy, "macro": d_macro, "other": 0}
        try:
            u_fit_shift = _shift_features_by_group(u_fit, feature_cols, delays)
            u_sel_shift = _shift_features_by_group(u_sel, feature_cols, delays)
            u_eval_shift = _shift_features_by_group(u_eval, feature_cols, delays)

            est_obj, ys_true, ys_hat, ys_anchor = _fit_and_predict(
                y_fit=y_fit,
                a_fit=a_fit,
                u_fit_shifted=u_fit_shift,
                y_eval=y_sel,
                u_eval_shifted=u_sel_shift,
                feature_cols=feature_cols,
                baseline_est=baseline_est,
                dt=args.dt,
            )
            _, ye_true, ye_hat, ye_anchor = _fit_and_predict(
                y_fit=y_fit,
                a_fit=a_fit,
                u_fit_shifted=u_fit_shift,
                y_eval=y_eval,
                u_eval_shifted=u_eval_shift,
                feature_cols=feature_cols,
                baseline_est=baseline_est,
                dt=args.dt,
            )

            m_sel = _metrics(ys_true, ys_hat, ys_anchor)
            m_eval = _metrics(ye_true, ye_hat, ye_anchor)
            lag_sel = _best_lag(_lag_scan(ys_true, ys_hat, ys_anchor, max_lag=args.max_lag))
            lag_eval = _best_lag(_lag_scan(ye_true, ye_hat, ye_anchor, max_lag=args.max_lag))
            turn_eval = _turn_metrics(ye_true, ye_hat, ye_anchor, eps=args.turn_eps, tol=args.turn_tol)

            rec = {
                "delays": delays,
                "status": "ok",
                "train_target_r2": float(est_obj.r2),
                "condition_number": float(est_obj.condition_number),
                "selection": {"metrics": m_sel, "best_lag": int(lag_sel["lag"]), "best_lag_corr": float(lag_sel["corr"])},
                "eval": {"metrics": m_eval, "best_lag": int(lag_eval["lag"]), "best_lag_corr": float(lag_eval["corr"]), "turn": turn_eval},
            }
            rec["gap_eval_vs_main"] = {
                "x_r2": float(m_eval["x_r2"] - float(baseline_metrics["x_r2"])),
                "x_rmse": float(m_eval["x_rmse"] - float(baseline_metrics["x_rmse"])),
                "best_lag": int(lag_eval["lag"] - int(baseline_phase["best_lag_by_corr"]["lag"])),
                "turn_recall": float(turn_eval["turn_hit_rate_recall"] - float(baseline_phase["turn_metrics_lag0"]["turn_hit_rate_recall"])),
                "turn_precision": float(turn_eval["turn_precision"] - float(baseline_phase["turn_metrics_lag0"]["turn_precision"])),
                "turn_mean_delay": float(turn_eval["mean_delay"] - float(baseline_phase["turn_metrics_lag0"]["mean_delay"])),
            }
        except Exception as e:
            rec = {"delays": delays, "status": f"failed: {e}"}
        candidates.append(rec)

    ok = [r for r in candidates if r.get("status") == "ok"]
    if not ok:
        raise RuntimeError("all delay candidates failed")

    best_selection = sorted(
        ok,
        key=lambda r: (
            abs(int(r["selection"]["best_lag"])),
            float(r["selection"]["metrics"]["x_rmse"]),
            -float(r["selection"]["metrics"]["x_corr"]),
        ),
    )[0]
    best_eval = sorted(
        ok,
        key=lambda r: (
            abs(int(r["eval"]["best_lag"])),
            float(r["eval"]["metrics"]["x_rmse"]),
            -float(r["eval"]["metrics"]["x_corr"]),
        ),
    )[0]

    summary = {
        "config": {
            "panel": str(args.panel),
            "fit_split": args.fit_split,
            "selection_split": args.selection_split,
            "eval_split": args.eval_split,
            "delay_values": delay_values,
            "groups": ["market", "policy", "macro"],
        },
        "baseline": {
            "x_r2": float(baseline_metrics["x_r2"]),
            "x_rmse": float(baseline_metrics["x_rmse"]),
            "best_lag": int(baseline_phase["best_lag_by_corr"]["lag"]),
            "turn_recall": float(baseline_phase["turn_metrics_lag0"]["turn_hit_rate_recall"]),
            "turn_precision": float(baseline_phase["turn_metrics_lag0"]["turn_precision"]),
            "turn_mean_delay": float(baseline_phase["turn_metrics_lag0"]["mean_delay"]),
        },
        "best_by_selection": best_selection,
        "best_by_eval": best_eval,
        "candidates": candidates,
    }
    (out_dir / "explicit_delay_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with (out_dir / "explicit_delay_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "d_market",
                "d_policy",
                "d_macro",
                "status",
                "train_target_r2",
                "condition_number",
                "sel_x_r2",
                "sel_x_rmse",
                "sel_x_corr",
                "sel_best_lag",
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
            d = r["delays"]
            if r.get("status") != "ok":
                writer.writerow(
                    {
                        "d_market": d["market"],
                        "d_policy": d["policy"],
                        "d_macro": d["macro"],
                        "status": r.get("status"),
                    }
                )
                continue
            writer.writerow(
                {
                    "d_market": d["market"],
                    "d_policy": d["policy"],
                    "d_macro": d["macro"],
                    "status": r["status"],
                    "train_target_r2": r["train_target_r2"],
                    "condition_number": r["condition_number"],
                    "sel_x_r2": r["selection"]["metrics"]["x_r2"],
                    "sel_x_rmse": r["selection"]["metrics"]["x_rmse"],
                    "sel_x_corr": r["selection"]["metrics"]["x_corr"],
                    "sel_best_lag": r["selection"]["best_lag"],
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

    print("[done] step14 explicit-delay scan complete")
    print(f"[done] candidates={len(candidates)}, ok={len(ok)}")
    print(
        "[done] best_by_selection: "
        f"d={best_selection['delays']}, sel_best_lag={best_selection['selection']['best_lag']}, "
        f"eval_best_lag={best_selection['eval']['best_lag']}, eval_x_r2={best_selection['eval']['metrics']['x_r2']:.6g}"
    )
    print(
        "[done] best_by_eval: "
        f"d={best_eval['delays']}, eval_best_lag={best_eval['eval']['best_lag']}, "
        f"eval_x_r2={best_eval['eval']['metrics']['x_r2']:.6g}, eval_x_rmse={best_eval['eval']['metrics']['x_rmse']:.6g}"
    )
    print(f"[done] output: {out_dir}")


if __name__ == "__main__":
    main()
