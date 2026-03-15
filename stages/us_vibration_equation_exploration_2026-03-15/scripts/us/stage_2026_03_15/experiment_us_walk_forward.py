"""Step 3 experiment: walk-forward re-estimation vs static main model (nonlinear_absv)."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from identification.ols_identifier import identify_ols_y_next_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward validation for nonlinear main model.")
    parser.add_argument("--panel", default="data/us/us_regression_panel.csv")
    parser.add_argument("--baseline-est", default="data/us/us_ols_estimation.json")
    parser.add_argument("--baseline-metrics", default="data/us/plot/fit_metrics.json")
    parser.add_argument("--output-dir", default="data/us/experiments/step3_walk_forward")
    parser.add_argument("--min-train", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--step", type=int, default=6)
    parser.add_argument("--dt", type=float, default=1.0)
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


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty panel: {path}")
    return rows


def _extract(rows: list[dict[str, str]], features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dates = np.array([datetime.strptime(r["date"], "%Y-%m-%d") for r in rows])
    a = np.array([_to_float(r["a"]) for r in rows], dtype=float)
    v = np.array([_to_float(r["v"]) for r in rows], dtype=float)
    y = np.array([_to_float(r["Y"]) for r in rows], dtype=float)
    u = np.array([[_to_float(r[c]) for c in features] for r in rows], dtype=float)
    return dates, a, v, y, u


def _predict_one_step(
    y_prev: float,
    y_cur: float,
    u_cur: np.ndarray,
    *,
    c: float,
    c_nl: float,
    k: float,
    m: float,
    intercept: float,
    beta_vec: np.ndarray,
    dt: float,
) -> float:
    v = (y_cur - y_prev) / dt
    dt2 = dt * dt
    return float(
        2.0 * y_cur
        - y_prev
        + dt2
        * (intercept - (c / m) * v - (c_nl / m) * np.abs(v) * v - (k / m) * y_cur + float(u_cur @ beta_vec))
    )


def _fold_metrics(y_true: np.ndarray, y_hat: np.ndarray, y_anchor: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((y_true - y_hat) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_hat)))
    r2 = _r2(y_true, y_hat)
    corr = float(np.corrcoef(y_true, y_hat)[0, 1]) if len(y_true) > 1 else float("nan")
    direction = float(np.mean(np.sign(y_true - y_anchor) == np.sign(y_hat - y_anchor))) if len(y_true) else float("nan")
    return {
        "n_obs": int(len(y_true)),
        "x_r2": r2,
        "x_rmse": rmse,
        "x_mae": mae,
        "x_corr": corr,
        "x_direction": direction,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(Path(args.panel))
    baseline_est = json.loads(Path(args.baseline_est).read_text(encoding="utf-8"))
    baseline_metrics = json.loads(Path(args.baseline_metrics).read_text(encoding="utf-8"))

    feature_names = list(baseline_est.get("betas", {}).keys())
    if not feature_names:
        feature_names = [c for c in rows[0].keys() if c.startswith("g_") or c.startswith("c_") or c.startswith("i_")]

    dates, a_all, _, y_all, u_all = _extract(rows, feature_names)

    orth = bool(baseline_est.get("orthogonalize_country", True))
    n = len(rows)

    m = float(baseline_est.get("m", 1.0))
    ridge_alpha = float(baseline_est.get("ridge_alpha", 0.05))
    enforce = bool(baseline_est.get("enforce_physical", True))
    c_min = float(baseline_est.get("c_min", 1e-4))
    c_nl_min = float(baseline_est.get("c_nl_min", 0.0))
    k_min = float(baseline_est.get("k_min", 0.0))
    tail_mode = str(baseline_est.get("tail_weight_mode", "none"))
    tail_q = float(baseline_est.get("tail_weight_q", 0.9))
    tail_scale = float(baseline_est.get("tail_weight_scale", 3.0))
    tail_power = float(baseline_est.get("tail_weight_power", 1.0))
    robust_mode = str(baseline_est.get("robust_mode", "none"))
    robust_tuning = float(baseline_est.get("robust_tuning", 1.345))

    beta_static = np.array([float(baseline_est["betas"][c]) / m for c in feature_names], dtype=float)
    c_static = float(baseline_est["c"])
    c_nl_static = float(baseline_est.get("c_nl", 0.0))
    k_static = float(baseline_est["k"])
    intercept_static = float(baseline_est.get("intercept", 0.0))

    folds: list[dict[str, object]] = []

    start = max(args.min_train, 3)
    while start + 2 < n:
        end = min(start + args.horizon, n - 1)
        t_idx = np.arange(start, end, dtype=int)
        t_idx = t_idx[(t_idx >= 1) & (t_idx + 1 < n)]
        if len(t_idx) < 3:
            start += args.step
            continue

        # train slice: [0, start)
        y_tr = y_all[:start].copy()
        u_tr = u_all[:start].copy()
        a_tr = a_all[:start].copy()
        if orth:
            u_tr = _orthogonalize_country_features(u_tr, feature_names)

        idx_tr = np.arange(1, len(y_tr) - 1)
        try:
            est_dyn = identify_ols_y_next_panel(
                y_prev=y_tr[idx_tr - 1],
                y_cur=y_tr[idx_tr],
                y_next=y_tr[idx_tr + 1],
                u_cur=u_tr[idx_tr],
                feature_names=feature_names,
                a_for_weight=a_tr[idx_tr],
                m=m,
                dt=args.dt,
                fit_intercept=True,
                ridge_alpha=ridge_alpha,
                enforce_physical=enforce,
                c_min=c_min,
                k_min=k_min,
                damping_mode="nonlinear_absv",
                c_nl_min=c_nl_min,
                tail_weight_mode=tail_mode,
                tail_weight_q=tail_q,
                tail_weight_scale=tail_scale,
                tail_weight_power=tail_power,
                robust_mode=robust_mode,
                robust_tuning=robust_tuning,
            )
        except Exception:
            start += args.step
            continue

        beta_dyn = np.array([float(est_dyn.betas[c]) / est_dyn.m for c in feature_names], dtype=float)

        # eval block transform
        u_block = u_all[t_idx].copy()
        if orth:
            u_block = _orthogonalize_country_features(u_block, feature_names)

        y_true = []
        y_hat_dyn = []
        y_hat_static = []
        y_anchor = []

        for j, t in enumerate(t_idx):
            y_prev = float(y_all[t - 1])
            y_cur = float(y_all[t])
            y_nxt = float(y_all[t + 1])
            u_t = u_block[j]
            if not (np.isfinite(y_prev) and np.isfinite(y_cur) and np.isfinite(y_nxt) and np.all(np.isfinite(u_t))):
                continue

            pred_dyn = _predict_one_step(
                y_prev,
                y_cur,
                u_t,
                c=float(est_dyn.c),
                c_nl=float(est_dyn.c_nl),
                k=float(est_dyn.k),
                m=float(est_dyn.m),
                intercept=float(est_dyn.intercept),
                beta_vec=beta_dyn,
                dt=args.dt,
            )
            pred_static = _predict_one_step(
                y_prev,
                y_cur,
                u_t,
                c=c_static,
                c_nl=c_nl_static,
                k=k_static,
                m=m,
                intercept=intercept_static,
                beta_vec=beta_static,
                dt=args.dt,
            )
            y_true.append(y_nxt)
            y_hat_dyn.append(pred_dyn)
            y_hat_static.append(pred_static)
            y_anchor.append(y_cur)

        if len(y_true) < 3:
            start += args.step
            continue

        y_true_arr = np.array(y_true, dtype=float)
        y_dyn_arr = np.array(y_hat_dyn, dtype=float)
        y_sta_arr = np.array(y_hat_static, dtype=float)
        y_anchor_arr = np.array(y_anchor, dtype=float)

        m_dyn = _fold_metrics(y_true_arr, y_dyn_arr, y_anchor_arr)
        m_sta = _fold_metrics(y_true_arr, y_sta_arr, y_anchor_arr)

        folds.append(
            {
                "fold_start_idx": int(start),
                "fold_end_idx": int(end - 1),
                "date_start": dates[start].strftime("%Y-%m-%d"),
                "date_end": dates[end - 1].strftime("%Y-%m-%d"),
                "n_eval": int(len(y_true_arr)),
                "dynamic": m_dyn,
                "static": m_sta,
                "gap_dynamic_minus_static": {
                    "x_r2": float(m_dyn["x_r2"] - m_sta["x_r2"]),
                    "x_rmse": float(m_dyn["x_rmse"] - m_sta["x_rmse"]),
                    "x_direction": float(m_dyn["x_direction"] - m_sta["x_direction"]),
                },
                "dynamic_condition_number": float(est_dyn.condition_number),
                "dynamic_train_target_r2": float(est_dyn.r2),
            }
        )

        start += args.step

    if not folds:
        raise RuntimeError("no folds generated")

    dyn_x_r2 = np.array([f["dynamic"]["x_r2"] for f in folds], dtype=float)
    dyn_x_rmse = np.array([f["dynamic"]["x_rmse"] for f in folds], dtype=float)
    dyn_x_dir = np.array([f["dynamic"]["x_direction"] for f in folds], dtype=float)

    sta_x_r2 = np.array([f["static"]["x_r2"] for f in folds], dtype=float)
    sta_x_rmse = np.array([f["static"]["x_rmse"] for f in folds], dtype=float)
    sta_x_dir = np.array([f["static"]["x_direction"] for f in folds], dtype=float)

    summary = {
        "baseline": {
            "estimation_path": str(args.baseline_est),
            "metrics_path": str(args.baseline_metrics),
            "overall_test_metrics": baseline_metrics,
        },
        "walk_forward_config": {
            "min_train": args.min_train,
            "horizon": args.horizon,
            "step": args.step,
            "n_folds": len(folds),
            "damping_mode": "nonlinear_absv",
            "ridge_alpha": ridge_alpha,
        },
        "aggregate": {
            "dynamic_mean_x_r2": float(np.mean(dyn_x_r2)),
            "dynamic_mean_x_rmse": float(np.mean(dyn_x_rmse)),
            "dynamic_mean_x_direction": float(np.mean(dyn_x_dir)),
            "dynamic_min_x_r2": float(np.min(dyn_x_r2)),
            "dynamic_max_x_r2": float(np.max(dyn_x_r2)),
            "static_mean_x_r2": float(np.mean(sta_x_r2)),
            "static_mean_x_rmse": float(np.mean(sta_x_rmse)),
            "static_mean_x_direction": float(np.mean(sta_x_dir)),
            "gap_dynamic_minus_static_x_r2": float(np.mean(dyn_x_r2 - sta_x_r2)),
            "gap_dynamic_minus_static_x_rmse": float(np.mean(dyn_x_rmse - sta_x_rmse)),
            "gap_dynamic_minus_static_x_direction": float(np.mean(dyn_x_dir - sta_x_dir)),
            "gap_dynamic_mean_x_r2_vs_main_test": float(np.mean(dyn_x_r2) - float(baseline_metrics["x_r2"])),
            "gap_dynamic_mean_x_rmse_vs_main_test": float(np.mean(dyn_x_rmse) - float(baseline_metrics["x_rmse"])),
        },
        "folds": folds,
    }

    (out_dir / "walk_forward_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with (out_dir / "walk_forward_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "date_start",
                "date_end",
                "n_eval",
                "dynamic_x_r2",
                "dynamic_x_rmse",
                "dynamic_x_direction",
                "static_x_r2",
                "static_x_rmse",
                "static_x_direction",
                "gap_x_r2",
                "gap_x_rmse",
                "gap_x_direction",
                "dynamic_condition_number",
                "dynamic_train_target_r2",
            ],
        )
        writer.writeheader()
        for frow in folds:
            writer.writerow(
                {
                    "date_start": frow["date_start"],
                    "date_end": frow["date_end"],
                    "n_eval": frow["n_eval"],
                    "dynamic_x_r2": frow["dynamic"]["x_r2"],
                    "dynamic_x_rmse": frow["dynamic"]["x_rmse"],
                    "dynamic_x_direction": frow["dynamic"]["x_direction"],
                    "static_x_r2": frow["static"]["x_r2"],
                    "static_x_rmse": frow["static"]["x_rmse"],
                    "static_x_direction": frow["static"]["x_direction"],
                    "gap_x_r2": frow["gap_dynamic_minus_static"]["x_r2"],
                    "gap_x_rmse": frow["gap_dynamic_minus_static"]["x_rmse"],
                    "gap_x_direction": frow["gap_dynamic_minus_static"]["x_direction"],
                    "dynamic_condition_number": frow["dynamic_condition_number"],
                    "dynamic_train_target_r2": frow["dynamic_train_target_r2"],
                }
            )

    agg = summary["aggregate"]
    print("[done] step3 walk-forward complete")
    print(f"[done] folds={len(folds)}")
    print(
        "[done] dynamic mean: "
        f"x_r2={agg['dynamic_mean_x_r2']:.6g}, x_rmse={agg['dynamic_mean_x_rmse']:.6g}, x_dir={agg['dynamic_mean_x_direction']:.6g}"
    )
    print(
        "[done] static mean: "
        f"x_r2={agg['static_mean_x_r2']:.6g}, x_rmse={agg['static_mean_x_rmse']:.6g}, x_dir={agg['static_mean_x_direction']:.6g}"
    )
    print(
        "[done] gap(dynamic-static): "
        f"x_r2={agg['gap_dynamic_minus_static_x_r2']:+.6g}, x_rmse={agg['gap_dynamic_minus_static_x_rmse']:+.6g}, "
        f"x_dir={agg['gap_dynamic_minus_static_x_direction']:+.6g}"
    )
    print(f"[done] output: {out_dir}")


if __name__ == "__main__":
    main()
