"""Step 4 experiment: tail-weight training for extreme scenarios, compared to main model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from datetime import datetime
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from identification.ols_identifier import identify_ols_y_next_from_regression_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan tail-weight configs for extreme-case performance.")
    parser.add_argument("--panel", default="data/us/us_regression_panel.csv")
    parser.add_argument("--baseline-est", default="data/us/us_ols_estimation.json")
    parser.add_argument("--baseline-metrics", default="data/us/plot/fit_metrics.json")
    parser.add_argument("--output-dir", default="data/us/experiments/step4_tail_weight_extreme")
    parser.add_argument("--tail-quantile", type=float, default=0.9, help="Quantile for defining extreme test samples by |a|.")
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


def _split_rows(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    if split == "all":
        return rows
    col = f"is_{split}"
    out = [r for r in rows if (r.get(col) or "").strip() == "1"]
    if not out:
        raise ValueError(f"no rows for split={split}")
    return out


def _extract(rows: list[dict[str, str]], features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dates = np.array([datetime.strptime(r["date"], "%Y-%m-%d") for r in rows])
    a = np.array([_to_float(r["a"]) for r in rows], dtype=float)
    v = np.array([_to_float(r["v"]) for r in rows], dtype=float)
    y = np.array([_to_float(r["Y"]) for r in rows], dtype=float)
    u = np.array([[_to_float(r[c]) for c in features] for r in rows], dtype=float)
    return dates, a, v, y, u


def _eval_metrics(rows_test: list[dict[str, str]], feature_names: list[str], est: dict, dt: float, tail_q: float) -> dict[str, float]:
    _, a, v, y, u = _extract(rows_test, feature_names)
    if bool(est.get("orthogonalize_country", False)):
        u = _orthogonalize_country_features(u, feature_names)

    finite = np.isfinite(a) & np.isfinite(v) & np.isfinite(y) & np.all(np.isfinite(u), axis=1)
    a = a[finite]
    v = v[finite]
    y = y[finite]
    u = u[finite]

    m = float(est["m"])
    c = float(est["c"])
    c_nl = float(est.get("c_nl", 0.0))
    k = float(est["k"])
    intercept = float(est.get("intercept", 0.0))
    beta_vec = np.array([float(est["betas"][cname]) / m for cname in feature_names], dtype=float)

    v_used = v.copy()
    if len(v_used) > 1:
        v_used[1:] = (y[1:] - y[:-1]) / dt
        v_used[0] = v_used[1]

    a_hat = intercept + (-c / m) * v_used + (-c_nl / m) * np.abs(v_used) * v_used + (-k / m) * y + u @ beta_vec

    idx = np.arange(1, len(y) - 1)
    v_step = (y[idx] - y[idx - 1]) / dt
    dt2 = dt * dt
    y_hat = 2.0 * y[idx] - y[idx - 1] + dt2 * (
        intercept - (c / m) * v_step - (c_nl / m) * np.abs(v_step) * v_step - (k / m) * y[idx] + (u[idx] @ beta_vec)
    )
    y_true = y[idx + 1]

    overall = {
        "r2": _r2(a, a_hat),
        "rmse": float(np.sqrt(np.mean((a - a_hat) ** 2))),
        "direction_accuracy": float(np.mean(np.sign(a) == np.sign(a_hat))),
        "x_r2": _r2(y_true, y_hat),
        "x_rmse": float(np.sqrt(np.mean((y_true - y_hat) ** 2))),
    }

    if len(a) < 10:
        tail = {"n_tail": 0, "a_r2": float("nan"), "a_rmse": float("nan"), "a_direction": float("nan"), "x_r2": float("nan"), "x_rmse": float("nan")}
        return {"overall": overall, "tail": tail}

    th = float(np.quantile(np.abs(a), tail_q))
    tail_mask_a = np.abs(a) >= th
    a_tail = a[tail_mask_a]
    a_hat_tail = a_hat[tail_mask_a]

    # align x-tail by center acceleration a[idx]
    a_center = a[idx]
    tail_mask_x = np.abs(a_center) >= th
    y_true_tail = y_true[tail_mask_x]
    y_hat_tail = y_hat[tail_mask_x]

    tail = {
        "n_tail": int(np.sum(tail_mask_a)),
        "a_r2": _r2(a_tail, a_hat_tail) if len(a_tail) >= 3 else float("nan"),
        "a_rmse": float(np.sqrt(np.mean((a_tail - a_hat_tail) ** 2))) if len(a_tail) > 0 else float("nan"),
        "a_direction": float(np.mean(np.sign(a_tail) == np.sign(a_hat_tail))) if len(a_tail) > 0 else float("nan"),
        "x_r2": _r2(y_true_tail, y_hat_tail) if len(y_true_tail) >= 3 else float("nan"),
        "x_rmse": float(np.sqrt(np.mean((y_true_tail - y_hat_tail) ** 2))) if len(y_true_tail) > 0 else float("nan"),
    }

    return {"overall": overall, "tail": tail}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_est = json.loads(Path(args.baseline_est).read_text(encoding="utf-8"))
    baseline_metrics = json.loads(Path(args.baseline_metrics).read_text(encoding="utf-8"))
    rows_all = _load_rows(Path(args.panel))
    rows_test = _split_rows(rows_all, "test")

    features = list(baseline_est.get("betas", {}).keys())
    if not features:
        features = [c for c in rows_all[0].keys() if c.startswith("g_") or c.startswith("c_") or c.startswith("i_")]

    configs = [
        {"tail_weight_mode": "none", "tail_weight_q": 0.9, "tail_weight_scale": 3.0, "tail_weight_power": 1.0, "name": "main_like_none"},
        {"tail_weight_mode": "abs_power", "tail_weight_q": 0.85, "tail_weight_scale": 2.0, "tail_weight_power": 1.0, "name": "tw_q85_s2_p1"},
        {"tail_weight_mode": "abs_power", "tail_weight_q": 0.9, "tail_weight_scale": 2.0, "tail_weight_power": 1.0, "name": "tw_q90_s2_p1"},
        {"tail_weight_mode": "abs_power", "tail_weight_q": 0.9, "tail_weight_scale": 3.0, "tail_weight_power": 1.0, "name": "tw_q90_s3_p1"},
        {"tail_weight_mode": "abs_power", "tail_weight_q": 0.9, "tail_weight_scale": 4.0, "tail_weight_power": 1.0, "name": "tw_q90_s4_p1"},
        {"tail_weight_mode": "abs_power", "tail_weight_q": 0.93, "tail_weight_scale": 3.0, "tail_weight_power": 1.0, "name": "tw_q93_s3_p1"},
        {"tail_weight_mode": "abs_power", "tail_weight_q": 0.9, "tail_weight_scale": 3.0, "tail_weight_power": 1.5, "name": "tw_q90_s3_p15"},
        {"tail_weight_mode": "abs_power", "tail_weight_q": 0.9, "tail_weight_scale": 5.0, "tail_weight_power": 1.5, "name": "tw_q90_s5_p15"},
    ]

    candidates = []

    for cfg in configs:
        try:
            est_obj = identify_ols_y_next_from_regression_panel(
                panel_path=args.panel,
                split="train",
                feature_cols=features,
                m=float(baseline_est.get("m", 1.0)),
                dt=args.dt,
                fit_intercept=True,
                ridge_alpha=float(baseline_est.get("ridge_alpha", 0.05)),
                enforce_physical=bool(baseline_est.get("enforce_physical", True)),
                c_min=float(baseline_est.get("c_min", 1e-4)),
                k_min=float(baseline_est.get("k_min", 0.0)),
                orthogonalize_country=bool(baseline_est.get("orthogonalize_country", True)),
                damping_mode="nonlinear_absv",
                c_nl_min=float(baseline_est.get("c_nl_min", 0.0)),
                tail_weight_mode=cfg["tail_weight_mode"],
                tail_weight_q=float(cfg["tail_weight_q"]),
                tail_weight_scale=float(cfg["tail_weight_scale"]),
                tail_weight_power=float(cfg["tail_weight_power"]),
                robust_mode=str(baseline_est.get("robust_mode", "none")),
                robust_tuning=float(baseline_est.get("robust_tuning", 1.345)),
                robust_max_iter=int(baseline_est.get("robust_max_iter", 20)),
                robust_tol=float(baseline_est.get("robust_tol", 1e-6)),
            )
            est = {
                "m": est_obj.m,
                "c": est_obj.c,
                "c_nl": est_obj.c_nl,
                "k": est_obj.k,
                "intercept": est_obj.intercept,
                "betas": est_obj.betas,
                "orthogonalize_country": bool(baseline_est.get("orthogonalize_country", True)),
            }
            metrics = _eval_metrics(rows_test, features, est, dt=args.dt, tail_q=args.tail_quantile)
            rec = {
                "name": cfg["name"],
                "train_target_r2": float(est_obj.r2),
                "condition_number": float(est_obj.condition_number),
                "config": cfg,
                "metrics": metrics,
                "gap_vs_main": {
                    "overall_x_r2": float(metrics["overall"]["x_r2"] - float(baseline_metrics["x_r2"])),
                    "overall_x_rmse": float(metrics["overall"]["x_rmse"] - float(baseline_metrics["x_rmse"])),
                    "overall_direction": float(metrics["overall"]["direction_accuracy"] - float(baseline_metrics["direction_accuracy"])),
                },
                "status": "ok",
            }
        except Exception as e:
            rec = {"name": cfg["name"], "config": cfg, "status": f"failed: {e}"}
        candidates.append(rec)

    ok = [r for r in candidates if r.get("status") == "ok"]
    if not ok:
        raise RuntimeError("all tail-weight configs failed")

    ok.sort(
        key=lambda r: (
            float(r["metrics"]["tail"]["x_r2"] if np.isfinite(r["metrics"]["tail"]["x_r2"]) else -1e9),
            float(r["metrics"]["tail"]["a_direction"] if np.isfinite(r["metrics"]["tail"]["a_direction"]) else -1e9),
            -float(r["metrics"]["tail"]["x_rmse"] if np.isfinite(r["metrics"]["tail"]["x_rmse"]) else 1e9),
        ),
        reverse=True,
    )
    best = ok[0]

    summary = {
        "baseline": {
            "estimation_path": str(args.baseline_est),
            "metrics_path": str(args.baseline_metrics),
            "overall_test_metrics": baseline_metrics,
            "tail_quantile": args.tail_quantile,
        },
        "best": best,
        "candidates": candidates,
    }

    (out_dir / "tail_weight_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with (out_dir / "tail_weight_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "tail_mode",
                "tail_q",
                "tail_scale",
                "tail_power",
                "train_target_r2",
                "condition_number",
                "overall_x_r2",
                "overall_x_rmse",
                "overall_direction",
                "tail_n",
                "tail_a_r2",
                "tail_a_rmse",
                "tail_a_direction",
                "tail_x_r2",
                "tail_x_rmse",
                "gap_overall_x_r2_vs_main",
                "status",
            ],
        )
        writer.writeheader()
        for r in candidates:
            if r.get("status") != "ok":
                cfg = r.get("config", {})
                writer.writerow(
                    {
                        "name": r.get("name"),
                        "tail_mode": cfg.get("tail_weight_mode"),
                        "tail_q": cfg.get("tail_weight_q"),
                        "tail_scale": cfg.get("tail_weight_scale"),
                        "tail_power": cfg.get("tail_weight_power"),
                        "status": r.get("status"),
                    }
                )
                continue
            cfg = r["config"]
            m = r["metrics"]
            writer.writerow(
                {
                    "name": r["name"],
                    "tail_mode": cfg["tail_weight_mode"],
                    "tail_q": cfg["tail_weight_q"],
                    "tail_scale": cfg["tail_weight_scale"],
                    "tail_power": cfg["tail_weight_power"],
                    "train_target_r2": r["train_target_r2"],
                    "condition_number": r["condition_number"],
                    "overall_x_r2": m["overall"]["x_r2"],
                    "overall_x_rmse": m["overall"]["x_rmse"],
                    "overall_direction": m["overall"]["direction_accuracy"],
                    "tail_n": m["tail"]["n_tail"],
                    "tail_a_r2": m["tail"]["a_r2"],
                    "tail_a_rmse": m["tail"]["a_rmse"],
                    "tail_a_direction": m["tail"]["a_direction"],
                    "tail_x_r2": m["tail"]["x_r2"],
                    "tail_x_rmse": m["tail"]["x_rmse"],
                    "gap_overall_x_r2_vs_main": r["gap_vs_main"]["overall_x_r2"],
                    "status": "ok",
                }
            )

    print("[done] step4 tail-weight scan complete")
    print(f"[done] candidates={len(candidates)}, ok={len(ok)}")
    print(
        "[done] best: "
        f"{best['name']}, tail_x_r2={best['metrics']['tail']['x_r2']:.6g}, tail_x_rmse={best['metrics']['tail']['x_rmse']:.6g}, "
        f"overall_x_r2={best['metrics']['overall']['x_r2']:.6g}"
    )
    print(f"[done] output: {out_dir}")


if __name__ == "__main__":
    main()
