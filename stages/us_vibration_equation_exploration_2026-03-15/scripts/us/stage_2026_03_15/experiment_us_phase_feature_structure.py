"""Step 9 experiment: scan feature structures for phase de-lag."""

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

from identification.ols_identifier import identify_ols_y_next_from_regression_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan feature structures to reduce phase lag.")
    parser.add_argument("--panel", default="data/us/us_regression_panel.csv")
    parser.add_argument("--baseline-est", default="data/us/us_ols_estimation.json")
    parser.add_argument("--baseline-metrics", default="data/us/plot/fit_metrics.json")
    parser.add_argument("--baseline-phase", default="data/us/experiments/step5_phase_bias_baseline/phase_bias_summary.json")
    parser.add_argument("--output-dir", default="data/us/experiments/step9_phase_feature_structure")
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


def _extract_test_series(rows_test: list[dict[str, str]], feature_names: list[str], est: dict, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.array([_to_float(r["Y"]) for r in rows_test], dtype=float)
    u = np.array([[_to_float(r[c]) for c in feature_names] for r in rows_test], dtype=float)
    finite = np.isfinite(y) & np.all(np.isfinite(u), axis=1)
    y = y[finite]
    u = u[finite]
    if bool(est.get("orthogonalize_country", False)):
        u = _orthogonalize_country_features(u, feature_names)

    m = float(est["m"])
    c = float(est["c"])
    c_nl = float(est.get("c_nl", 0.0))
    k = float(est["k"])
    intercept = float(est.get("intercept", 0.0))
    beta_vec = np.array([float(est["betas"][name]) / m for name in feature_names], dtype=float)

    idx = np.arange(1, len(y) - 1)
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
    y_true = y[idx + 1]
    y_anchor = y[idx]
    return y_true, y_hat, y_anchor


def _build_feature_groups(all_features: list[str], baseline_betas: dict[str, float]) -> list[tuple[str, list[str]]]:
    base = [c for c in all_features if not c.endswith("_lag1")]
    lag1 = [c for c in all_features if c.endswith("_lag1")]
    g_lag1 = [c for c in lag1 if c.startswith("g_")]
    c_lag1 = [c for c in lag1 if c.startswith("c_")]

    lag1_sorted = sorted(lag1, key=lambda c: abs(float(baseline_betas.get(c, 0.0))), reverse=True)
    top4 = lag1_sorted[:4]

    groups: list[tuple[str, list[str]]] = []
    groups.append(("full_main", list(all_features)))
    groups.append(("no_lag1", list(base)))
    groups.append(("base_plus_g_lag1", list(base + g_lag1)))
    groups.append(("base_plus_c_lag1", list(base + c_lag1)))
    groups.append(("base_plus_top4_lag1", list(base + top4)))
    groups.append(("lag1_only", list(lag1)))

    # deduplicate while preserving first occurrence
    dedup: list[tuple[str, list[str]]] = []
    seen_keys: set[tuple[str, ...]] = set()
    for name, cols in groups:
        uniq = []
        s: set[str] = set()
        for c in cols:
            if c not in s:
                s.add(c)
                uniq.append(c)
        key = tuple(uniq)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        dedup.append((name, uniq))
    return dedup


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_est = json.loads(Path(args.baseline_est).read_text(encoding="utf-8"))
    baseline_metrics = json.loads(Path(args.baseline_metrics).read_text(encoding="utf-8"))
    baseline_phase = json.loads(Path(args.baseline_phase).read_text(encoding="utf-8"))
    rows_all = _load_rows(Path(args.panel))
    rows_test = _split_rows(rows_all, split="test")

    baseline_features = list(baseline_est.get("betas", {}).keys())
    if not baseline_features:
        baseline_features = [c for c in rows_all[0].keys() if c.startswith("g_") or c.startswith("c_") or c.startswith("i_")]

    groups = _build_feature_groups(baseline_features, baseline_betas=baseline_est.get("betas", {}))
    results: list[dict[str, object]] = []

    for name, feat_cols in groups:
        if len(feat_cols) < 2:
            results.append({"name": name, "n_features": len(feat_cols), "status": "skipped: too few features"})
            continue
        try:
            est_obj = identify_ols_y_next_from_regression_panel(
                panel_path=args.panel,
                split="train",
                feature_cols=feat_cols,
                m=float(baseline_est.get("m", 1.0)),
                dt=args.dt,
                fit_intercept=True,
                ridge_alpha=float(baseline_est.get("ridge_alpha", 0.05)),
                enforce_physical=bool(baseline_est.get("enforce_physical", True)),
                c_min=float(baseline_est.get("c_min", 1e-4)),
                k_min=float(baseline_est.get("k_min", 0.0)),
                orthogonalize_country=bool(baseline_est.get("orthogonalize_country", True)),
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
            est = {
                "m": est_obj.m,
                "c": est_obj.c,
                "c_nl": est_obj.c_nl,
                "k": est_obj.k,
                "intercept": est_obj.intercept,
                "betas": est_obj.betas,
                "orthogonalize_country": bool(baseline_est.get("orthogonalize_country", True)),
            }
            y_true, y_hat, y_anchor = _extract_test_series(rows_test, feature_names=feat_cols, est=est, dt=args.dt)
            lag_scan = _lag_scan(y_true, y_hat, y_anchor, max_lag=args.max_lag)
            best = _best_lag(lag_scan)
            turn = _turn_metrics(y_true, y_hat, y_anchor, eps=args.turn_eps, tol=args.turn_tol)
            rec = {
                "name": name,
                "status": "ok",
                "n_features": len(feat_cols),
                "n_obs_eval": int(len(y_true)),
                "features": feat_cols,
                "train_target_r2": float(est_obj.r2),
                "condition_number": float(est_obj.condition_number),
                "test": {
                    "x_r2": _r2(y_true, y_hat),
                    "x_rmse": float(np.sqrt(np.mean((y_true - y_hat) ** 2))),
                    "x_corr": _corr(y_true, y_hat),
                    "x_direction": _direction_acc(y_true, y_hat, y_anchor),
                },
                "phase": {
                    "best_lag": int(best["lag"]),
                    "best_lag_corr": float(best["corr"]),
                    "best_lag_rmse": float(best["rmse"]),
                },
                "turn": turn,
            }
            rec["gap_vs_main"] = {
                "x_r2": float(rec["test"]["x_r2"] - float(baseline_metrics["x_r2"])),
                "x_rmse": float(rec["test"]["x_rmse"] - float(baseline_metrics["x_rmse"])),
                "best_lag": int(rec["phase"]["best_lag"] - int(baseline_phase["best_lag_by_corr"]["lag"])),
                "turn_recall": float(rec["turn"]["turn_hit_rate_recall"] - float(baseline_phase["turn_metrics_lag0"]["turn_hit_rate_recall"])),
                "turn_precision": float(rec["turn"]["turn_precision"] - float(baseline_phase["turn_metrics_lag0"]["turn_precision"])),
                "turn_mean_delay": float(rec["turn"]["mean_delay"] - float(baseline_phase["turn_metrics_lag0"]["mean_delay"])),
            }
        except Exception as e:
            rec = {"name": name, "n_features": len(feat_cols), "status": f"failed: {e}"}
        results.append(rec)

    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        raise RuntimeError("all feature-structure candidates failed")

    best_phase = sorted(
        ok,
        key=lambda r: (
            abs(int(r["phase"]["best_lag"])),
            float(r["test"]["x_r2"]),
            -float(r["test"]["x_rmse"]),
        ),
        reverse=False,
    )[0]
    best_overall = sorted(ok, key=lambda r: (float(r["test"]["x_r2"]), -float(r["test"]["x_rmse"])), reverse=True)[0]

    summary = {
        "baseline": {
            "estimation_path": str(args.baseline_est),
            "metrics_path": str(args.baseline_metrics),
            "phase_path": str(args.baseline_phase),
            "main_x_r2": float(baseline_metrics["x_r2"]),
            "main_x_rmse": float(baseline_metrics["x_rmse"]),
            "main_best_lag": int(baseline_phase["best_lag_by_corr"]["lag"]),
        },
        "candidates": results,
        "best_phase": best_phase,
        "best_overall": best_overall,
    }
    (out_dir / "feature_structure_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with (out_dir / "feature_structure_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "status",
                "n_features",
                "train_target_r2",
                "condition_number",
                "x_r2",
                "x_rmse",
                "x_corr",
                "x_direction",
                "best_lag",
                "best_lag_corr",
                "best_lag_rmse",
                "turn_recall",
                "turn_precision",
                "turn_mean_delay",
                "gap_x_r2_vs_main",
                "gap_x_rmse_vs_main",
            ],
        )
        writer.writeheader()
        for r in results:
            if r.get("status") != "ok":
                writer.writerow(
                    {
                        "name": r.get("name"),
                        "status": r.get("status"),
                        "n_features": r.get("n_features"),
                    }
                )
                continue
            writer.writerow(
                {
                    "name": r["name"],
                    "status": r["status"],
                    "n_features": r["n_features"],
                    "train_target_r2": r["train_target_r2"],
                    "condition_number": r["condition_number"],
                    "x_r2": r["test"]["x_r2"],
                    "x_rmse": r["test"]["x_rmse"],
                    "x_corr": r["test"]["x_corr"],
                    "x_direction": r["test"]["x_direction"],
                    "best_lag": r["phase"]["best_lag"],
                    "best_lag_corr": r["phase"]["best_lag_corr"],
                    "best_lag_rmse": r["phase"]["best_lag_rmse"],
                    "turn_recall": r["turn"]["turn_hit_rate_recall"],
                    "turn_precision": r["turn"]["turn_precision"],
                    "turn_mean_delay": r["turn"]["mean_delay"],
                    "gap_x_r2_vs_main": r["gap_vs_main"]["x_r2"],
                    "gap_x_rmse_vs_main": r["gap_vs_main"]["x_rmse"],
                }
            )

    print("[done] step9 feature-structure scan complete")
    print(f"[done] candidates={len(results)}, ok={len(ok)}")
    print(
        "[done] best_phase: "
        f"{best_phase['name']}, best_lag={best_phase['phase']['best_lag']}, "
        f"x_r2={best_phase['test']['x_r2']:.6g}, x_rmse={best_phase['test']['x_rmse']:.6g}"
    )
    print(
        "[done] best_overall: "
        f"{best_overall['name']}, best_lag={best_overall['phase']['best_lag']}, "
        f"x_r2={best_overall['test']['x_r2']:.6g}, x_rmse={best_overall['test']['x_rmse']:.6g}"
    )
    print(f"[done] output: {out_dir}")


if __name__ == "__main__":
    main()
