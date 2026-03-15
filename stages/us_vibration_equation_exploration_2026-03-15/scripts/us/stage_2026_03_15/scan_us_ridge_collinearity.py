"""Step 1 experiment: ridge + collinearity reduction scan for US nonlinear main model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from datetime import datetime

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from identification.ols_identifier import identify_ols_y_next_from_regression_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan ridge and collinearity settings against current main model.")
    parser.add_argument("--panel", default="data/us/us_regression_panel.csv")
    parser.add_argument("--baseline-est", default="data/us/us_ols_estimation.json")
    parser.add_argument("--baseline-metrics", default="data/us/plot/fit_metrics.json")
    parser.add_argument("--output-dir", default="data/us/experiments/step1_ridge_collinearity")
    parser.add_argument("--ridge-grid", default="0.01,0.03,0.05,0.08,0.1,0.15,0.2")
    parser.add_argument("--corr-threshold-grid", default="none,0.99,0.97,0.95,0.93,0.90,0.88,0.85")
    parser.add_argument("--min-features", type=int, default=8)
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


def _load_rows(panel_path: Path) -> list[dict[str, str]]:
    with panel_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty panel file: {panel_path}")
    return rows


def _split_rows(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    if split == "all":
        return rows
    col = f"is_{split}"
    out = [r for r in rows if (r.get(col) or "").strip() == "1"]
    if not out:
        raise ValueError(f"no rows for split={split}")
    return out


def _extract_arrays(rows: list[dict[str, str]], features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dates = np.array([datetime.strptime(r["date"], "%Y-%m-%d") for r in rows])
    a = np.array([_to_float(r["a"]) for r in rows], dtype=float)
    v = np.array([_to_float(r["v"]) for r in rows], dtype=float)
    y = np.array([_to_float(r["Y"]) for r in rows], dtype=float)
    u = np.array([[_to_float(r[c]) for c in features] for r in rows], dtype=float)
    return dates, a, v, y, u


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return 0.0
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 1e-12 or sy <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _select_features_by_corr(
    y_train: np.ndarray,
    u_train: np.ndarray,
    feature_names: list[str],
    threshold: float,
    min_features: int,
    dt: float,
) -> list[str]:
    idx = np.arange(1, len(y_train) - 1)
    finite = np.isfinite(y_train[idx - 1]) & np.isfinite(y_train[idx]) & np.isfinite(y_train[idx + 1]) & np.all(np.isfinite(u_train[idx]), axis=1)
    idx = idx[finite]
    if len(idx) < 10:
        return feature_names

    y_prev = y_train[idx - 1]
    y_cur = y_train[idx]
    y_next = y_train[idx + 1]
    target = y_next - 2.0 * y_cur + y_prev
    x = u_train[idx]

    scores = []
    for j in range(x.shape[1]):
        scores.append(abs(_corr(x[:, j], target)))
    order = np.argsort(np.asarray(scores))[::-1]

    corr_mat = np.corrcoef(x, rowvar=False)
    corr_mat = np.nan_to_num(corr_mat, nan=0.0)

    kept: list[int] = []
    for j in order:
        if not kept:
            kept.append(int(j))
            continue
        if all(abs(float(corr_mat[j, k])) < threshold for k in kept):
            kept.append(int(j))

    if len(kept) < min_features:
        for j in order:
            if int(j) not in kept:
                kept.append(int(j))
                if len(kept) >= min_features:
                    break

    kept = sorted(set(kept))
    return [feature_names[i] for i in kept]


def _predict_y_next_nonlin(
    y: np.ndarray,
    u: np.ndarray,
    beta_vec: np.ndarray,
    *,
    c: float,
    c_nl: float,
    k: float,
    m: float,
    intercept: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(y) < 3:
        return np.array([], dtype=float), np.array([], dtype=float)
    idx = np.arange(1, len(y) - 1)
    v = (y[idx] - y[idx - 1]) / dt
    dt2 = dt * dt
    y_hat = 2.0 * y[idx] - y[idx - 1] + dt2 * (
        intercept - (c / m) * v - (c_nl / m) * np.abs(v) * v - (k / m) * y[idx] + (u[idx] @ beta_vec)
    )
    y_true = y[idx + 1]
    return y_true, y_hat


def _eval_test_metrics(rows_test: list[dict[str, str]], feature_names: list[str], est: dict, dt: float) -> dict[str, float]:
    _, a, v, y, u = _extract_arrays(rows_test, feature_names)
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

    y_true, y_hat = _predict_y_next_nonlin(y, u, beta_vec, c=c, c_nl=c_nl, k=k, m=m, intercept=intercept, dt=dt)
    finite_x = np.isfinite(y_true) & np.isfinite(y_hat)
    y_true = y_true[finite_x]
    y_hat = y_hat[finite_x]

    mse = float(np.mean((a - a_hat) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(a - a_hat)))
    r2 = _r2(a, a_hat)
    corr = float(np.corrcoef(a, a_hat)[0, 1]) if len(a) > 1 else float("nan")
    direction_accuracy = float(np.mean(np.sign(a) == np.sign(a_hat))) if len(a) else float("nan")

    x_mse = float(np.mean((y_true - y_hat) ** 2))
    x_rmse = float(np.sqrt(x_mse))
    x_mae = float(np.mean(np.abs(y_true - y_hat)))
    x_r2 = _r2(y_true, y_hat)
    x_corr = float(np.corrcoef(y_true, y_hat)[0, 1]) if len(y_true) > 1 else float("nan")

    return {
        "n_obs": int(len(a)),
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "corr": corr,
        "direction_accuracy": direction_accuracy,
        "x_eval_n_obs": int(len(y_true)),
        "x_r2": x_r2,
        "x_rmse": x_rmse,
        "x_mae": x_mae,
        "x_corr": x_corr,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_path = Path(args.panel)
    rows_all = _load_rows(panel_path)
    baseline_est = json.loads(Path(args.baseline_est).read_text(encoding="utf-8"))
    baseline_metrics = json.loads(Path(args.baseline_metrics).read_text(encoding="utf-8"))

    all_features = [c for c in rows_all[0].keys() if c.startswith("g_") or c.startswith("c_") or c.startswith("i_")]

    rows_train = _split_rows(rows_all, "train")
    rows_test = _split_rows(rows_all, "test")

    _, _, _, y_train, u_train = _extract_arrays(rows_train, all_features)
    if bool(baseline_est.get("orthogonalize_country", False)):
        u_train = _orthogonalize_country_features(u_train, all_features)

    ridge_grid = [float(x.strip()) for x in args.ridge_grid.split(",") if x.strip()]
    corr_tokens = [x.strip().lower() for x in args.corr_threshold_grid.split(",") if x.strip()]

    corr_grid: list[float | None] = []
    for tok in corr_tokens:
        if tok in {"none", "all", "na"}:
            corr_grid.append(None)
        else:
            corr_grid.append(float(tok))

    candidates: list[dict[str, object]] = []

    for thr in corr_grid:
        if thr is None:
            selected_features = all_features[:]
        else:
            selected_features = _select_features_by_corr(
                y_train=y_train,
                u_train=u_train,
                feature_names=all_features,
                threshold=thr,
                min_features=args.min_features,
                dt=args.dt,
            )

        for ridge_alpha in ridge_grid:
            try:
                est_obj = identify_ols_y_next_from_regression_panel(
                    panel_path=panel_path,
                    split="train",
                    feature_cols=selected_features,
                    m=float(baseline_est.get("m", 1.0)),
                    dt=args.dt,
                    fit_intercept=True,
                    ridge_alpha=ridge_alpha,
                    enforce_physical=bool(baseline_est.get("enforce_physical", True)),
                    c_min=float(baseline_est.get("c_min", 1e-4)),
                    k_min=float(baseline_est.get("k_min", 0.0)),
                    orthogonalize_country=bool(baseline_est.get("orthogonalize_country", True)),
                    damping_mode="nonlinear_absv",
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
            except Exception as e:
                candidates.append(
                    {
                        "ridge_alpha": ridge_alpha,
                        "corr_threshold": thr,
                        "n_features": len(selected_features),
                        "status": f"failed: {e}",
                    }
                )
                continue

            est_payload = {
                "m": est_obj.m,
                "c": est_obj.c,
                "c_nl": est_obj.c_nl,
                "k": est_obj.k,
                "x0": est_obj.x0,
                "intercept": est_obj.intercept,
                "betas": est_obj.betas,
                "orthogonalize_country": bool(baseline_est.get("orthogonalize_country", True)),
            }
            test_metrics = _eval_test_metrics(rows_test, selected_features, est_payload, dt=args.dt)

            rec = {
                "ridge_alpha": ridge_alpha,
                "corr_threshold": thr,
                "n_features": len(selected_features),
                "features": selected_features,
                "train_target_r2": float(est_obj.r2),
                "condition_number": float(est_obj.condition_number),
                "c": float(est_obj.c),
                "c_nl": float(est_obj.c_nl),
                "k": float(est_obj.k),
                "test_metrics": test_metrics,
                "gap_vs_main": {
                    "test_r2": float(test_metrics["r2"] - float(baseline_metrics["r2"])),
                    "test_rmse": float(test_metrics["rmse"] - float(baseline_metrics["rmse"])),
                    "test_direction": float(test_metrics["direction_accuracy"] - float(baseline_metrics["direction_accuracy"])),
                    "test_x_r2": float(test_metrics["x_r2"] - float(baseline_metrics["x_r2"])),
                    "test_x_rmse": float(test_metrics["x_rmse"] - float(baseline_metrics["x_rmse"])),
                },
                "status": "ok",
            }
            candidates.append(rec)

    ok_rows = [r for r in candidates if r.get("status") == "ok"]
    if not ok_rows:
        raise RuntimeError("all candidates failed")

    ok_rows.sort(
        key=lambda r: (
            float(r["test_metrics"]["x_r2"]),
            float(r["test_metrics"]["direction_accuracy"]),
            -float(r["condition_number"]),
        ),
        reverse=True,
    )
    best = ok_rows[0]

    summary = {
        "baseline": {
            "estimation_path": str(args.baseline_est),
            "metrics_path": str(args.baseline_metrics),
            "ridge_alpha": float(baseline_est.get("ridge_alpha", 0.05)),
            "feature_count": int(len(baseline_est.get("betas", {}))),
            "condition_number": float(baseline_est.get("condition_number", float("nan"))),
            "test_metrics": baseline_metrics,
        },
        "grid": {
            "ridge_alpha": ridge_grid,
            "corr_threshold": [None if x is None else float(x) for x in corr_grid],
            "min_features": args.min_features,
        },
        "best": best,
        "candidates": candidates,
    }

    (output_dir / "scan_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = output_dir / "scan_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ridge_alpha",
                "corr_threshold",
                "n_features",
                "train_target_r2",
                "condition_number",
                "c",
                "c_nl",
                "k",
                "test_r2",
                "test_rmse",
                "test_direction_accuracy",
                "test_x_r2",
                "test_x_rmse",
                "gap_test_r2",
                "gap_test_x_r2",
                "gap_test_direction",
                "status",
            ],
        )
        writer.writeheader()
        for r in candidates:
            if r.get("status") != "ok":
                writer.writerow(
                    {
                        "ridge_alpha": r.get("ridge_alpha"),
                        "corr_threshold": r.get("corr_threshold"),
                        "n_features": r.get("n_features"),
                        "status": r.get("status"),
                    }
                )
                continue
            tm = r["test_metrics"]
            gp = r["gap_vs_main"]
            writer.writerow(
                {
                    "ridge_alpha": r["ridge_alpha"],
                    "corr_threshold": r["corr_threshold"],
                    "n_features": r["n_features"],
                    "train_target_r2": r["train_target_r2"],
                    "condition_number": r["condition_number"],
                    "c": r["c"],
                    "c_nl": r["c_nl"],
                    "k": r["k"],
                    "test_r2": tm["r2"],
                    "test_rmse": tm["rmse"],
                    "test_direction_accuracy": tm["direction_accuracy"],
                    "test_x_r2": tm["x_r2"],
                    "test_x_rmse": tm["x_rmse"],
                    "gap_test_r2": gp["test_r2"],
                    "gap_test_x_r2": gp["test_x_r2"],
                    "gap_test_direction": gp["test_direction"],
                    "status": "ok",
                }
            )

    print("[done] step1 scan complete")
    print(f"[done] candidates={len(candidates)}, ok={len(ok_rows)}")
    print(
        "[done] best: "
        f"ridge={best['ridge_alpha']}, corr_thr={best['corr_threshold']}, n_features={best['n_features']}, "
        f"x_r2={best['test_metrics']['x_r2']:.6g}, dir={best['test_metrics']['direction_accuracy']:.6g}, cond={best['condition_number']:.4g}"
    )
    print(f"[done] output: {output_dir}")


if __name__ == "__main__":
    main()
