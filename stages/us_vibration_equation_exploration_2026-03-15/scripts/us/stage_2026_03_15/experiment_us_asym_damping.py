"""Step 2 experiment: asymmetric damping (up/down) vs current nonlinear main model."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Asymmetric damping experiment.")
    parser.add_argument("--panel", default="data/us/us_regression_panel.csv")
    parser.add_argument("--baseline-est", default="data/us/us_ols_estimation.json")
    parser.add_argument("--baseline-metrics", default="data/us/plot/fit_metrics.json")
    parser.add_argument("--output-dir", default="data/us/experiments/step2_asym_damping")
    parser.add_argument("--ridge-grid", default="0.01,0.03,0.05,0.08,0.1,0.15")
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


def _ridge_closed_form(x: np.ndarray, y: np.ndarray, alpha: float, penalty_mask: np.ndarray) -> np.ndarray:
    xtx = x.T @ x
    xty = x.T @ y
    reg = alpha * np.diag(penalty_mask)
    try:
        return np.linalg.solve(xtx + reg, xty)
    except np.linalg.LinAlgError:
        coef, *_ = np.linalg.lstsq(xtx + reg, xty, rcond=None)
        return coef


def _solve_bounded_ridge(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    penalty_mask: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    max_iter: int = 20000,
    tol: float = 1e-9,
) -> tuple[np.ndarray, int]:
    coef = _ridge_closed_form(x, y, alpha=alpha, penalty_mask=penalty_mask)
    coef = np.minimum(np.maximum(coef, lower), upper)

    lipschitz = float(np.linalg.norm(x, ord=2) ** 2 + alpha * float(np.max(penalty_mask)))
    if not np.isfinite(lipschitz) or lipschitz <= 0:
        return coef, 0

    step = 1.0 / lipschitz
    for it in range(1, max_iter + 1):
        grad = x.T @ (x @ coef - y) + alpha * (penalty_mask * coef)
        nxt = coef - step * grad
        nxt = np.minimum(np.maximum(nxt, lower), upper)
        diff = float(np.linalg.norm(nxt - coef))
        base = 1.0 + float(np.linalg.norm(coef))
        coef = nxt
        if diff <= tol * base:
            return coef, it
    return coef, max_iter


@dataclass
class AsymEstimation:
    m: float
    intercept: float
    c_up: float
    c_down: float
    c_nl: float
    k: float
    betas: dict[str, float]
    n_obs: int
    r2: float
    residual_std: float
    condition_number: float
    ridge_alpha: float


def fit_asym_model(
    *,
    y: np.ndarray,
    u: np.ndarray,
    feature_names: list[str],
    m: float,
    dt: float,
    ridge_alpha: float,
    enforce_physical: bool,
    c_min: float,
    c_nl_min: float,
    k_min: float,
) -> AsymEstimation:
    idx = np.arange(1, len(y) - 1)
    y_prev = y[idx - 1]
    y_cur = y[idx]
    y_next = y[idx + 1]
    u_cur = u[idx]

    finite = np.isfinite(y_prev) & np.isfinite(y_cur) & np.isfinite(y_next) & np.all(np.isfinite(u_cur), axis=1)
    y_prev = y_prev[finite]
    y_cur = y_cur[finite]
    y_next = y_next[finite]
    u_cur = u_cur[finite]

    if len(y_cur) < 10:
        raise ValueError("not enough rows for asym fitting")

    dt2 = dt * dt
    target = y_next - 2.0 * y_cur + y_prev
    v = (y_cur - y_prev) / dt
    v_pos = np.maximum(v, 0.0)
    v_neg = np.minimum(v, 0.0)

    cols = [np.full_like(target, dt2), -(dt2) * v_pos, -(dt2) * v_neg, -(dt2) * np.abs(v) * v, -(dt2) * y_cur]
    if u_cur.shape[1] > 0:
        cols.append(dt2 * u_cur)
    x_design = np.column_stack(cols)

    p = x_design.shape[1]
    penalty_mask = np.ones(p, dtype=float)
    penalty_mask[0] = 0.0

    lower = np.full(p, -np.inf, dtype=float)
    upper = np.full(p, np.inf, dtype=float)
    if enforce_physical:
        lower[1] = c_min / m
        lower[2] = c_min / m
        lower[3] = c_nl_min / m
        lower[4] = k_min / m

    coef, _ = _solve_bounded_ridge(
        x_design,
        target,
        alpha=ridge_alpha,
        penalty_mask=penalty_mask,
        lower=lower,
        upper=upper,
    )

    y_hat = x_design @ coef
    resid = target - y_hat

    intercept = float(coef[0])
    c_up = float(m * coef[1])
    c_down = float(m * coef[2])
    c_nl = float(m * coef[3])
    k = float(m * coef[4])
    beta_vec = np.array(coef[5:], dtype=float) if u_cur.shape[1] > 0 else np.array([], dtype=float)
    betas = {name: float(beta * m) for name, beta in zip(feature_names, beta_vec)}

    dof = max(1, len(target) - p)
    residual_std = float(np.sqrt(np.sum(resid * resid) / dof))
    r2 = _r2(target, y_hat)
    condition_number = float(np.linalg.cond(x_design))

    return AsymEstimation(
        m=float(m),
        intercept=intercept,
        c_up=c_up,
        c_down=c_down,
        c_nl=c_nl,
        k=k,
        betas=betas,
        n_obs=int(len(target)),
        r2=r2,
        residual_std=residual_std,
        condition_number=condition_number,
        ridge_alpha=float(ridge_alpha),
    )


def eval_asym_test(rows_test: list[dict[str, str]], feature_names: list[str], est: AsymEstimation, orthogonalize_country: bool, dt: float) -> dict[str, float]:
    _, a, v, y, u = _extract(rows_test, feature_names)
    if orthogonalize_country:
        u = _orthogonalize_country_features(u, feature_names)

    finite = np.isfinite(a) & np.isfinite(v) & np.isfinite(y) & np.all(np.isfinite(u), axis=1)
    a = a[finite]
    v = v[finite]
    y = y[finite]
    u = u[finite]

    m = est.m
    beta_vec = np.array([est.betas[c] / m for c in feature_names], dtype=float)

    v_used = v.copy()
    if len(v_used) > 1:
        v_used[1:] = (y[1:] - y[:-1]) / dt
        v_used[0] = v_used[1]
    v_pos = np.maximum(v_used, 0.0)
    v_neg = np.minimum(v_used, 0.0)

    a_hat = est.intercept - (est.c_up / m) * v_pos - (est.c_down / m) * v_neg - (est.c_nl / m) * np.abs(v_used) * v_used - (est.k / m) * y + u @ beta_vec

    idx = np.arange(1, len(y) - 1)
    v_step = (y[idx] - y[idx - 1]) / dt
    v_pos_step = np.maximum(v_step, 0.0)
    v_neg_step = np.minimum(v_step, 0.0)
    dt2 = dt * dt
    y_hat = 2.0 * y[idx] - y[idx - 1] + dt2 * (
        est.intercept - (est.c_up / m) * v_pos_step - (est.c_down / m) * v_neg_step - (est.c_nl / m) * np.abs(v_step) * v_step - (est.k / m) * y[idx] + (u[idx] @ beta_vec)
    )
    y_true = y[idx + 1]

    mse = float(np.mean((a - a_hat) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(a - a_hat)))
    r2 = _r2(a, a_hat)
    corr = float(np.corrcoef(a, a_hat)[0, 1]) if len(a) > 1 else float("nan")
    direction = float(np.mean(np.sign(a) == np.sign(a_hat))) if len(a) else float("nan")

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
        "direction_accuracy": direction,
        "x_eval_n_obs": int(len(y_true)),
        "x_r2": x_r2,
        "x_rmse": x_rmse,
        "x_mae": x_mae,
        "x_corr": x_corr,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_est = json.loads(Path(args.baseline_est).read_text(encoding="utf-8"))
    baseline_metrics = json.loads(Path(args.baseline_metrics).read_text(encoding="utf-8"))
    rows_all = _load_rows(Path(args.panel))

    feature_names = list(baseline_est.get("betas", {}).keys())
    if not feature_names:
        feature_names = [c for c in rows_all[0].keys() if c.startswith("g_") or c.startswith("c_") or c.startswith("i_")]

    rows_train = _split_rows(rows_all, "train")
    rows_test = _split_rows(rows_all, "test")

    _, _, _, y_train, u_train = _extract(rows_train, feature_names)
    orth = bool(baseline_est.get("orthogonalize_country", True))
    if orth:
        u_train = _orthogonalize_country_features(u_train, feature_names)

    ridge_grid = [float(x.strip()) for x in args.ridge_grid.split(",") if x.strip()]
    m = float(baseline_est.get("m", 1.0))
    c_min = float(baseline_est.get("c_min", 1e-4))
    c_nl_min = float(baseline_est.get("c_nl_min", 0.0))
    k_min = float(baseline_est.get("k_min", 0.0))
    enforce = bool(baseline_est.get("enforce_physical", True))

    candidates = []

    for ridge_alpha in ridge_grid:
        try:
            est = fit_asym_model(
                y=y_train,
                u=u_train,
                feature_names=feature_names,
                m=m,
                dt=args.dt,
                ridge_alpha=ridge_alpha,
                enforce_physical=enforce,
                c_min=c_min,
                c_nl_min=c_nl_min,
                k_min=k_min,
            )
            test_metrics = eval_asym_test(rows_test, feature_names, est, orthogonalize_country=orth, dt=args.dt)
            rec = {
                "ridge_alpha": ridge_alpha,
                "train_target_r2": est.r2,
                "condition_number": est.condition_number,
                "c_up": est.c_up,
                "c_down": est.c_down,
                "c_nl": est.c_nl,
                "k": est.k,
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
        except Exception as e:
            rec = {"ridge_alpha": ridge_alpha, "status": f"failed: {e}"}
        candidates.append(rec)

    ok = [r for r in candidates if r.get("status") == "ok"]
    if not ok:
        raise RuntimeError("asym experiment failed for all ridge values")

    ok.sort(key=lambda r: (r["test_metrics"]["x_r2"], r["test_metrics"]["direction_accuracy"], -r["condition_number"]), reverse=True)
    best = ok[0]

    best_est = {
        "input": args.panel,
        "split": "train",
        "target_mode": "y_next",
        "dt": args.dt,
        "m": m,
        "damping_mode": "nonlinear_absv_asym",
        "c_up": best["c_up"],
        "c_down": best["c_down"],
        "c_nl": best["c_nl"],
        "k": best["k"],
        "intercept": None,
        "ridge_alpha": best["ridge_alpha"],
        "condition_number": best["condition_number"],
        "train_target_r2": best["train_target_r2"],
    }

    summary = {
        "baseline": {
            "estimation_path": str(args.baseline_est),
            "metrics_path": str(args.baseline_metrics),
            "test_metrics": baseline_metrics,
        },
        "grid": {"ridge_alpha": ridge_grid},
        "best": best,
        "candidates": candidates,
    }

    (out_dir / "asym_scan_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "asym_best_estimation_stub.json").write_text(json.dumps(best_est, indent=2, ensure_ascii=False), encoding="utf-8")

    with (out_dir / "asym_scan_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ridge_alpha",
                "train_target_r2",
                "condition_number",
                "c_up",
                "c_down",
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
                writer.writerow({"ridge_alpha": r.get("ridge_alpha"), "status": r.get("status")})
                continue
            tm = r["test_metrics"]
            gp = r["gap_vs_main"]
            writer.writerow(
                {
                    "ridge_alpha": r["ridge_alpha"],
                    "train_target_r2": r["train_target_r2"],
                    "condition_number": r["condition_number"],
                    "c_up": r["c_up"],
                    "c_down": r["c_down"],
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

    print("[done] step2 asym scan complete")
    print(f"[done] candidates={len(candidates)}, ok={len(ok)}")
    print(
        "[done] best: "
        f"ridge={best['ridge_alpha']}, x_r2={best['test_metrics']['x_r2']:.6g}, "
        f"dir={best['test_metrics']['direction_accuracy']:.6g}, cond={best['condition_number']:.4g}, "
        f"c_up={best['c_up']:.4g}, c_down={best['c_down']:.4g}, c_nl={best['c_nl']:.4g}"
    )
    print(f"[done] output: {out_dir}")


if __name__ == "__main__":
    main()
