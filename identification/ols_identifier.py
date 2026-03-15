"""Baseline OLS identifiers for forced damped system."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EstimatedParams:
    """Legacy single-input baseline result."""

    m: float
    c: float
    k: float
    x0: float
    beta_u: float


@dataclass
class PanelEstimatedParams:
    """Panel-style OLS result from pre-built regression columns."""

    m: float
    c: float
    k: float
    x0: float
    intercept: float
    betas: dict[str, float]
    n_obs: int
    r2: float
    residual_std: float
    ridge_alpha: float
    enforce_physical: bool
    condition_number: float
    optimization_method: str
    tail_weight_mode: str
    tail_weight_q: float
    tail_weight_scale: float
    tail_weight_power: float
    robust_mode: str
    robust_tuning: float
    robust_iterations: int
    c_nl: float = 0.0


def identify_ols(x: np.ndarray, u: np.ndarray, dt: float = 1.0) -> EstimatedParams:
    """Legacy baseline: estimate from raw x, u using numerical gradients."""
    if len(x) != len(u):
        raise ValueError("x and u must have same length")
    if len(x) < 5:
        raise ValueError("need at least 5 samples")

    v = np.gradient(x, dt)
    a = np.gradient(v, dt)
    x0 = float(np.mean(x))

    x_design = np.column_stack([v, x - x0, u])
    coef, *_ = np.linalg.lstsq(x_design, a, rcond=None)

    b1, b2, b3 = coef
    m = 1.0
    c = float(-b1 * m)
    k = float(-b2 * m)
    beta_u = float(b3 * m)

    return EstimatedParams(m=m, c=c, k=k, x0=x0, beta_u=beta_u)


def _to_float(value: str) -> float:
    text = (value or "").strip()
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    mean_y = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - mean_y) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _ridge_closed_form(x_design: np.ndarray, y: np.ndarray, ridge_alpha: float, penalty_mask: np.ndarray) -> np.ndarray:
    xtx = x_design.T @ x_design
    xty = x_design.T @ y
    reg = ridge_alpha * np.diag(penalty_mask)
    try:
        return np.linalg.solve(xtx + reg, xty)
    except np.linalg.LinAlgError:
        coef, *_ = np.linalg.lstsq(xtx + reg, xty, rcond=None)
        return coef


def _solve_bounded_ridge(
    x_design: np.ndarray,
    y: np.ndarray,
    ridge_alpha: float,
    penalty_mask: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, int]:
    coef = _ridge_closed_form(x_design, y, ridge_alpha=ridge_alpha, penalty_mask=penalty_mask)
    coef = np.minimum(np.maximum(coef, lower), upper)

    lipschitz = float(np.linalg.norm(x_design, ord=2) ** 2 + ridge_alpha * float(np.max(penalty_mask)))
    if not np.isfinite(lipschitz) or lipschitz <= 0:
        return coef, 0

    step = 1.0 / lipschitz
    for it in range(1, max_iter + 1):
        grad = x_design.T @ (x_design @ coef - y) + ridge_alpha * (penalty_mask * coef)
        nxt = coef - step * grad
        nxt = np.minimum(np.maximum(nxt, lower), upper)
        diff = float(np.linalg.norm(nxt - coef))
        base = 1.0 + float(np.linalg.norm(coef))
        coef = nxt
        if diff <= tol * base:
            return coef, it
    return coef, max_iter


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


def _build_tail_weights(
    a: np.ndarray,
    mode: str,
    q: float,
    scale: float,
    power: float,
) -> np.ndarray:
    if mode == "none":
        return np.ones_like(a, dtype=float)
    if mode != "abs_power":
        raise ValueError("tail_weight_mode must be one of: none, abs_power")
    if not (0.0 < q < 1.0):
        raise ValueError("tail_weight_q must be in (0,1)")
    if scale < 0:
        raise ValueError("tail_weight_scale must be >= 0")
    if power <= 0:
        raise ValueError("tail_weight_power must be > 0")

    out = np.full_like(a, np.nan, dtype=float)
    finite = np.isfinite(a)
    if not np.any(finite):
        return out

    abs_a = np.abs(a[finite])
    th = float(np.quantile(abs_a, q))
    max_abs = float(np.max(abs_a))
    denom = max(max_abs - th, 1e-12)
    ex = np.clip((abs_a - th) / denom, 0.0, None)
    w = 1.0 + scale * np.power(ex, power)

    out[finite] = w
    return out


def _robust_scale_mad(resid: np.ndarray) -> float:
    finite = np.isfinite(resid)
    if not np.any(finite):
        return 1.0
    v = resid[finite]
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    scale = 1.4826 * mad
    if (not np.isfinite(scale)) or scale <= 1e-12:
        sd = float(np.std(v))
        scale = sd if np.isfinite(sd) and sd > 1e-12 else 1.0
    return scale


def _huber_weights(resid: np.ndarray, scale: float, tuning: float) -> np.ndarray:
    if scale <= 0 or tuning <= 0:
        raise ValueError("scale and tuning must be > 0 for huber weights")
    u = np.abs(resid) / (tuning * scale)
    out = np.ones_like(resid, dtype=float)
    mask = u > 1.0
    out[mask] = 1.0 / u[mask]
    return out


def identify_ols_panel(
    *,
    a: np.ndarray,
    v: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    feature_names: list[str],
    m: float = 1.0,
    fit_intercept: bool = True,
    ridge_alpha: float = 0.0,
    enforce_physical: bool = False,
    c_min: float = 0.0,
    k_min: float = 0.0,
    sample_weight: np.ndarray | None = None,
    max_iter: int = 20000,
    tol: float = 1e-9,
    tail_weight_mode: str = "none",
    tail_weight_q: float = 0.9,
    tail_weight_scale: float = 0.0,
    tail_weight_power: float = 1.0,
    robust_mode: str = "none",
    robust_tuning: float = 1.345,
    robust_max_iter: int = 20,
    robust_tol: float = 1e-6,
) -> PanelEstimatedParams:
    """Estimate params from aligned regression columns."""
    if m <= 0:
        raise ValueError("m must be > 0")
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha must be >= 0")
    if c_min < 0 or k_min < 0:
        raise ValueError("c_min and k_min must be >= 0")
    if u.ndim != 2:
        raise ValueError("u must be a 2D array with shape (n_obs, n_features)")

    n = len(a)
    if len(v) != n or len(y) != n or u.shape[0] != n:
        raise ValueError("a, v, y, u must have aligned length")
    if len(feature_names) != u.shape[1]:
        raise ValueError("feature_names length must match u.shape[1]")
    if n < 5:
        raise ValueError("need at least 5 observations")
    if robust_mode not in {"none", "huber"}:
        raise ValueError("robust_mode must be one of: none, huber")
    if robust_tuning <= 0:
        raise ValueError("robust_tuning must be > 0")
    if robust_max_iter < 1:
        raise ValueError("robust_max_iter must be >= 1")
    if robust_tol <= 0:
        raise ValueError("robust_tol must be > 0")

    if sample_weight is None:
        sample_weight = np.ones(n, dtype=float)
    if len(sample_weight) != n:
        raise ValueError("sample_weight length must equal number of observations")

    finite_mask = (
        np.isfinite(a)
        & np.isfinite(v)
        & np.isfinite(y)
        & np.all(np.isfinite(u), axis=1)
        & np.isfinite(sample_weight)
        & (sample_weight > 0)
    )
    a_fit = a[finite_mask]
    v_fit = v[finite_mask]
    y_fit = y[finite_mask]
    u_fit = u[finite_mask]
    w_fit = sample_weight[finite_mask]

    if len(a_fit) < 5:
        raise ValueError("not enough finite observations after filtering")

    cols = []
    if fit_intercept:
        cols.append(np.ones_like(a_fit))
    cols.extend([v_fit, y_fit])
    if u_fit.shape[1] > 0:
        cols.append(u_fit)

    x_design = np.column_stack(cols)
    p = x_design.shape[1]

    penalty_mask = np.ones(p, dtype=float)
    offset = 0
    if fit_intercept:
        penalty_mask[0] = 0.0
        offset = 1

    lower = np.full(p, -np.inf, dtype=float)
    upper = np.full(p, np.inf, dtype=float)
    if enforce_physical:
        upper[offset] = -float(c_min) / float(m)
        upper[offset + 1] = -float(k_min) / float(m)

    def solve_given_weight(weight: np.ndarray) -> tuple[np.ndarray, str, np.ndarray]:
        sqrt_w = np.sqrt(weight)
        x_opt_local = x_design * sqrt_w[:, None]
        y_opt_local = a_fit * sqrt_w

        if not enforce_physical and ridge_alpha == 0:
            coef_local, *_ = np.linalg.lstsq(x_opt_local, y_opt_local, rcond=None)
            method_local = "wls_lstsq"
        elif not enforce_physical and ridge_alpha > 0:
            coef_local = _ridge_closed_form(
                x_opt_local,
                y_opt_local,
                ridge_alpha=ridge_alpha,
                penalty_mask=penalty_mask,
            )
            method_local = "weighted_ridge_closed_form"
        else:
            coef_local, _ = _solve_bounded_ridge(
                x_opt_local,
                y_opt_local,
                ridge_alpha=ridge_alpha,
                penalty_mask=penalty_mask,
                lower=lower,
                upper=upper,
                max_iter=max_iter,
                tol=tol,
            )
            method_local = "bounded_projected_gradient"

        return np.asarray(coef_local, dtype=float), method_local, x_opt_local

    if robust_mode == "none":
        coef, base_method, x_opt = solve_given_weight(w_fit)
        robust_iterations = 0
        optimization_method = base_method
    else:
        total_weight = w_fit.copy()
        coef, base_method, x_opt = solve_given_weight(total_weight)
        robust_iterations = 0
        for it in range(1, robust_max_iter + 1):
            resid_now = a_fit - (x_design @ coef)
            scale = _robust_scale_mad(resid_now)
            w_robust = _huber_weights(resid_now, scale=scale, tuning=robust_tuning)
            new_weight = total_weight * w_robust

            coef_new, _, x_opt_new = solve_given_weight(new_weight)
            diff = float(np.linalg.norm(coef_new - coef))
            base = 1.0 + float(np.linalg.norm(coef))

            coef = coef_new
            x_opt = x_opt_new
            total_weight = new_weight
            robust_iterations = it
            if diff <= robust_tol * base:
                break

        optimization_method = f"irls_huber+{base_method}"

    y_hat = x_design @ coef
    resid = a_fit - y_hat

    if fit_intercept:
        intercept = float(coef[0])
    else:
        intercept = 0.0

    b_v = float(coef[offset])
    b_y = float(coef[offset + 1])
    beta_vec = np.array(coef[offset + 2 :], dtype=float) if u_fit.shape[1] > 0 else np.array([], dtype=float)

    c = float(-b_v * m)
    k = float(-b_y * m)
    if enforce_physical:
        c = max(float(c_min), c)
        k = max(float(k_min), k)

    if fit_intercept and abs(k) > 1e-12:
        x0 = float(intercept / k)
    else:
        x0 = 0.0

    betas = {name: float(beta * m) for name, beta in zip(feature_names, beta_vec)}
    dof = max(1, len(a_fit) - p)
    residual_std = float(np.sqrt(np.sum(resid * resid) / dof))
    r2 = _r2_score(a_fit, y_hat)
    condition_number = float(np.linalg.cond(x_opt))

    return PanelEstimatedParams(
        m=float(m),
        c=c,
        k=k,
        x0=x0,
        intercept=intercept,
        betas=betas,
        n_obs=int(len(a_fit)),
        r2=r2,
        residual_std=residual_std,
        ridge_alpha=float(ridge_alpha),
        enforce_physical=bool(enforce_physical),
        condition_number=condition_number,
        optimization_method=optimization_method,
        tail_weight_mode=tail_weight_mode,
        tail_weight_q=float(tail_weight_q),
        tail_weight_scale=float(tail_weight_scale),
        tail_weight_power=float(tail_weight_power),
        robust_mode=robust_mode,
        robust_tuning=float(robust_tuning),
        robust_iterations=int(robust_iterations),
    )


def identify_ols_from_regression_panel(
    panel_path: str | Path,
    *,
    split: str = "train",
    feature_cols: list[str] | None = None,
    m: float = 1.0,
    fit_intercept: bool = True,
    ridge_alpha: float = 0.0,
    enforce_physical: bool = False,
    c_min: float = 0.0,
    k_min: float = 0.0,
    orthogonalize_country: bool = False,
    tail_weight_mode: str = "none",
    tail_weight_q: float = 0.9,
    tail_weight_scale: float = 0.0,
    tail_weight_power: float = 1.0,
    robust_mode: str = "none",
    robust_tuning: float = 1.345,
    robust_max_iter: int = 20,
    robust_tol: float = 1e-6,
) -> PanelEstimatedParams:
    """Estimate model from regression-ready CSV produced by scripts/us builder."""
    split_norm = split.strip().lower()
    if split_norm not in {"train", "valid", "test", "all"}:
        raise ValueError("split must be one of: train, valid, test, all")

    path = Path(panel_path)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty panel file: {path}")

    if feature_cols is None:
        feature_cols = [
            c
            for c in rows[0].keys()
            if c.startswith("g_") or c.startswith("c_") or c.startswith("i_")
        ]

    required_cols = {"a", "v", "Y", *feature_cols}
    missing = [c for c in required_cols if c not in rows[0]]
    if missing:
        raise ValueError(f"missing required columns in panel: {missing}")

    if split_norm == "all":
        selected = rows
    else:
        split_col = f"is_{split_norm}"
        if split_col not in rows[0]:
            raise ValueError(f"split column not found in panel: {split_col}")
        selected = [r for r in rows if (r.get(split_col) or "").strip() == "1"]

    if len(selected) < 5:
        raise ValueError(f"not enough rows for split={split_norm}; got {len(selected)}")

    a = np.array([_to_float(r["a"]) for r in selected], dtype=float)
    v = np.array([_to_float(r["v"]) for r in selected], dtype=float)
    y = np.array([_to_float(r["Y"]) for r in selected], dtype=float)
    u = np.array([[_to_float(r[c]) for c in feature_cols] for r in selected], dtype=float)

    if orthogonalize_country:
        u = _orthogonalize_country_features(u, feature_cols)

    sample_weight = _build_tail_weights(
        a,
        mode=tail_weight_mode,
        q=tail_weight_q,
        scale=tail_weight_scale,
        power=tail_weight_power,
    )

    return identify_ols_panel(
        a=a,
        v=v,
        y=y,
        u=u,
        feature_names=feature_cols,
        m=m,
        fit_intercept=fit_intercept,
        ridge_alpha=ridge_alpha,
        enforce_physical=enforce_physical,
        c_min=c_min,
        k_min=k_min,
        sample_weight=sample_weight,
        tail_weight_mode=tail_weight_mode,
        tail_weight_q=tail_weight_q,
        tail_weight_scale=tail_weight_scale,
        tail_weight_power=tail_weight_power,
        robust_mode=robust_mode,
        robust_tuning=robust_tuning,
        robust_max_iter=robust_max_iter,
        robust_tol=robust_tol,
    )



def identify_ols_y_next_panel(
    *,
    y_prev: np.ndarray,
    y_cur: np.ndarray,
    y_next: np.ndarray,
    u_cur: np.ndarray,
    feature_names: list[str],
    a_for_weight: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    m: float = 1.0,
    dt: float = 1.0,
    fit_intercept: bool = True,
    ridge_alpha: float = 0.0,
    enforce_physical: bool = False,
    c_min: float = 0.0,
    k_min: float = 0.0,
    damping_mode: str = "linear",
    c_nl_min: float = 0.0,
    max_iter: int = 20000,
    tol: float = 1e-9,
    tail_weight_mode: str = "none",
    tail_weight_q: float = 0.9,
    tail_weight_scale: float = 0.0,
    tail_weight_power: float = 1.0,
    robust_mode: str = "none",
    robust_tuning: float = 1.345,
    robust_max_iter: int = 20,
    robust_tol: float = 1e-6,
) -> PanelEstimatedParams:
    """Estimate params by fitting one-step closed-form equation on Y(t+1)."""
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if m <= 0:
        raise ValueError("m must be > 0")
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha must be >= 0")
    if c_min < 0 or k_min < 0 or c_nl_min < 0:
        raise ValueError("c_min, c_nl_min and k_min must be >= 0")
    if robust_mode not in {"none", "huber"}:
        raise ValueError("robust_mode must be one of: none, huber")
    if robust_tuning <= 0:
        raise ValueError("robust_tuning must be > 0")
    if robust_max_iter < 1:
        raise ValueError("robust_max_iter must be >= 1")
    if robust_tol <= 0:
        raise ValueError("robust_tol must be > 0")
    if damping_mode not in {"linear", "nonlinear_absv"}:
        raise ValueError("damping_mode must be one of: linear, nonlinear_absv")

    y_prev = np.asarray(y_prev, dtype=float)
    y_cur = np.asarray(y_cur, dtype=float)
    y_next = np.asarray(y_next, dtype=float)
    u_cur = np.asarray(u_cur, dtype=float)
    n = len(y_cur)

    if len(y_prev) != n or len(y_next) != n:
        raise ValueError("y_prev, y_cur, y_next must have aligned length")
    if u_cur.ndim != 2 or u_cur.shape[0] != n:
        raise ValueError("u_cur must be a 2D array with shape (n_obs, n_features)")
    if len(feature_names) != u_cur.shape[1]:
        raise ValueError("feature_names length must match u_cur.shape[1]")

    if a_for_weight is not None:
        a_for_weight = np.asarray(a_for_weight, dtype=float)
        if len(a_for_weight) != n:
            raise ValueError("a_for_weight length must align with y_cur")

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if len(sample_weight) != n:
            raise ValueError("sample_weight length must align with y_cur")
    elif a_for_weight is not None:
        sample_weight = _build_tail_weights(
            a_for_weight,
            mode=tail_weight_mode,
            q=tail_weight_q,
            scale=tail_weight_scale,
            power=tail_weight_power,
        )
    else:
        sample_weight = np.ones(n, dtype=float)

    finite = (
        np.isfinite(y_prev)
        & np.isfinite(y_cur)
        & np.isfinite(y_next)
        & np.all(np.isfinite(u_cur), axis=1)
        & np.isfinite(sample_weight)
        & (sample_weight > 0)
    )
    y_prev = y_prev[finite]
    y_cur = y_cur[finite]
    y_next = y_next[finite]
    u_cur = u_cur[finite]
    w_cur = sample_weight[finite]

    if len(y_cur) < 5:
        raise ValueError("not enough finite one-step rows after filtering")

    dt2 = dt * dt
    target = y_next - 2.0 * y_cur + y_prev

    cols = []
    if fit_intercept:
        cols.append(np.full_like(target, dt2))

    if damping_mode == "nonlinear_absv":
        v_proxy = (y_cur - y_prev) / dt
        cols.extend([-(dt2) * v_proxy, -(dt2) * np.abs(v_proxy) * v_proxy, -(dt2) * y_cur])
        if u_cur.shape[1] > 0:
            cols.append(dt2 * u_cur)
        x_design = np.column_stack(cols)
        p = x_design.shape[1]

        penalty_mask = np.ones(p, dtype=float)
        offset = 0
        if fit_intercept:
            penalty_mask[0] = 0.0
            offset = 1

        lower = np.full(p, -np.inf, dtype=float)
        upper = np.full(p, np.inf, dtype=float)
        if enforce_physical:
            lower[offset] = float(c_min) / float(m)
            lower[offset + 1] = float(c_nl_min) / float(m)
            lower[offset + 2] = float(k_min) / float(m)

        idx_kappa = offset + 2
        beta_start = offset + 3
    else:
        cols.extend([y_prev - y_next, -(dt2) * y_cur])
        if u_cur.shape[1] > 0:
            cols.append(dt2 * u_cur)
        x_design = np.column_stack(cols)
        p = x_design.shape[1]

        penalty_mask = np.ones(p, dtype=float)
        offset = 0
        if fit_intercept:
            penalty_mask[0] = 0.0
            offset = 1

        lower = np.full(p, -np.inf, dtype=float)
        upper = np.full(p, np.inf, dtype=float)
        if enforce_physical:
            lower[offset] = float(c_min) * float(dt) / (2.0 * float(m))
            lower[offset + 1] = float(k_min) / float(m)

        idx_kappa = offset + 1
        beta_start = offset + 2

    def solve_given_weight(weight: np.ndarray) -> tuple[np.ndarray, str, np.ndarray]:
        sqrt_w = np.sqrt(weight)
        x_opt_local = x_design * sqrt_w[:, None]
        y_opt_local = target * sqrt_w

        if not enforce_physical and ridge_alpha == 0:
            coef_local, *_ = np.linalg.lstsq(x_opt_local, y_opt_local, rcond=None)
            method_local = "wls_lstsq_y_next"
        elif not enforce_physical and ridge_alpha > 0:
            coef_local = _ridge_closed_form(
                x_opt_local,
                y_opt_local,
                ridge_alpha=ridge_alpha,
                penalty_mask=penalty_mask,
            )
            method_local = "weighted_ridge_closed_form_y_next"
        else:
            coef_local, _ = _solve_bounded_ridge(
                x_opt_local,
                y_opt_local,
                ridge_alpha=ridge_alpha,
                penalty_mask=penalty_mask,
                lower=lower,
                upper=upper,
                max_iter=max_iter,
                tol=tol,
            )
            method_local = "bounded_projected_gradient_y_next"

        return np.asarray(coef_local, dtype=float), method_local, x_opt_local

    if robust_mode == "none":
        coef, base_method, x_opt = solve_given_weight(w_cur)
        robust_iterations = 0
        optimization_method = base_method
    else:
        total_weight = w_cur.copy()
        coef, base_method, x_opt = solve_given_weight(total_weight)
        robust_iterations = 0
        for it in range(1, robust_max_iter + 1):
            resid_now = target - (x_design @ coef)
            scale = _robust_scale_mad(resid_now)
            w_robust = _huber_weights(resid_now, scale=scale, tuning=robust_tuning)
            new_weight = total_weight * w_robust

            coef_new, _, x_opt_new = solve_given_weight(new_weight)
            diff = float(np.linalg.norm(coef_new - coef))
            base = 1.0 + float(np.linalg.norm(coef))

            coef = coef_new
            x_opt = x_opt_new
            total_weight = new_weight
            robust_iterations = it
            if diff <= robust_tol * base:
                break
        optimization_method = f"irls_huber+{base_method}"

    y_hat = x_design @ coef
    resid = target - y_hat

    gamma0 = float(coef[0]) if fit_intercept else 0.0
    kappa = float(coef[idx_kappa])
    beta_vec = np.array(coef[beta_start:], dtype=float) if u_cur.shape[1] > 0 else np.array([], dtype=float)

    if damping_mode == "nonlinear_absv":
        theta_v = float(coef[offset])
        theta_v2 = float(coef[offset + 1])
        c = float(m) * theta_v
        c_nl = float(m) * theta_v2
    else:
        lam = float(coef[offset])
        c = 2.0 * float(m) * lam / float(dt)
        c_nl = 0.0

    k = float(m) * kappa
    if enforce_physical:
        c = max(float(c_min), c)
        c_nl = max(float(c_nl_min), c_nl)
        k = max(float(k_min), k)

    x0 = float(gamma0 / kappa) if fit_intercept and abs(kappa) > 1e-12 else 0.0
    intercept = gamma0
    betas = {name: float(beta * m) for name, beta in zip(feature_names, beta_vec)}

    dof = max(1, len(target) - p)
    residual_std = float(np.sqrt(np.sum(resid * resid) / dof))
    r2 = _r2_score(target, y_hat)
    condition_number = float(np.linalg.cond(x_opt))

    return PanelEstimatedParams(
        m=float(m),
        c=c,
        k=k,
        x0=x0,
        intercept=intercept,
        betas=betas,
        n_obs=int(len(target)),
        r2=r2,
        residual_std=residual_std,
        ridge_alpha=float(ridge_alpha),
        enforce_physical=bool(enforce_physical),
        condition_number=condition_number,
        optimization_method=optimization_method,
        tail_weight_mode=tail_weight_mode,
        tail_weight_q=float(tail_weight_q),
        tail_weight_scale=float(tail_weight_scale),
        tail_weight_power=float(tail_weight_power),
        robust_mode=robust_mode,
        robust_tuning=float(robust_tuning),
        robust_iterations=int(robust_iterations),
        c_nl=float(c_nl),
    )


def identify_ols_y_next_from_regression_panel(
    panel_path: str | Path,
    *,
    split: str = "train",
    feature_cols: list[str] | None = None,
    m: float = 1.0,
    dt: float = 1.0,
    fit_intercept: bool = True,
    ridge_alpha: float = 0.0,
    enforce_physical: bool = False,
    c_min: float = 0.0,
    k_min: float = 0.0,
    orthogonalize_country: bool = False,
    damping_mode: str = "linear",
    c_nl_min: float = 0.0,
    tail_weight_mode: str = "none",
    tail_weight_q: float = 0.9,
    tail_weight_scale: float = 0.0,
    tail_weight_power: float = 1.0,
    robust_mode: str = "none",
    robust_tuning: float = 1.345,
    robust_max_iter: int = 20,
    robust_tol: float = 1e-6,
    max_iter: int = 20000,
    tol: float = 1e-9,
) -> PanelEstimatedParams:
    """Estimate params by fitting one-step closed-form equation on Y(t+1)."""
    split_norm = split.strip().lower()
    if split_norm not in {"train", "valid", "test", "all"}:
        raise ValueError("split must be one of: train, valid, test, all")

    path_obj = Path(panel_path)
    with path_obj.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty panel file: {path_obj}")

    if feature_cols is None:
        feature_cols = [
            c
            for c in rows[0].keys()
            if c.startswith("g_") or c.startswith("c_") or c.startswith("i_")
        ]

    required_cols = {"a", "Y", *feature_cols}
    missing = [c for c in required_cols if c not in rows[0]]
    if missing:
        raise ValueError(f"missing required columns in panel: {missing}")

    if split_norm == "all":
        selected = rows
    else:
        split_col = f"is_{split_norm}"
        if split_col not in rows[0]:
            raise ValueError(f"split column not found in panel: {split_col}")
        selected = [r for r in rows if (r.get(split_col) or "").strip() == "1"]

    if len(selected) < 7:
        raise ValueError(f"not enough rows for split={split_norm}; got {len(selected)}")

    a = np.array([_to_float(r["a"]) for r in selected], dtype=float)
    y = np.array([_to_float(r["Y"]) for r in selected], dtype=float)
    u = np.array([[_to_float(r[c]) for c in feature_cols] for r in selected], dtype=float)

    if orthogonalize_country:
        u = _orthogonalize_country_features(u, feature_cols)

    base_weight = _build_tail_weights(
        a,
        mode=tail_weight_mode,
        q=tail_weight_q,
        scale=tail_weight_scale,
        power=tail_weight_power,
    )

    idx = np.arange(1, len(y) - 1)
    return identify_ols_y_next_panel(
        y_prev=y[idx - 1],
        y_cur=y[idx],
        y_next=y[idx + 1],
        u_cur=u[idx],
        feature_names=feature_cols,
        sample_weight=base_weight[idx],
        m=m,
        dt=dt,
        fit_intercept=fit_intercept,
        ridge_alpha=ridge_alpha,
        enforce_physical=enforce_physical,
        c_min=c_min,
        k_min=k_min,
        damping_mode=damping_mode,
        c_nl_min=c_nl_min,
        max_iter=max_iter,
        tol=tol,
        tail_weight_mode=tail_weight_mode,
        tail_weight_q=tail_weight_q,
        tail_weight_scale=tail_weight_scale,
        tail_weight_power=tail_weight_power,
        robust_mode=robust_mode,
        robust_tuning=robust_tuning,
        robust_max_iter=robust_max_iter,
        robust_tol=robust_tol,
    )
