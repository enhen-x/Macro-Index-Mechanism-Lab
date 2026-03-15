"""Run OLS estimation on US regression panel and save parameter summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from identification.ols_identifier import (
    identify_ols_from_regression_panel,
    identify_ols_y_next_from_regression_panel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run US OLS estimation from regression-ready panel.")
    parser.add_argument("--input", default="data/us/us_regression_panel.csv", help="Input regression panel CSV.")
    parser.add_argument("--split", choices=["train", "valid", "test", "all"], default="train")
    parser.add_argument(
        "--target-mode",
        choices=["a", "y_next"],
        default="y_next",
        help="Regression target: acceleration a(t) or one-step Y(t+1) equation.",
    )
    parser.add_argument("--dt", type=float, default=1.0, help="Time step for y_next target mode.")
    parser.add_argument("--m", type=float, default=1.0, help="Mass scale used for parameter mapping.")
    parser.add_argument(
        "--no-intercept",
        action="store_true",
        help="Disable intercept term in OLS equation.",
    )
    parser.add_argument(
        "--features",
        default="",
        help="Comma-separated feature columns. Default: auto-detect g_/c_/i_ columns.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=0.05,
        help="Ridge penalty strength (>=0).",
    )
    parser.add_argument(
        "--enforce-physical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enforce c>=c_min and k>=k_min through coefficient sign constraints.",
    )
    parser.add_argument(
        "--c-min",
        type=float,
        default=1e-4,
        help="Strict lower bound for damping c when enforce_physical is on.",
    )
    parser.add_argument(
        "--k-min",
        type=float,
        default=0.0,
        help="Lower bound for stiffness k when enforce_physical is on.",
    )
    parser.add_argument(
        "--damping-mode",
        choices=["linear", "nonlinear_absv"],
        default="nonlinear_absv",
        help="Damping form for y_next target mode.",
    )
    parser.add_argument(
        "--c-nl-min",
        type=float,
        default=0.0,
        help="Lower bound for nonlinear damping coefficient when damping-mode=nonlinear_absv.",
    )
    parser.add_argument(
        "--orthogonalize-country",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Residualize c_* features on g_* features before estimation.",
    )
    parser.add_argument(
        "--tail-weight-mode",
        choices=["none", "abs_power"],
        default="none",
        help="Tail weighting strategy for weighted regression.",
    )
    parser.add_argument(
        "--tail-weight-q",
        type=float,
        default=0.9,
        help="Tail threshold quantile based on |a|.",
    )
    parser.add_argument(
        "--tail-weight-scale",
        type=float,
        default=3.0,
        help="Additional weight scale on tail samples.",
    )
    parser.add_argument(
        "--tail-weight-power",
        type=float,
        default=1.0,
        help="Power for tail excess weighting.",
    )
    parser.add_argument(
        "--robust-mode",
        choices=["none", "huber"],
        default="none",
        help="Robust regression mode.",
    )
    parser.add_argument(
        "--robust-tuning",
        type=float,
        default=1.345,
        help="Huber tuning constant for robust_mode=huber.",
    )
    parser.add_argument(
        "--robust-max-iter",
        type=int,
        default=20,
        help="Maximum IRLS iterations for robust_mode=huber.",
    )
    parser.add_argument(
        "--robust-tol",
        type=float,
        default=1e-6,
        help="Convergence tolerance for robust_mode=huber.",
    )
    parser.add_argument(
        "--output",
        default="data/us/us_ols_estimation.json",
        help="Output JSON file for estimated parameters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    feature_cols = None
    if args.features.strip():
        feature_cols = [c.strip() for c in args.features.split(",") if c.strip()]
        if not feature_cols:
            raise ValueError("--features is provided but no valid column name found")

    common_kwargs = {
        "panel_path": args.input,
        "split": args.split,
        "feature_cols": feature_cols,
        "m": args.m,
        "fit_intercept": not args.no_intercept,
        "ridge_alpha": args.ridge_alpha,
        "enforce_physical": args.enforce_physical,
        "c_min": args.c_min,
        "k_min": args.k_min,
        "orthogonalize_country": args.orthogonalize_country,
        "tail_weight_mode": args.tail_weight_mode,
        "tail_weight_q": args.tail_weight_q,
        "tail_weight_scale": args.tail_weight_scale,
        "tail_weight_power": args.tail_weight_power,
        "robust_mode": args.robust_mode,
        "robust_tuning": args.robust_tuning,
        "robust_max_iter": args.robust_max_iter,
        "robust_tol": args.robust_tol,
    }
    if args.target_mode == "y_next":
        result = identify_ols_y_next_from_regression_panel(
            **common_kwargs,
            dt=args.dt,
            damping_mode=args.damping_mode,
            c_nl_min=args.c_nl_min,
        )
    else:
        result = identify_ols_from_regression_panel(**common_kwargs)

    payload = {
        "input": str(args.input),
        "split": args.split,
        "target_mode": args.target_mode,
        "dt": args.dt,
        "m": result.m,
        "c": result.c,
        "c_nl": getattr(result, "c_nl", 0.0),
        "k": result.k,
        "x0": result.x0,
        "intercept": result.intercept,
        "n_obs": result.n_obs,
        "r2": result.r2,
        "residual_std": result.residual_std,
        "ridge_alpha": result.ridge_alpha,
        "enforce_physical": result.enforce_physical,
        "orthogonalize_country": args.orthogonalize_country,
        "c_min": args.c_min,
        "k_min": args.k_min,
        "damping_mode": args.damping_mode if args.target_mode == "y_next" else "linear",
        "c_nl_min": args.c_nl_min,
        "tail_weight_mode": result.tail_weight_mode,
        "tail_weight_q": result.tail_weight_q,
        "tail_weight_scale": result.tail_weight_scale,
        "tail_weight_power": result.tail_weight_power,
        "robust_mode": result.robust_mode,
        "robust_tuning": result.robust_tuning,
        "robust_iterations": result.robust_iterations,
        "condition_number": result.condition_number,
        "optimization_method": result.optimization_method,
        "betas": result.betas,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[done] split={args.split}, target_mode={args.target_mode}, rows={result.n_obs}")
    print(
        f"[done] c={result.c:.6g}, c_nl={getattr(result, 'c_nl', 0.0):.6g}, k={result.k:.6g}, "
        f"x0={result.x0:.6g}, r2={result.r2:.6g}, cond={result.condition_number:.4g}"
    )
    print(
        f"[done] method={result.optimization_method}, ridge_alpha={result.ridge_alpha}, "
        f"tail={result.tail_weight_mode}(q={result.tail_weight_q},scale={result.tail_weight_scale},p={result.tail_weight_power}), "
        f"robust={result.robust_mode}(tuning={result.robust_tuning},iter={result.robust_iterations})"
    )
    print(f"[done] output: {output_path}")


if __name__ == "__main__":
    main()
