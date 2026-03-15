"""Run state-dependent local projections for macro->equity mechanism analysis.

Model (for each horizon h and each shock):
  y_{t,h} = a_h + b_h * shock_t + c_h * state_t + d_h * (shock_t*state_t) + G_h'controls_t + e_{t,h}

where y_{t,h} = log(P_{t+h}) - log(P_t).

Also estimates a baseline LP without state interaction for comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run state-dependent LP on US macro panel.")
    parser.add_argument(
        "--panel",
        default="stages/us_discount-rate vs cash-flow/data/us_macro_mechanism_panel.csv",
        help="Input mechanism panel CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="stages/us_discount-rate vs cash-flow/data/outputs/sd_lp",
        help="Output directory.",
    )
    parser.add_argument("--target-col", default="log_sp500", choices=["log_sp500"], help="LP target base series.")
    parser.add_argument("--state-col", default="state_high_infl", help="State indicator column (0/1).")
    parser.add_argument(
        "--shocks",
        default="shock_policy,shock_inflation,shock_growth,shock_labor",
        help="Comma-separated shock columns.",
    )
    parser.add_argument(
        "--controls",
        default="control_term_spread,control_vix_log,control_dxy_mom,control_infl_yoy,control_ip_mom",
        help="Comma-separated control columns.",
    )
    parser.add_argument("--h-min", type=int, default=1)
    parser.add_argument("--h-max", type=int, default=12)
    parser.add_argument("--nw-lag", type=int, default=3, help="Newey-West truncation lag.")
    parser.add_argument("--split", choices=["all", "train", "valid", "test"], default="all")
    parser.add_argument("--min-obs", type=int, default=60)
    return parser.parse_args()


def _to_float(v: str) -> float:
    t = (v or "").strip()
    if not t:
        return float("nan")
    try:
        return float(t)
    except ValueError:
        return float("nan")


def _parse_cols(raw: str) -> list[str]:
    out: list[str] = []
    for p in raw.split(","):
        c = p.strip()
        if c and c not in out:
            out.append(c)
    if not out:
        raise ValueError("empty column list")
    return out


def _read_panel(path: Path) -> list[dict[str, float | str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty panel: {path}")

    out: list[dict[str, float | str]] = []
    for r in rows:
        item: dict[str, float | str] = {"date": r.get("date", "")}
        for k, v in r.items():
            if k == "date":
                continue
            item[k] = _to_float(v)
        out.append(item)
    return out


def _nw_ols(X: np.ndarray, y: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    n, k = X.shape
    if n <= k:
        raise ValueError("insufficient observations for OLS")

    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    S = np.zeros((k, k), dtype=float)
    for t in range(n):
        xt = X[t : t + 1].T
        S += (resid[t] ** 2) * (xt @ xt.T)

    lag_eff = int(max(0, min(lag, n - 1)))
    for l in range(1, lag_eff + 1):
        w = 1.0 - l / (lag_eff + 1.0)
        G = np.zeros((k, k), dtype=float)
        for t in range(l, n):
            xt = X[t : t + 1].T
            xl = X[t - l : t - l + 1].T
            G += resid[t] * resid[t - l] * (xt @ xl.T + xl @ xt.T)
        S += w * G

    V = XtX_inv @ S @ XtX_inv
    diag = np.maximum(np.diag(V), 0.0)
    se = np.sqrt(diag)

    y_mean = float(np.mean(y))
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return beta, se, V, r2, lag_eff


def _pick_rows(rows: list[dict[str, float | str]], split: str) -> list[dict[str, float | str]]:
    if split == "all":
        return rows
    key = f"is_{split}"
    out = []
    for r in rows:
        v = r.get(key, float("nan"))
        if np.isfinite(v) and int(v) == 1:
            out.append(r)
    if not out:
        raise ValueError(f"no rows for split={split}")
    return out


def _build_arrays(rows: list[dict[str, float | str]], cols: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for c in cols:
        out[c] = np.array([float(r[c]) for r in rows], dtype=float)
    return out


def _se_linear_combo(c: np.ndarray, V: np.ndarray) -> float:
    v = float(c @ V @ c)
    return float(np.sqrt(max(v, 0.0)))


def main() -> None:
    args = parse_args()
    shocks = _parse_cols(args.shocks)
    controls = _parse_cols(args.controls)

    root = Path(__file__).resolve().parents[3]
    panel_path = (root / args.panel).resolve() if not Path(args.panel).is_absolute() else Path(args.panel)
    out_dir = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_all = _read_panel(panel_path)
    rows = _pick_rows(rows_all, split=args.split)

    required = ["date", args.target_col, args.state_col] + shocks + controls
    arr = _build_arrays(rows, [c for c in required if c != "date"])

    valid = np.ones(len(rows), dtype=bool)
    for c in [args.target_col, args.state_col] + shocks + controls:
        valid &= np.isfinite(arr[c])

    date = [str(r["date"]) for r in rows]
    logp = arr[args.target_col]
    state = arr[args.state_col]

    records: list[dict[str, float | int | str]] = []

    for shock in shocks:
        x_shock = arr[shock]
        x_ctrl = np.column_stack([arr[c] for c in controls]) if controls else np.empty((len(rows), 0), dtype=float)

        for h in range(args.h_min, args.h_max + 1):
            if h <= 0:
                continue
            idx = np.arange(0, len(rows) - h, dtype=int)
            if len(idx) <= args.min_obs:
                continue

            y_h = logp[idx + h] - logp[idx]
            s_h = state[idx]
            z_h = x_shock[idx]
            c_h = x_ctrl[idx] if x_ctrl.size > 0 else np.empty((len(idx), 0), dtype=float)
            mask = valid[idx] & np.isfinite(y_h)
            mask &= np.isfinite(s_h) & np.isfinite(z_h)
            if c_h.size > 0:
                mask &= np.all(np.isfinite(c_h), axis=1)

            y = y_h[mask]
            s = s_h[mask]
            z = z_h[mask]
            c = c_h[mask] if c_h.size > 0 else np.empty((np.sum(mask), 0), dtype=float)
            n = len(y)
            if n <= max(args.min_obs, 10):
                continue

            X_base = np.column_stack([np.ones(n), z, c])
            b_base, se_base, V_base, r2_base, lag_used_base = _nw_ols(X_base, y, lag=args.nw_lag)

            X_sd = np.column_stack([np.ones(n), z, s, z * s, c])
            b_sd, se_sd, V_sd, r2_sd, lag_used_sd = _nw_ols(X_sd, y, lag=args.nw_lag)

            beta0 = float(b_sd[1])
            delta = float(b_sd[3])
            beta1 = beta0 + delta
            se_beta0 = float(se_sd[1])
            se_delta = float(se_sd[3])
            comb = np.zeros(X_sd.shape[1], dtype=float)
            comb[1] = 1.0
            comb[3] = 1.0
            se_beta1 = _se_linear_combo(comb, V_sd)

            rec = {
                "shock": shock,
                "horizon": int(h),
                "n_obs": int(n),
                "avg_state": float(np.mean(s)),
                "beta_base": float(b_base[1]),
                "se_base": float(se_base[1]),
                "t_base": float(b_base[1] / se_base[1]) if se_base[1] > 0 else float("nan"),
                "r2_base": float(r2_base),
                "beta_state0": beta0,
                "se_state0": se_beta0,
                "t_state0": float(beta0 / se_beta0) if se_beta0 > 0 else float("nan"),
                "beta_interaction": delta,
                "se_interaction": se_delta,
                "t_interaction": float(delta / se_delta) if se_delta > 0 else float("nan"),
                "beta_state1": beta1,
                "se_state1": float(se_beta1),
                "t_state1": float(beta1 / se_beta1) if se_beta1 > 0 else float("nan"),
                "r2_sd": float(r2_sd),
                "nw_lag_used_base": int(lag_used_base),
                "nw_lag_used_sd": int(lag_used_sd),
            }
            records.append(rec)

    if not records:
        raise RuntimeError("no LP results generated; check panel/split/min-obs")

    coef_csv = out_dir / "sd_lp_coefficients.csv"
    fields = [
        "shock",
        "horizon",
        "n_obs",
        "avg_state",
        "beta_base",
        "se_base",
        "t_base",
        "r2_base",
        "beta_state0",
        "se_state0",
        "t_state0",
        "beta_interaction",
        "se_interaction",
        "t_interaction",
        "beta_state1",
        "se_state1",
        "t_state1",
        "r2_sd",
        "nw_lag_used_base",
        "nw_lag_used_sd",
    ]
    with coef_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    by_shock: dict[str, dict[str, float | int]] = {}
    for s in shocks:
        rs = [r for r in records if r["shock"] == s]
        rs_sorted = sorted(rs, key=lambda x: int(x["horizon"]))
        last = rs_sorted[-1]
        by_shock[s] = {
            "n_horizons": int(len(rs_sorted)),
            "state0_hmax": float(last["beta_state0"]),
            "state1_hmax": float(last["beta_state1"]),
            "interaction_hmax": float(last["beta_interaction"]),
            "avg_abs_t_interaction": float(np.mean([abs(float(r["t_interaction"])) for r in rs_sorted])),
            "avg_r2_sd": float(np.mean([float(r["r2_sd"]) for r in rs_sorted])),
            "avg_r2_base": float(np.mean([float(r["r2_base"]) for r in rs_sorted])),
        }

    summary = {
        "panel": str(panel_path),
        "split": args.split,
        "state_col": args.state_col,
        "shocks": shocks,
        "controls": controls,
        "h_min": args.h_min,
        "h_max": args.h_max,
        "nw_lag": args.nw_lag,
        "records": len(records),
        "by_shock": by_shock,
        "outputs": {
            "coefficients_csv": str(coef_csv),
        },
    }
    summary_path = out_dir / "sd_lp_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[done] coefficients: {coef_csv}")
    print(f"[done] summary: {summary_path}")
    print(f"[done] records: {len(records)}")


if __name__ == "__main__":
    main()
