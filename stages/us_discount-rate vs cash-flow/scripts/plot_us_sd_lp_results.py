"""Plot SD-LP outputs for interpretation.

Generates:
- Per-shock IRF charts (state0 vs state1 with 95% CI)
- Per-shock interaction charts (beta_interaction with CI)
- Overview heatmaps for interaction beta and interaction t-stat
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SD-LP regression outputs.")
    parser.add_argument(
        "--coeff-csv",
        default="stages/us_discount-rate vs cash-flow/data/outputs/sd_lp/sd_lp_coefficients.csv",
        help="Input coefficient CSV from run_us_sd_lp.py",
    )
    parser.add_argument(
        "--summary-json",
        default="stages/us_discount-rate vs cash-flow/data/outputs/sd_lp/sd_lp_summary.json",
        help="Input summary JSON from run_us_sd_lp.py",
    )
    parser.add_argument(
        "--output-dir",
        default="stages/us_discount-rate vs cash-flow/data/outputs/sd_lp/plots",
        help="Output plot directory.",
    )
    parser.add_argument("--ci-z", type=float, default=1.96, help="Z score for confidence bands.")
    parser.add_argument("--sig-t", type=float, default=1.96, help="Significance threshold on |t|.")
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args()


def _to_float(v: str) -> float:
    t = (v or "").strip()
    if not t:
        return float("nan")
    try:
        return float(t)
    except ValueError:
        return float("nan")


def _read_coeff(path: Path) -> dict[str, list[dict[str, float]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty coeff csv: {path}")

    by_shock: dict[str, list[dict[str, float]]] = {}
    for r in rows:
        item = {k: _to_float(v) for k, v in r.items() if k != "shock"}
        s = r["shock"]
        by_shock.setdefault(s, []).append(item)

    for s in by_shock:
        by_shock[s] = sorted(by_shock[s], key=lambda x: int(x["horizon"]))
    return by_shock


def _read_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_per_shock(shock: str, rows: list[dict[str, float]], out_dir: Path, ci_z: float, sig_t: float, dpi: int) -> dict:
    h = np.array([int(r["horizon"]) for r in rows], dtype=int)
    b0 = np.array([r["beta_state0"] for r in rows], dtype=float)
    s0 = np.array([r["se_state0"] for r in rows], dtype=float)
    b1 = np.array([r["beta_state1"] for r in rows], dtype=float)
    s1 = np.array([r["se_state1"] for r in rows], dtype=float)
    bi = np.array([r["beta_interaction"] for r in rows], dtype=float)
    si = np.array([r["se_interaction"] for r in rows], dtype=float)
    ti = np.array([r["t_interaction"] for r in rows], dtype=float)

    sig_mask = np.abs(ti) >= sig_t
    sig_h = h[sig_mask].tolist()

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax = axes[0]
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.plot(h, b0, marker="o", color="#1f77b4", label="State=0")
    ax.fill_between(h, b0 - ci_z * s0, b0 + ci_z * s0, color="#1f77b4", alpha=0.2)
    ax.plot(h, b1, marker="o", color="#d62728", label="State=1")
    ax.fill_between(h, b1 - ci_z * s1, b1 + ci_z * s1, color="#d62728", alpha=0.2)
    ax.set_title(f"{shock}: cumulative return response by state")
    ax.set_ylabel("Beta")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")

    ax2 = axes[1]
    ax2.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    colors = np.where(sig_mask, "#2ca02c", "#7f7f7f")
    ax2.bar(h, bi, color=colors, alpha=0.9, width=0.7)
    ax2.errorbar(h, bi, yerr=ci_z * si, fmt="none", ecolor="black", elinewidth=1.0, capsize=3)
    ax2.set_title("Interaction effect (State=1 minus State=0)")
    ax2.set_xlabel("Horizon (months)")
    ax2.set_ylabel("Beta interaction")
    ax2.grid(alpha=0.2)

    out = out_dir / f"irf_{shock}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)

    return {
        "shock": shock,
        "significant_horizons": sig_h,
        "avg_abs_t_interaction": float(np.mean(np.abs(ti))),
        "h12_state0": float(b0[-1]),
        "h12_state1": float(b1[-1]),
        "h12_interaction": float(bi[-1]),
    }


def _plot_heatmaps(by_shock: dict[str, list[dict[str, float]]], out_dir: Path, dpi: int) -> None:
    shocks = sorted(by_shock.keys())
    horizons = np.array([int(r["horizon"]) for r in by_shock[shocks[0]]], dtype=int)

    mat_beta = np.vstack([
        np.array([r["beta_interaction"] for r in by_shock[s]], dtype=float)
        for s in shocks
    ])
    mat_t = np.vstack([
        np.array([r["t_interaction"] for r in by_shock[s]], dtype=float)
        for s in shocks
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    vmax_b = float(np.nanmax(np.abs(mat_beta)))
    vmax_b = max(vmax_b, 1e-9)
    im0 = axes[0].imshow(mat_beta, aspect="auto", cmap="coolwarm", vmin=-vmax_b, vmax=vmax_b)
    axes[0].set_title("Interaction beta heatmap")
    axes[0].set_xticks(np.arange(len(horizons)))
    axes[0].set_xticklabels(horizons)
    axes[0].set_yticks(np.arange(len(shocks)))
    axes[0].set_yticklabels(shocks)
    axes[0].set_xlabel("Horizon")
    axes[0].set_ylabel("Shock")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    vmax_t = float(np.nanmax(np.abs(mat_t)))
    vmax_t = max(vmax_t, 1e-9)
    im1 = axes[1].imshow(mat_t, aspect="auto", cmap="PiYG", vmin=-vmax_t, vmax=vmax_t)
    axes[1].set_title("Interaction t-stat heatmap")
    axes[1].set_xticks(np.arange(len(horizons)))
    axes[1].set_xticklabels(horizons)
    axes[1].set_yticks(np.arange(len(shocks)))
    axes[1].set_yticklabels(shocks)
    axes[1].set_xlabel("Horizon")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_dir / "interaction_heatmaps.png", dpi=dpi)
    plt.close(fig)


def _plot_summary_bar(records: list[dict], out_dir: Path, dpi: int) -> None:
    shocks = [r["shock"] for r in records]
    vals = [r["avg_abs_t_interaction"] for r in records]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(shocks, vals, color="#4c78a8")
    ax.axhline(1.96, color="#d62728", linestyle="--", linewidth=1.2, label="|t|=1.96")
    ax.set_title("Average absolute interaction t-stat by shock")
    ax.set_ylabel("avg |t_interaction|")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "interaction_t_summary.png", dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[3]
    coeff_path = (root / args.coeff_csv).resolve() if not Path(args.coeff_csv).is_absolute() else Path(args.coeff_csv)
    summary_path = (root / args.summary_json).resolve() if not Path(args.summary_json).is_absolute() else Path(args.summary_json)
    out_dir = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_shock = _read_coeff(coeff_path)
    _ = _read_summary(summary_path)

    notes = []
    for shock in sorted(by_shock.keys()):
        rec = _plot_per_shock(shock, by_shock[shock], out_dir, ci_z=args.ci_z, sig_t=args.sig_t, dpi=args.dpi)
        notes.append(rec)

    _plot_heatmaps(by_shock, out_dir, dpi=args.dpi)
    _plot_summary_bar(notes, out_dir, dpi=args.dpi)

    notes_path = out_dir / "plot_notes.json"
    notes_path.write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[done] plots output dir: {out_dir}")
    print(f"[done] per-shock figures: {len(by_shock)}")
    print(f"[done] notes: {notes_path}")


if __name__ == "__main__":
    main()
