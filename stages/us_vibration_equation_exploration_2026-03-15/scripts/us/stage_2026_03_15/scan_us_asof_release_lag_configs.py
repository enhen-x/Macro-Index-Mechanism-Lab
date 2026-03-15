"""Step 13b: scan as-of release-lag configurations and compare phase metrics."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan release-lag configs for as-of alignment.")
    parser.add_argument("--input-monthly", default="data/us/us_monthly_panel.csv")
    parser.add_argument("--output-dir", default="data/us/experiments/step13b_asof_lag_scan")
    parser.add_argument("--baseline-metrics", default="data/us/plot/fit_metrics.json")
    parser.add_argument("--baseline-phase", default="data/us/experiments/step5_phase_bias_baseline/phase_bias_summary.json")
    return parser.parse_args()


def _run(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = json.loads(Path(args.baseline_metrics).read_text(encoding="utf-8"))
    baseline_phase = json.loads(Path(args.baseline_phase).read_text(encoding="utf-8"))

    # lag-map format: field:lag,field:lag...
    configs = [
        ("all_0", "walcl:0,bus_loans:0,cpi:0,indpro:0,unrate:0,fed_funds:0"),
        ("macro1_ff0", "walcl:1,bus_loans:1,cpi:1,indpro:1,unrate:1,fed_funds:0"),
        ("macro1_ff1", "walcl:1,bus_loans:1,cpi:1,indpro:1,unrate:1,fed_funds:1"),
        ("core1_slow0_ff0", "walcl:0,bus_loans:0,cpi:1,indpro:1,unrate:1,fed_funds:0"),
        ("macro2_ff1", "walcl:2,bus_loans:2,cpi:2,indpro:2,unrate:2,fed_funds:1"),
    ]

    rows: list[dict[str, object]] = []
    for name, lag_map in configs:
        cfg_dir = out_dir / name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        try:
            _run(
                [
                    sys.executable,
                    "scripts/us/stage_2026_03_15/build_us_asof_monthly_panel.py",
                    "--input",
                    args.input_monthly,
                    "--output",
                    str(cfg_dir / "us_monthly_panel_asof.csv"),
                    "--lag-map",
                    lag_map,
                ],
                cwd=ROOT,
            )
            _run(
                [
                    sys.executable,
                    "scripts/us/stage_2026_03_15/build_us_regression_panel.py",
                    "--input",
                    str(cfg_dir / "us_monthly_panel_asof.csv"),
                    "--output",
                    str(cfg_dir / "us_regression_panel_asof.csv"),
                ],
                cwd=ROOT,
            )
            _run(
                [
                    sys.executable,
                    "scripts/us/stage_2026_03_15/run_us_ols_estimation.py",
                    "--input",
                    str(cfg_dir / "us_regression_panel_asof.csv"),
                    "--target-mode",
                    "y_next",
                    "--split",
                    "train",
                    "--ridge-alpha",
                    "0.05",
                    "--damping-mode",
                    "nonlinear_absv",
                    "--output",
                    str(cfg_dir / "us_ols_estimation_asof.json"),
                ],
                cwd=ROOT,
            )
            _run(
                [
                    sys.executable,
                    "scripts/us/stage_2026_03_15/plot_us_ols_fit.py",
                    "--panel",
                    str(cfg_dir / "us_regression_panel_asof.csv"),
                    "--estimation",
                    str(cfg_dir / "us_ols_estimation_asof.json"),
                    "--split",
                    "test",
                    "--output-dir",
                    str(cfg_dir / "plot_test"),
                ],
                cwd=ROOT,
            )
            _run(
                [
                    sys.executable,
                    "scripts/us/stage_2026_03_15/experiment_us_phase_bias_baseline.py",
                    "--panel",
                    str(cfg_dir / "us_regression_panel_asof.csv"),
                    "--estimation",
                    str(cfg_dir / "us_ols_estimation_asof.json"),
                    "--split",
                    "test",
                    "--output-dir",
                    str(cfg_dir / "phase_test"),
                ],
                cwd=ROOT,
            )

            fit = json.loads((cfg_dir / "plot_test" / "fit_metrics.json").read_text(encoding="utf-8"))
            phase = json.loads((cfg_dir / "phase_test" / "phase_bias_summary.json").read_text(encoding="utf-8"))
            row = {
                "name": name,
                "lag_map": lag_map,
                "status": "ok",
                "x_r2": float(fit["x_r2"]),
                "x_rmse": float(fit["x_rmse"]),
                "x_direction": float(fit["direction_accuracy"]),
                "best_lag": int(phase["best_lag_by_corr"]["lag"]),
                "best_lag_corr": float(phase["best_lag_by_corr"]["corr"]),
                "turn_recall": float(phase["turn_metrics_lag0"]["turn_hit_rate_recall"]),
                "turn_precision": float(phase["turn_metrics_lag0"]["turn_precision"]),
                "turn_mean_delay": float(phase["turn_metrics_lag0"]["mean_delay"]),
                "gap_x_r2_vs_main": float(fit["x_r2"] - float(baseline_metrics["x_r2"])),
                "gap_x_rmse_vs_main": float(fit["x_rmse"] - float(baseline_metrics["x_rmse"])),
                "gap_best_lag_vs_main": int(int(phase["best_lag_by_corr"]["lag"]) - int(baseline_phase["best_lag_by_corr"]["lag"])),
            }
        except Exception as e:
            row = {"name": name, "lag_map": lag_map, "status": f"failed: {e}"}
        rows.append(row)

    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        raise RuntimeError("all lag configs failed")

    best_phase = sorted(
        ok,
        key=lambda r: (
            abs(int(r["best_lag"])),
            float(r["turn_mean_delay"]),
            -float(r["x_r2"]),
        ),
    )[0]
    best_overall = sorted(
        ok,
        key=lambda r: (float(r["x_r2"]), -float(r["x_rmse"])),
        reverse=True,
    )[0]

    summary = {
        "baseline_metrics_path": str(args.baseline_metrics),
        "baseline_phase_path": str(args.baseline_phase),
        "configs": rows,
        "best_phase": best_phase,
        "best_overall": best_overall,
    }
    (out_dir / "asof_lag_scan_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with (out_dir / "asof_lag_scan_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "status",
                "lag_map",
                "x_r2",
                "x_rmse",
                "x_direction",
                "best_lag",
                "best_lag_corr",
                "turn_recall",
                "turn_precision",
                "turn_mean_delay",
                "gap_x_r2_vs_main",
                "gap_x_rmse_vs_main",
                "gap_best_lag_vs_main",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("[done] step13b as-of lag scan complete")
    print(f"[done] candidates={len(rows)}, ok={len(ok)}")
    print(
        "[done] best_phase: "
        f"{best_phase['name']}, best_lag={best_phase['best_lag']}, "
        f"turn_delay={best_phase['turn_mean_delay']:.6g}, x_r2={best_phase['x_r2']:.6g}"
    )
    print(
        "[done] best_overall: "
        f"{best_overall['name']}, best_lag={best_overall['best_lag']}, "
        f"x_r2={best_overall['x_r2']:.6g}, x_rmse={best_overall['x_rmse']:.6g}"
    )
    print(f"[done] output: {out_dir}")


if __name__ == "__main__":
    main()
