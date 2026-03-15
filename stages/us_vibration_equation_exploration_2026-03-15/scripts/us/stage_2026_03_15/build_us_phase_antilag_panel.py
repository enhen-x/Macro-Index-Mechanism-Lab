"""Build anti-lag feature panel by adding first-difference features i_d_*."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add anti-lag delta features to existing regression panel.")
    parser.add_argument("--input", default="data/us/us_regression_panel.csv")
    parser.add_argument("--output", default="data/us/experiments/step7_phase_antilag_features/us_regression_panel_antilag.csv")
    return parser.parse_args()


def _to_float(value: str) -> float | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.10g}"


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    with in_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty input: {in_path}")

    cols = list(rows[0].keys())
    base_cols = [c for c in cols if (c.startswith("g_") or c.startswith("c_")) and not c.endswith("_lag1")]

    derived_cols: list[str] = []
    for base in base_cols:
        lag_col = f"{base}_lag1"
        if lag_col not in cols:
            continue
        new_col = f"i_d_{base}"
        derived_cols.append(new_col)
        for r in rows:
            cur = _to_float(r.get(base, ""))
            lag = _to_float(r.get(lag_col, ""))
            if cur is None or lag is None:
                r[new_col] = ""
            else:
                r[new_col] = _fmt(cur - lag)

    out_fields = cols + derived_cols
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[done] input rows={len(rows)}")
    print(f"[done] base_cols={len(base_cols)}, derived_cols={len(derived_cols)}")
    print(f"[done] output: {out_path}")


if __name__ == "__main__":
    main()

