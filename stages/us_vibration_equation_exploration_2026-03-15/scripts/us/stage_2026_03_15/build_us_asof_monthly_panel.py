"""Build an as-of aligned monthly panel by applying release lags (in months).

Rule:
- lag = L means value at month t can only use raw value from month t-L.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


DEFAULT_RELEASE_LAG = {
    # market variables: effectively contemporaneous in monthly aggregation
    "sp500": 0,
    "ust10y": 0,
    "dxy_broad": 0,
    "vix": 0,
    # policy rate often known quickly for month, keep 0 in base setting
    "fed_funds": 0,
    # macro releases: apply conservative +1 month as-of lag
    "walcl": 1,
    "bus_loans": 1,
    "cpi": 1,
    "indpro": 1,
    "unrate": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build as-of monthly panel with release-lag mapping.")
    parser.add_argument("--input", default="data/us/us_monthly_panel.csv")
    parser.add_argument("--output", default="data/us/experiments/step13_asof_alignment/us_monthly_panel_asof.csv")
    parser.add_argument(
        "--lag-map",
        default="",
        help=(
            "Override lag mapping: comma-separated key:int pairs, "
            "e.g. cpi:1,indpro:1,unrate:1,fed_funds:0"
        ),
    )
    parser.add_argument(
        "--drop-leading-na",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, drop early rows that become NA after lag shift.",
    )
    return parser.parse_args()


def _to_float(value: str) -> float:
    text = (value or "").strip()
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def _fmt(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return f"{float(value):.10g}"


def _parse_lag_map(raw: str) -> dict[str, int]:
    out: dict[str, int] = {}
    text = (raw or "").strip()
    if not text:
        return out
    for part in text.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"invalid lag-map part: {chunk}")
        key, val = chunk.split(":", 1)
        k = key.strip()
        v = int(val.strip())
        if v < 0:
            raise ValueError(f"lag must be >=0 for {k}")
        out[k] = v
    return out


def _shift_by_lag(x: np.ndarray, lag: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    if lag == 0:
        out[:] = x
        return out
    if lag >= len(x):
        return out
    out[lag:] = x[:-lag]
    return out


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    with in_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty input: {in_path}")

    fields = list(rows[0].keys())
    if "date" not in fields:
        raise ValueError("input panel missing date column")

    lag_map = dict(DEFAULT_RELEASE_LAG)
    lag_map.update(_parse_lag_map(args.lag_map))

    dates = [r["date"] for r in rows]
    numeric_cols = [c for c in fields if c != "date"]
    arrays: dict[str, np.ndarray] = {}
    for col in numeric_cols:
        arrays[col] = np.array([_to_float(r.get(col, "")) for r in rows], dtype=float)

    shifted: dict[str, np.ndarray] = {}
    for col in numeric_cols:
        lag = int(lag_map.get(col, 0))
        shifted[col] = _shift_by_lag(arrays[col], lag=lag)

    valid_mask = np.ones(len(rows), dtype=bool)
    if args.drop_leading_na:
        for col in numeric_cols:
            valid_mask &= np.isfinite(shifted[col])

    out_rows: list[dict[str, str]] = []
    for i in range(len(rows)):
        if not valid_mask[i]:
            continue
        rec: dict[str, str] = {"date": dates[i]}
        for col in numeric_cols:
            rec[col] = _fmt(shifted[col][i])
        out_rows.append(rec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[done] input rows={len(rows)}, output rows={len(out_rows)}")
    print(f"[done] applied lag map: {lag_map}")
    print(f"[done] output: {out_path}")


if __name__ == "__main__":
    main()

