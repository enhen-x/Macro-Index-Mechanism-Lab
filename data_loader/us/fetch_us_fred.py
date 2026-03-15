"""Fetch US macro/market series from FRED and save to data/us.

Usage (PowerShell):
  python data_loader/us/fetch_us_fred.py
  python data_loader/us/fetch_us_fred.py --start-date 2000-01-01
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from urllib.parse import urlencode
from urllib.request import urlopen


@dataclass
class SeriesSpec:
    field_name: str
    source: str
    series_id: str
    frequency: str
    role: str
    notes: str
    fallback_source: str = ""
    fallback_url: str = ""


FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch US series from FRED.")
    parser.add_argument("--start-date", default="", help="Observation start date (YYYY-MM-DD).")
    parser.add_argument("--api-key", default=os.getenv("FRED_API_KEY", ""), help="FRED API key.")
    return parser.parse_args()


def load_local_config() -> dict[str, str]:
    """Load local config from untracked file if present.

    Supported format in data_loader/us/fred_api_key.txt:
      api_key=YOUR_KEY
      start_date=2005-01-01
    """
    candidates = [
        Path(__file__).resolve().parent / "fred_api_key.txt",
        Path.cwd() / "data_loader" / "us" / "fred_api_key.txt",
        Path("data_loader/us/fred_api_key.txt"),
    ]

    key_file: Path | None = None
    for p in candidates:
        if p.exists():
            key_file = p
            break
    if key_file is None:
        return {}

    # Handle UTF-8 BOM created by some Windows editors/commands.
    text = key_file.read_text(encoding="utf-8").lstrip("\ufeff")

    cfg: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            key = k.strip().lower().lstrip("\ufeff")
            cfg[key] = v.strip()
        else:
            # Backward compatibility: single-line file treated as api key.
            cfg["api_key"] = line
    return cfg


def load_series_map(path: Path) -> list[SeriesSpec]:
    specs: list[SeriesSpec] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("source", "").strip().upper() != "FRED":
                continue
            specs.append(
                SeriesSpec(
                    field_name=row["field_name"].strip(),
                    source=row["source"].strip(),
                    series_id=row["series_id"].strip(),
                    frequency=row["frequency"].strip(),
                    role=row["role"].strip(),
                    notes=row["notes"].strip(),
                    fallback_source=(row.get("fallback_source") or "").strip(),
                    fallback_url=(row.get("fallback_url") or "").strip(),
                )
            )
    return specs


def fetch_fred_series(series_id: str, api_key: str, start_date: str) -> list[tuple[str, float]]:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
    }
    url = f"{FRED_BASE}?{urlencode(params)}"
    with urlopen(url, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    out: list[tuple[str, float]] = []
    for obs in payload.get("observations", []):
        date_str = obs.get("date", "")
        value_str = obs.get("value", ".")
        if not date_str or value_str in (".", ""):
            continue
        try:
            value = float(value_str)
        except ValueError:
            continue
        out.append((date_str, value))
    return out


def fetch_stooq_series(url: str) -> list[tuple[str, float]]:
    """Fetch daily OHLC CSV from Stooq, return (date, close)."""
    with urlopen(url, timeout=30) as resp:
        text = resp.read().decode("utf-8", errors="replace")

    out: list[tuple[str, float]] = []
    reader = csv.DictReader(text.splitlines())
    for row in reader:
        d = (row.get("Date") or "").strip()
        c = (row.get("Close") or "").strip()
        if not d or not c or c.upper() == "N/A":
            continue
        try:
            out.append((d, float(c)))
        except ValueError:
            continue
    return out


def append_fred_series_with_overlap(
    base_rows: list[tuple[str, float]],
    append_rows: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Append a second FRED series with overlap scaling for continuity.

    If overlap exists, scale append series by median(base/append) on overlap dates.
    Then keep append rows strictly after base max date.
    """
    if not base_rows:
        return append_rows
    if not append_rows:
        return base_rows

    base_map = {d: v for d, v in base_rows}
    append_map = {d: v for d, v in append_rows}
    overlap_dates = sorted(set(base_map).intersection(append_map))

    scale = 1.0
    if overlap_dates:
        ratios = []
        for d in overlap_dates:
            den = append_map[d]
            if den != 0:
                ratios.append(base_map[d] / den)
        if ratios:
            scale = median(ratios)

    base_last = max(base_map.keys())
    tail = [(d, v * scale) for d, v in append_rows if d > base_last]
    return base_rows + tail


def prepend_fred_series_with_overlap(
    base_rows: list[tuple[str, float]],
    prepend_rows: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Prepend an older series with overlap scaling for continuity.

    If overlap exists, scale prepend series by median(base/prepend) on overlap dates.
    Then keep prepend rows strictly before base min date.
    """
    if not base_rows:
        return prepend_rows
    if not prepend_rows:
        return base_rows

    base_map = {d: v for d, v in base_rows}
    pre_map = {d: v for d, v in prepend_rows}
    overlap_dates = sorted(set(base_map).intersection(pre_map))

    scale = 1.0
    if overlap_dates:
        ratios = []
        for d in overlap_dates:
            den = pre_map[d]
            if den != 0:
                ratios.append(base_map[d] / den)
        if ratios:
            scale = median(ratios)

    base_first = min(base_map.keys())
    head = [(d, v * scale) for d, v in prepend_rows if d < base_first]
    return head + base_rows


def month_key(date_str: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.strftime("%Y-%m")


def is_later_than(d1: str, d2: str) -> bool:
    return datetime.strptime(d1, "%Y-%m-%d") > datetime.strptime(d2, "%Y-%m-%d")


def aggregate_to_monthly(rows: list[tuple[str, float]], frequency: str) -> list[tuple[str, float]]:
    """Normalize any input series to monthly rows with YYYY-MM-01 dates.

    Rules:
    - daily/weekly: monthly average
    - monthly: last observation in month (robust to accidental higher-frequency data)
    """
    freq = frequency.lower().strip()
    buckets: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for d, v in rows:
        buckets[month_key(d)].append((d, v))

    out: list[tuple[str, float]] = []
    for ym in sorted(buckets.keys()):
        items = buckets[ym]
        if freq == "monthly":
            # Keep the latest observation in the month to avoid creating daily rows.
            items_sorted = sorted(items, key=lambda x: x[0])
            val = items_sorted[-1][1]
        else:
            vals = [v for _, v in items]
            val = sum(vals) / len(vals)
        out.append((f"{ym}-01", val))
    return out


def save_raw_csv(path: Path, rows: list[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "value"])
        writer.writerows(rows)


def build_monthly_panel(series_rows: dict[str, list[tuple[str, float]]]) -> list[dict[str, str]]:
    all_dates = set()
    for rows in series_rows.values():
        all_dates.update(d for d, _ in rows)
    sorted_dates = sorted(all_dates)

    by_field_date: dict[str, dict[str, float]] = {}
    for field, rows in series_rows.items():
        by_field_date[field] = {d: v for d, v in rows}

    panel: list[dict[str, str]] = []
    for d in sorted_dates:
        record: dict[str, str] = {"date": d}
        for field in sorted(by_field_date.keys()):
            val = by_field_date[field].get(d)
            record[field] = "" if val is None else f"{val:.10g}"
        panel.append(record)
    return panel


def save_panel_csv(path: Path, panel: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not panel:
        return
    fields = list(panel[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(panel)


def main() -> None:
    args = parse_args()
    local_cfg = load_local_config()
    api_key = args.api_key.strip() or local_cfg.get("api_key", "")
    start_date = args.start_date.strip() or local_cfg.get("start_date", "2005-01-01")

    if not api_key:
        raise SystemExit("Missing FRED API key. Set FRED_API_KEY env var or pass --api-key.")

    repo_root = Path(__file__).resolve().parents[2]
    series_map_path = Path(__file__).resolve().parent / "series_map.csv"
    specs = load_series_map(series_map_path)
    if not specs:
        raise SystemExit("No FRED series found in series_map.csv.")

    raw_root = repo_root / "data" / "us" / "raw"
    panel_path = repo_root / "data" / "us" / "us_monthly_panel.csv"

    monthly_rows_by_field: dict[str, list[tuple[str, float]]] = {}
    for spec in specs:
        print(f"[fetch] {spec.field_name} ({spec.series_id})")
        rows = fetch_fred_series(spec.series_id, api_key, start_date)
        if (
            rows
            and spec.fallback_source.upper() == "STOOQ"
            and spec.fallback_url
            and is_later_than(rows[0][0], start_date)
        ):
            print(
                f"[fallback] {spec.field_name}: FRED starts at {rows[0][0]}, "
                f"fallback to {spec.fallback_source} for longer history"
            )
            try:
                fb_rows = fetch_stooq_series(spec.fallback_url)
                if fb_rows:
                    rows = [(d, v) for (d, v) in fb_rows if d >= start_date]
            except Exception as exc:
                print(f"[warn] fallback failed for {spec.field_name}: {exc}")
        monthly_rows = aggregate_to_monthly(rows, spec.frequency)

        if spec.fallback_source.upper() == "FRED_APPEND" and spec.fallback_url:
            try:
                append_rows = fetch_fred_series(spec.fallback_url, api_key, start_date)
                if append_rows:
                    append_monthly = aggregate_to_monthly(append_rows, spec.frequency)
                    monthly_rows = append_fred_series_with_overlap(monthly_rows, append_monthly)
                    print(
                        f"[append] {spec.field_name}: base {spec.series_id} + {spec.fallback_url}, "
                        f"monthly_rows={len(monthly_rows)}"
                    )
            except Exception as exc:
                print(f"[warn] append failed for {spec.field_name}: {exc}")
        elif spec.fallback_source.upper() == "FRED_PREPEND" and spec.fallback_url:
            try:
                pre_rows = fetch_fred_series(spec.fallback_url, api_key, start_date)
                if pre_rows:
                    pre_monthly = aggregate_to_monthly(pre_rows, spec.frequency)
                    monthly_rows = prepend_fred_series_with_overlap(monthly_rows, pre_monthly)
                    print(
                        f"[prepend] {spec.field_name}: {spec.fallback_url} -> {spec.series_id}, "
                        f"monthly_rows={len(monthly_rows)}"
                    )
            except Exception as exc:
                print(f"[warn] prepend failed for {spec.field_name}: {exc}")

        save_raw_csv(raw_root / f"{spec.field_name}.csv", rows)
        monthly_rows_by_field[spec.field_name] = monthly_rows

    panel = build_monthly_panel(monthly_rows_by_field)
    save_panel_csv(panel_path, panel)
    print(f"[done] raw series -> {raw_root}")
    print(f"[done] monthly panel -> {panel_path}")


if __name__ == "__main__":
    main()
