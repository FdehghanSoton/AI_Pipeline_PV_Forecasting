"""Create a compact audit of the supplied PV and weather data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analyze_pv_v2 import load_pv_v2

ROOT = Path(__file__).parent


def missing_runs(mask: pd.Series) -> pd.DataFrame:
    groups = mask.ne(mask.shift()).cumsum()
    rows = []
    for _, block in mask.groupby(groups):
        if not bool(block.iloc[0]):
            continue
        rows.append(
            {
                "start": block.index.min(),
                "end": block.index.max(),
                "hours": int(len(block)),
            }
        )
    return pd.DataFrame(rows).sort_values("hours", ascending=False)


def main() -> None:
    raw = pd.read_csv(ROOT / "PV_data.csv", comment="#", skip_blank_lines=True)
    raw["_time"] = pd.to_datetime(raw["_time"], utc=True)
    raw["_value"] = pd.to_numeric(raw["_value"], errors="coerce")
    raw = raw[raw["_field"] == "PPV"].sort_values("_time")

    pv = load_pv_v2(ROOT / "PV_data.csv", time_shift_hours=-1)
    day_missing = pv.groupby(pv.index.normalize())["is_missing"].sum()
    runs = missing_runs(pv["is_missing"].astype(bool))
    runs.to_csv(ROOT / "data_audit_missing_runs.csv", index=False)

    weather = pd.read_csv(ROOT / "weather_cache.csv", parse_dates=["time"])
    weather["time"] = pd.to_datetime(weather["time"], utc=True)

    summary = {
        "pv_raw_rows": int(len(raw)),
        "pv_raw_start": str(raw["_time"].min()),
        "pv_raw_end": str(raw["_time"].max()),
        "pv_hourly_grid_rows_after_shift": int(len(pv)),
        "pv_missing_hours": int(pv["is_missing"].sum()),
        "pv_missing_percentage": float(pv["is_missing"].mean() * 100),
        "calendar_days_on_grid": int(len(day_missing)),
        "days_with_any_missing_pv": int((day_missing > 0).sum()),
        "fully_missing_days": int((day_missing == 24).sum()),
        "partially_missing_days": int(((day_missing > 0) & (day_missing < 24)).sum()),
        "longest_missing_run_hours": int(runs.iloc[0]["hours"]),
        "empirical_capacity_q999_kw": float(pv["y"].quantile(0.999)),
        "observed_max_kw": float(raw["_value"].max()),
        "weather_rows": int(len(weather)),
        "weather_start": str(weather["time"].min()),
        "weather_end": str(weather["time"].max()),
        "weather_columns": [c for c in weather.columns if c != "time"],
        "weather_missing_cells": int(weather.drop(columns="time").isna().sum().sum()),
        "weather_provenance_metadata_present": bool(
            (ROOT / "weather_cache.meta.json").exists()
        ),
    }
    (ROOT / "data_audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
