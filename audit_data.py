"""Create a compact audit of the supplied PV and weather data."""

from __future__ import annotations

import json
from zoneinfo import ZoneInfo

import pandas as pd

import paths
from analyze_pv_v2 import load_pv_v2

ROOT = paths.results_dir()
SITE_TIMEZONE = ZoneInfo("Europe/London")


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


def clock_change_hours(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Hours on the UTC grid at which the site's local clocks shift.

    A logger that records local time cannot write a row for the hour that
    local time skips at the spring transition, so a spring-forward instant is
    a place to look for a timestamp-convention artefact.
    """
    offsets = index.tz_convert(SITE_TIMEZONE).map(lambda t: t.utcoffset())
    changed = [
        index[i]
        for i in range(1, len(index))
        if offsets[i] != offsets[i - 1]
    ]
    return changed


def daylight_saving_report(pv: pd.DataFrame) -> dict[str, object]:
    """Check whether the record's clock-change hours are the missing ones.

    This is reported because it is evidence about the provenance of the
    timestamps rather than a mere curiosity: if the hours that local time
    skips are exactly the hours with no data, the logger was writing local
    time, which is the misalignment the pipeline corrects.
    """
    transitions = clock_change_hours(pv.index)
    spring_forward = [
        t
        for t in transitions
        if (t.tz_convert(SITE_TIMEZONE).utcoffset() or pd.Timedelta(0))
        > ((t - pd.Timedelta(hours=1)).tz_convert(SITE_TIMEZONE).utcoffset() or pd.Timedelta(0))
    ]
    missing_at_transition = [t for t in spring_forward if pv.loc[t, "is_missing"] == 1]
    single_hour = missing_runs(pv["is_missing"].astype(bool))
    single_hour = single_hour[single_hour["hours"] == 1]
    return {
        "spring_forward_hours_in_record": [str(t) for t in spring_forward],
        "spring_forward_hours_missing": [str(t) for t in missing_at_transition],
        "n_single_hour_gaps": int(len(single_hour)),
        "n_single_hour_gaps_at_spring_forward": int(len(missing_at_transition)),
        "all_spring_forward_hours_are_missing": bool(
            spring_forward and len(missing_at_transition) == len(spring_forward)
        ),
    }


def main() -> None:
    raw = pd.read_csv(paths.require_pv_csv(), comment="#", skip_blank_lines=True)
    raw["_time"] = pd.to_datetime(raw["_time"], utc=True)
    raw["_value"] = pd.to_numeric(raw["_value"], errors="coerce")
    raw = raw[raw["_field"] == "PPV"].sort_values("_time")

    pv = load_pv_v2(paths.require_pv_csv(), time_shift_hours=-1)
    day_missing = pv.groupby(pv.index.normalize())["is_missing"].sum()
    runs = missing_runs(pv["is_missing"].astype(bool))
    runs.to_csv(ROOT / "data_audit_missing_runs.csv", index=False)

    weather = pd.read_csv(paths.WEATHER_DIR / "weather_cache.csv", parse_dates=["time"])
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
        "n_missing_runs": int(len(runs)),
        "median_missing_run_hours": float(runs["hours"].median()),
        "pct_missing_hours_in_runs_over_3_days": float(
            100.0 * runs.loc[runs["hours"] > 72, "hours"].sum() / runs["hours"].sum()
        ),
        "daylight_saving": daylight_saving_report(pv),
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
