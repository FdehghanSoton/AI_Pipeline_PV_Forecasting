"""Test whether the timestamp offset is a clock convention or a local-time error.

The record shows two fingerprints of a timestamp problem, and they imply
different corrections. The two missing spring-forward hours say the logger was
aware of local time. An interval-ending convention, where each hourly value is
stamped with the end of the period it summarises, says something else. The
distinction matters: a logger writing British Summer Time as though it were UTC
would be offset by one hour in summer and not at all in winter, whereas an
interval-ending convention is offset by one hour all year.

So the question is decidable from the data. This script runs the alignment
scan separately over the British Summer Time and Greenwich Mean Time parts of
the record. A shift that is the same in both periods is a clock convention; a
shift that appears only in summer is a local-time error.

The instrument is the same parameter-free clear-sky proxy used by
``scan_time_shift.py``: ``y_hat = capacity * GHI / 1000`` on daylight hours,
which has no free parameters and so cannot absorb an offset the way a learned
model can.

Run ``python scan_shift_by_dst.py``. Writes ``pv_time_shift_by_dst.csv``.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import paths
from analyze_pv_cnn2d import flag_daylight, load_pv, load_weather

SITE_TIMEZONE = ZoneInfo("Europe/London")
SHIFTS = range(-3, 4)
OUT_PATH = paths.results_dir() / "pv_time_shift_by_dst.csv"


def utc_offset_hours(index: pd.DatetimeIndex) -> np.ndarray:
    """Local offset from UTC at each timestamp: 1 during BST, 0 during GMT."""
    local = index.tz_convert(SITE_TIMEZONE)
    return np.array([o.total_seconds() / 3600.0 for o in local.map(lambda t: t.utcoffset())])


def scan(pv: pd.DataFrame, capacity: float, label: str, mask: np.ndarray) -> list[dict]:
    rows = []
    for shift in SHIFTS:
        df = pv.copy()
        df["y"] = df["y"].shift(shift)
        df["is_missing"] = df["is_missing"].shift(shift).fillna(1).astype(int)
        df["period"] = mask
        df = df.dropna(subset=["y"])

        ok = (
            df["is_daylight"].to_numpy().astype(bool)
            & (df["is_missing"].to_numpy() == 0)
            & df["period"].to_numpy().astype(bool)
        )
        if ok.sum() < 100:
            continue
        y = df["y"].to_numpy()[ok]
        y_hat = (capacity * df["shortwave_radiation"].to_numpy() / 1000.0)[ok]
        rows.append(
            {
                "period": label,
                "shift_hours": shift,
                "R2": float(r2_score(y, y_hat)),
                "n": int(ok.sum()),
            }
        )
    return rows


def dst_aware_scan(
    pv: pd.DataFrame, capacity: float, offset: np.ndarray
) -> list[dict]:
    """Score corrections that treat the timestamps as local time.

    A logger writing local wall-clock time into a field read as UTC produces a
    record that is ahead of true UTC by the local offset, which is one hour in
    British Summer Time and zero in Greenwich Mean Time. Undoing that means
    shifting each row by its own offset rather than by a constant. An extra
    constant is swept alongside to absorb any interval-labelling convention
    that remains once the clock error is removed.
    """
    rows = []
    for extra in [-2, -1, 0, 1]:
        shifted = pv.copy()
        # Each row moves by its own local offset plus the constant, so the
        # correction is applied per row rather than to the series as a whole.
        per_row = -offset + extra
        y = np.full(len(pv), np.nan)
        missing = np.ones(len(pv), dtype=int)
        source = np.arange(len(pv)) - per_row.astype(int)
        valid = (source >= 0) & (source < len(pv))
        y[valid] = pv["y"].to_numpy()[source[valid]]
        missing[valid] = pv["is_missing"].to_numpy()[source[valid]]
        shifted["y"] = y
        shifted["is_missing"] = missing

        ok = (
            shifted["is_daylight"].to_numpy().astype(bool)
            & (missing == 0)
            & ~np.isnan(y)
        )
        y_hat = capacity * shifted["shortwave_radiation"].to_numpy() / 1000.0
        rows.append(
            {
                "correction": f"local time, then {extra:+d} h",
                "R2": float(r2_score(y[ok], y_hat[ok])),
                "n": int(ok.sum()),
            }
        )
    return rows


def main() -> None:
    pv = load_pv(paths.require_pv_csv())
    pv = flag_daylight(pv)
    capacity = float(pv["capacity"].iloc[0])
    wx = load_weather()
    pv = pv.join(wx[["shortwave_radiation"]], how="left")
    pv["shortwave_radiation"] = pv["shortwave_radiation"].ffill().bfill()

    offset = utc_offset_hours(pv.index)
    periods = {
        "BST (UTC+1)": offset == 1,
        "GMT (UTC+0)": offset == 0,
        "whole record": np.ones(len(pv), dtype=bool),
    }

    rows: list[dict] = []
    for label, mask in periods.items():
        rows.extend(scan(pv, capacity, label, mask))

    table = pd.DataFrame(rows)
    table.to_csv(OUT_PATH, index=False)

    print("\n=== Clear-sky proxy daylight R2 by candidate shift and clock period ===\n")
    wide = table.pivot(index="shift_hours", columns="period", values="R2")
    print(wide.to_string(float_format="%.4f"))

    print("\nBest shift in each period:")
    for label in periods:
        block = table[table["period"] == label]
        if block.empty:
            continue
        best = block.loc[block["R2"].idxmax()]
        print(
            f"  {label:<14} shift={int(best['shift_hours']):+d} h  "
            f"R2={best['R2']:.4f}  (n={int(best['n'])} daylight hours)"
        )

    bst = table[table["period"] == "BST (UTC+1)"]
    gmt = table[table["period"] == "GMT (UTC+0)"]
    if not bst.empty and not gmt.empty:
        best_bst = int(bst.loc[bst["R2"].idxmax(), "shift_hours"])
        best_gmt = int(gmt.loc[gmt["R2"].idxmax(), "shift_hours"])
        print()
        if best_bst == best_gmt:
            print(
                f"Both periods prefer {best_bst:+d} h. The offset is a fixed clock "
                "convention, not a daylight-saving error: a logger writing local "
                "time as UTC would need different corrections in the two periods."
            )
        else:
            print(
                f"BST prefers {best_bst:+d} h and GMT prefers {best_gmt:+d} h, a "
                "difference of one hour, which is the signature of local time "
                "being recorded as though it were UTC."
            )

    print("\n=== Treating the timestamps as local time instead ===\n")
    constant = table[table["period"] == "whole record"]
    best_constant = constant.loc[constant["R2"].idxmax()]
    print(
        f"  best constant shift    {int(best_constant['shift_hours']):+d} h        "
        f"R2={best_constant['R2']:.4f}"
    )
    for row in dst_aware_scan(pv, capacity, offset):
        print(f"  {row['correction']:<22} R2={row['R2']:.4f}  (n={row['n']})")

    print(f"\nWrote {OUT_PATH.name}")


if __name__ == "__main__":
    main()
