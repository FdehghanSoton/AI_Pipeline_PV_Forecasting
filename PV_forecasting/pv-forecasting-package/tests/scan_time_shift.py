"""Sweep the candidate PV-vs-weather timestamp shift and record alignment skill.

The alignment instrument is the *parameter-free* clear-sky physics proxy
``y_hat = capacity * GHI / 1000`` evaluated on daylight hours. Unlike a learned
model with calendar features (which can silently absorb a one-hour offset), the
physics proxy has no free parameters, so its daylight R2 directly reflects how
well the PV timestamps line up with the irradiance timestamps. The companion
GHI-PV cross-correlation peak lag is recorded for the same reason.

For each integer shift ``s`` the PV target is moved by ``s`` hours relative to the
(unshifted) weather, mirroring ``test_time_shift.py``. The resulting
``pv_time_shift_scan.csv`` is consumed by ``make_appendix_figures.py``
(figure_alignment); skill peaks at the shift the pipeline uses by default.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from analyze_pv_cnn2d import flag_daylight, load_pv, load_weather
from pipeline_paths import OUTPUT_DIR, PV_DATA_PATH

CSV_PATH = PV_DATA_PATH
OUT_PATH = OUTPUT_DIR / "pv_time_shift_scan.csv"
SHIFTS = range(-3, 4)


def _xcorr_peak_lag(y: np.ndarray, g: np.ndarray, ok: np.ndarray) -> int:
    best_lag, best_corr = 0, -np.inf
    for lag in range(-3, 4):
        if lag >= 0:
            yy, gg, mm = y[lag:], g[: len(g) - lag], ok[lag:]
        else:
            yy, gg, mm = y[:lag], g[-lag:], ok[:lag]
        if mm.sum() < 10:
            continue
        c = np.corrcoef(yy[mm], gg[mm])[0, 1]
        if c > best_corr:
            best_corr, best_lag = c, lag
    return best_lag


def _skill_at_shift(pv: pd.DataFrame, capacity: float, shift: int) -> dict:
    df = pv.copy()
    df["y"] = df["y"].shift(shift)
    df["is_missing"] = df["is_missing"].shift(shift).fillna(1).astype(int)
    df = df.dropna(subset=["y"])

    y = df["y"].to_numpy()
    g = df["shortwave_radiation"].to_numpy()
    is_day = df["is_daylight"].to_numpy().astype(bool)
    ok = is_day & (df["is_missing"].to_numpy() == 0)

    y_hat = capacity * g / 1000.0
    rmse = float(np.sqrt(mean_squared_error(y[ok], y_hat[ok])))
    return {
        "shift_hours": shift,
        "R2": float(r2_score(y[ok], y_hat[ok])),
        "nRMSE_pct": rmse / capacity * 100,
        "xcorr_peak_lag": _xcorr_peak_lag(y, g, ok),
        "n": int(ok.sum()),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pv = load_pv(CSV_PATH)
    pv = flag_daylight(pv)
    capacity = float(pv["capacity"].iloc[0])
    wx = load_weather()
    pv = pv.join(wx[["shortwave_radiation"]], how="left")
    pv["shortwave_radiation"] = pv["shortwave_radiation"].ffill().bfill()

    rows = []
    for shift in SHIFTS:
        row = _skill_at_shift(pv, capacity, shift)
        rows.append(row)
        print(
            f"shift={row['shift_hours']:+d}h  physics daylight R2={row['R2']:.3f}  "
            f"nRMSE={row['nRMSE_pct']:.2f}%  xcorr_peak_lag={row['xcorr_peak_lag']:+d}  "
            f"n={row['n']}"
        )

    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH.name}")


if __name__ == "__main__":
    main()
