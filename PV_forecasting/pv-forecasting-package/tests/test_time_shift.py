"""Test the +1h shift hypothesis.

If PV(t) corresponds to GHI(t-1) (right-edge vs left-edge labelling),
shifting PV one hour back should:
  (a) raise R² dramatically on physics + ML baselines
  (b) bring the clear-sky-index distribution back to the healthy 0–1 range
  (c) make the cross-correlation peak at lag 0
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from analyze_pv_cnn2d import WEATHER_CHANNELS, flag_daylight, load_pv, load_weather

CSV_PATH = Path(__file__).with_name("PV_data.csv")


def metrics(name, y, yh, cap):
    m = np.isfinite(y) & np.isfinite(yh)
    y, yh = y[m], yh[m]
    return {
        "model": name,
        "n": int(len(y)),
        "R2": float(r2_score(y, yh)),
        "MAE": float(mean_absolute_error(y, yh)),
        "RMSE": float(np.sqrt(mean_squared_error(y, yh))),
        "nRMSE_%cap": float(np.sqrt(mean_squared_error(y, yh)) / cap * 100),
    }


def evaluate(pv: pd.DataFrame, capacity: float, label: str) -> list[dict]:
    print(f"\n=========  {label}  =========")
    g = pv["shortwave_radiation"].to_numpy()
    dni = pv["direct_normal_irradiance"].to_numpy()
    y = pv["y"].to_numpy()
    is_day = pv["is_daylight"].to_numpy().astype(bool)
    is_miss = pv["is_missing"].to_numpy().astype(bool)

    # Cross-correlation
    ok = is_day & ~is_miss & np.isfinite(g)
    print("lag  |  corr")
    for L in range(-2, 4):
        if L >= 0:
            yy = y[L:][ok[L:]]
            gg = g[: len(g) - L][ok[L:]]
        else:
            yy = y[:L][ok[:L]]
            gg = g[-L:][ok[:L]]
        c = np.corrcoef(yy, gg)[0, 1]
        marker = (
            " <-- peak"
            if L in (-2, -1, 0, 1, 2)
            and c
            == max(
                np.corrcoef(
                    y[max(0, k) : len(y) + min(0, k)][
                        ok[max(0, k) : len(y) + min(0, k)]
                    ]
                    if k >= 0
                    else y[:k][ok[:k]],
                    g[: len(g) - max(0, k)][ok[max(0, k) : len(g) + min(0, k)]]
                    if k >= 0
                    else g[-k:][ok[:k]],
                )[0, 1]
                for k in range(-2, 3)
            )
            else ""
        )
        print(f" {L:+2d}  |  {c:.4f}{marker}")

    # Clear-sky index sanity
    physics = capacity * g / 1000.0
    ok_csi = is_day & ~is_miss & (g > 50)
    k = y[ok_csi] / np.maximum(physics[ok_csi], 1e-3)
    print(
        f"\nCSI: median={np.median(k):.3f}  P50={np.percentile(k, 50):.3f}  "
        f"P90={np.percentile(k, 90):.3f}  %k>1.2 = {(k > 1.2).mean() * 100:.1f}%"
    )

    rows = []
    for tag, mask in [("ALL", ~is_miss), ("daylight", ~is_miss & is_day)]:
        rows.append(
            metrics(
                f"physics y=cap*GHI/1000 [{tag}]",
                y[mask],
                (capacity * g / 1000)[mask],
                capacity,
            )
        )
        # OLS GHI+DNI
        X = np.stack([g, dni], axis=1)
        good = mask & np.isfinite(X).all(axis=1)
        lr = LinearRegression().fit(X[good], y[good])
        rows.append(
            metrics(
                f"OLS y~GHI+DNI [{tag}]",
                y[mask],
                np.clip(lr.predict(X), 0, None)[mask],
                capacity,
            )
        )

    # GBM with calendar+weather, last-90d holdout
    df = pv.copy()
    hr = df.index.hour
    doy = df.index.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    feats = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"] + WEATHER_CHANNELS
    cutoff = df.index.max() - pd.Timedelta(days=90)
    train = df[(df.index < cutoff) & (df["is_missing"] == 0)]
    test = df[(df.index >= cutoff) & (df["is_missing"] == 0)]
    gbm = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.05,
        max_depth=8,
        min_samples_leaf=30,
        l2_regularization=0.1,
        random_state=0,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.1,
    )
    gbm.fit(train[feats], train["y"])
    yh = np.clip(gbm.predict(test[feats]), 0, None)
    y_test = test["y"].to_numpy()
    is_day_test = test["is_daylight"].to_numpy().astype(bool)
    rows.append(metrics("GBM holdout last90d [ALL]", y_test, yh, capacity))
    rows.append(
        metrics(
            "GBM holdout last90d [daylight]",
            y_test[is_day_test],
            yh[is_day_test],
            capacity,
        )
    )

    df_r = pd.DataFrame(rows)
    print()
    print(df_r.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return rows


def main():
    pv = load_pv(CSV_PATH)
    pv = flag_daylight(pv)
    capacity = float(pv["capacity"].iloc[0])
    wx = load_weather()
    pv = pv.join(wx, how="left")
    pv[WEATHER_CHANNELS] = pv[WEATHER_CHANNELS].ffill().bfill()

    evaluate(pv, capacity, "ORIGINAL (no shift)")

    # Try shifting PV by -1h: PV(t) -> PV(t-1)
    pv_shift = pv.copy()
    pv_shift["y"] = pv_shift["y"].shift(-1)
    pv_shift["is_missing"] = pv_shift["is_missing"].shift(-1).fillna(1).astype(int)
    pv_shift = pv_shift.iloc[:-1]
    evaluate(pv_shift, capacity, "PV shifted -1h (PV(t) -> hour t-1)")

    # Try shifting weather by +1h instead (equivalent but cleaner: weather(t) -> hour t+1)
    pv_wxshift = pv.copy()
    for c in WEATHER_CHANNELS:
        pv_wxshift[c] = pv_wxshift[c].shift(1)
    pv_wxshift = pv_wxshift.iloc[1:]
    evaluate(pv_wxshift, capacity, "Weather shifted +1h (use prev-hour weather)")


if __name__ == "__main__":
    main()
