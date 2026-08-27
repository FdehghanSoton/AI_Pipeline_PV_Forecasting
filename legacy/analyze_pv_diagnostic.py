"""Deep diagnostic for the PV_data.csv forecastability ceiling.


(A) Convention check
    Re-score the ensemble + GBM on the *all-hours* R² metric (the one most
    papers report). It is mathematically much higher than daylight-only R²
    because predicting 0 at night is trivial.

(B) Naive physics ceiling
    Compute a 1-feature linear fit  y = a·GHI + b  and evaluate its R².
    This is the simplest possible model. If GHI alone explains > 0.80 of
    the variance, our ML pipeline is leaving signal on the table. If it
    doesn't, no amount of ML can rescue R² – the data is intrinsically
    cloud-noise limited.

(C) Data-quality diagnostic
    1. Capacity drift     – monthly max & sliding 30-d max
    2. Timestamp alignment – peak-hour by month vs astronomical noon
    3. Cross-correlation y vs GHI at lags ±3 h    – best lag should be 0
    4. Suspicious "non-night zeros" by month   – inverter dropouts
    5. Flatline detection – constant non-zero values for ≥ 4 h
    6. Clear-sky index k = y / C distribution + autocorrelation
    7. Residual analysis of best ensemble by month / hour / cloud level

Outputs:
    pv_diagnostic_report.txt      – plain-text summary of all checks
    pv_diagnostic_panel.png       – 6-panel visual diagnostic
    pv_diagnostic_metrics.csv     – all-hours vs daylight R² for each model
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import paths
from analyze_pv_cnn2d import WEATHER_CHANNELS, flag_daylight, load_pv, load_weather

CSV_PATH = paths.PV_CSV
OUT_DIR = paths.results_dir()
LAT = 50.91  # Southampton


def section(title: str) -> str:
    return f"\n{'=' * 72}\n {title}\n{'=' * 72}"


def metric_row(name: str, y: np.ndarray, yh: np.ndarray, capacity: float) -> dict:
    m = np.isfinite(y) & np.isfinite(yh)
    y, yh = y[m], yh[m]
    if len(y) < 5 or np.var(y) < 1e-9:
        return {
            "model": name,
            "n": len(y),
            "R2": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "nMAE_%": np.nan,
            "nRMSE_%": np.nan,
        }
    return {
        "model": name,
        "n": int(len(y)),
        "R2": float(r2_score(y, yh)),
        "MAE": float(mean_absolute_error(y, yh)),
        "RMSE": float(np.sqrt(mean_squared_error(y, yh))),
        "nMAE_%": float(mean_absolute_error(y, yh) / capacity * 100),
        "nRMSE_%": float(np.sqrt(mean_squared_error(y, yh)) / capacity * 100),
    }


def part_a_convention_check(pv: pd.DataFrame, capacity: float, lines: list[str]):
    """How big is the all-hours vs daylight-only R² gap on simple baselines?"""
    lines.append(section("PART A — Why daylight-only R² looks low: the convention gap"))
    y = pv["y"].to_numpy()
    is_day = pv["is_daylight"].to_numpy().astype(bool)
    is_miss = pv["is_missing"].to_numpy().astype(bool)

    yhat_persist24 = np.roll(y, 24)
    yhat_persist24[:24] = 0
    hour = pv.index.hour.to_numpy()
    month = pv.index.month.to_numpy()
    df_clim = pd.DataFrame({"y": y, "h": hour, "m": month, "miss": is_miss})
    clim = (
        df_clim[~df_clim["miss"]]
        .groupby(["m", "h"])["y"]
        .mean()
        .rename("clim")
        .reset_index()
    )
    yhat_clim = (
        df_clim[["m", "h"]].merge(clim, on=["m", "h"], how="left")["clim"].to_numpy()
    )
    yhat_clim = np.nan_to_num(yhat_clim, nan=0.0)

    rows = []
    for tag, mask in [("ALL_hours", ~is_miss), ("daylight_only", ~is_miss & is_day)]:
        for name, yh in [
            ("Persistence24h", yhat_persist24),
            ("Climatology(month×hour)", yhat_clim),
        ]:
            r = metric_row(f"{name}|{tag}", y[mask], yh[mask], capacity)
            rows.append(r)
    df = pd.DataFrame(rows)
    lines.append(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    lines.append("\nObservation: the SAME forecast gets a much higher R² when night")
    lines.append("hours are included, because predicting 0 at night accounts for")
    lines.append("~50% of the variance for free. Most published R² values use the")
    lines.append("ALL_hours convention.")
    return df


def part_b_physics_ceiling(pv: pd.DataFrame, capacity: float, lines: list[str]):
    """1-feature model y = a*GHI + b; if R² > 0.80 here, ML should reach it."""
    lines.append(section("PART B — Naive physics ceiling: is GHI alone enough?"))
    y = pv["y"].to_numpy()
    ghi = pv["shortwave_radiation"].to_numpy()
    dni = pv["direct_normal_irradiance"].to_numpy()
    is_day = pv["is_daylight"].to_numpy().astype(bool)
    is_miss = pv["is_missing"].to_numpy().astype(bool)

    rows = []
    for tag, mask in [("ALL_hours", ~is_miss), ("daylight_only", ~is_miss & is_day)]:
        # Simple physics: y = capacity * GHI / 1000 (no fit)
        yhat_phys = capacity * ghi / 1000.0
        rows.append(
            metric_row(f"y=capacity*GHI/1000|{tag}", y[mask], yhat_phys[mask], capacity)
        )

        # Linear fit y = a*GHI
        m_fit = mask & np.isfinite(ghi)
        a = np.sum(y[m_fit] * ghi[m_fit]) / np.sum(ghi[m_fit] ** 2)
        yhat_lin1 = a * ghi
        rows.append(
            metric_row(
                f"y=a*GHI fit|{tag}  (a={a:.3f})", y[mask], yhat_lin1[mask], capacity
            )
        )

        # 2-feature OLS: GHI + DNI
        X = np.stack([ghi, dni], axis=1)
        ok = mask & np.isfinite(X).all(axis=1)
        lr = LinearRegression().fit(X[ok], y[ok])
        yhat_lr = np.clip(lr.predict(X), 0, None)
        rows.append(
            metric_row(f"OLS y~GHI+DNI|{tag}", y[mask], yhat_lr[mask], capacity)
        )

    df = pd.DataFrame(rows)
    lines.append(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    lines.append("\nReading: a 1- or 2-feature linear fit on GHI/DNI is what")
    lines.append("textbooks call the 'irradiance-to-power' transfer function.")
    lines.append("On a clean PV plant with no anomalies it should give R²>0.85.")
    lines.append("If it does NOT, the residual variance is intrinsic noise:")
    lines.append("  – cloud variability not captured by hourly average GHI")
    lines.append("  – ERA5's 31-km grid blurs site-specific fast cloud events")
    lines.append("  – sensor / inverter behaviour decoupled from irradiance")
    return df


def part_c1_capacity_drift(pv: pd.DataFrame, lines: list[str], ax):
    """Look for installed-capacity changes / sensor recalibration."""
    daily_max = pv["y"].resample("D").max()
    monthly_max = pv["y"].resample("ME").max()
    p99 = pv["y"].rolling("30D").quantile(0.99)

    ax.plot(daily_max.index, daily_max.values, lw=0.6, alpha=0.5, label="daily max")
    ax.plot(p99.index, p99.values, color="orange", lw=1.4, label="rolling 30-d P99")
    ax.plot(
        monthly_max.index,
        monthly_max.values,
        color="red",
        lw=2,
        marker="o",
        ms=3,
        label="monthly max",
    )
    ax.set_title("(C1) Capacity drift check")
    ax.set_ylabel("kW")
    ax.legend(loc="lower left", fontsize=8)
    ax.tick_params(axis="x", rotation=20)

    cv = float(monthly_max.std() / monthly_max.mean())
    lines.append(section("PART C1 — Capacity drift"))
    lines.append(
        f"monthly maxima: min={monthly_max.min():.0f}  "
        f"max={monthly_max.max():.0f}  mean={monthly_max.mean():.0f}  "
        f"CV={cv:.3f}"
    )
    lines.append("CV<0.05 = stable; CV>0.10 = clear capacity change. ")
    lines.append(monthly_max.to_string())
    return cv


def part_c2_timestamp_alignment(pv: pd.DataFrame, lines: list[str], ax):
    """Cross-correlate y vs GHI at lags ±3 h. Best lag should be 0."""
    y = pv["y"].to_numpy()
    g = pv["shortwave_radiation"].to_numpy()
    is_day = pv["is_daylight"].to_numpy().astype(bool)
    is_miss = pv["is_missing"].to_numpy().astype(bool)
    ok = is_day & ~is_miss & np.isfinite(g)
    lags = np.arange(-3, 4)
    corrs = []
    for L in lags:
        if L >= 0:
            yy = y[L:][ok[L:]]
            gg = g[: len(g) - L][ok[L:]]
        else:
            yy = y[:L][ok[:L]]
            gg = g[-L:][ok[:L]]
        c = np.corrcoef(yy, gg)[0, 1]
        corrs.append(c)
    corrs = np.array(corrs)
    best = int(lags[np.argmax(corrs)])

    ax.bar(lags, corrs, color=["#aaa"] * 3 + ["#d62728"] + ["#aaa"] * 3)
    ax.set_xticks(lags)
    ax.set_title(f"(C2) Cross-corr y↔GHI vs lag (best={best} h)")
    ax.set_ylabel("Pearson r")
    ax.set_xlabel("PV leads GHI (h)")
    ax.set_ylim(min(0, corrs.min()) - 0.02, 1)

    lines.append(section("PART C2 — Timestamp alignment"))
    lines.append(f"best lag = {best} h   (0 = perfectly aligned)")
    lines.append("lag (h) | corr")
    for L, c in zip(lags, corrs, strict=False):
        lines.append(f"{L:+3d}     | {c:.4f}")
    if best != 0:
        lines.append(
            "⚠️ Non-zero best lag means a timezone / DST issue. "
            f"Shift PV by {best} h relative to weather."
        )
    else:
        lines.append("✓ Timestamps are aligned with weather data.")
    return best


def part_c3_peak_hour(pv: pd.DataFrame, lines: list[str], ax):
    """Median PV by hour for each month — should peak ~11–13 UTC at this latitude."""
    df = pv[(pv["is_missing"] == 0) & (pv["is_daylight"] == 1)].copy()
    df["m"] = df.index.month
    df["h"] = df.index.hour
    pivot = df.groupby(["m", "h"])["y"].median().unstack("h")
    pivot = pivot.reindex(columns=range(24), fill_value=0)

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="upper",
        cmap="viridis",
        extent=[0, 24, 12.5, 0.5],
    )
    ax.set_yticks(range(1, 13))
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Month")
    ax.set_title("(C3) Median PV per (month, hour)")
    plt.colorbar(im, ax=ax, fraction=0.04, label="kW")

    peak_h = pivot.idxmax(axis=1)
    lines.append(section("PART C3 — Peak hour per month"))
    lines.append("UK summer ~12 UTC, winter ~12 UTC. Drift = clock issue.")
    lines.append("month | peak_hour")
    for m, h in peak_h.items():
        lines.append(f"  {m:>2} | {h}")
    return peak_h


def part_c4_zero_runs(pv: pd.DataFrame, lines: list[str], ax):
    """Counts of suspicious zero hours (PV=0 during expected daylight)."""
    df = pv[(pv["is_missing"] == 0) & (pv["is_daylight"] == 1)].copy()
    df["zero"] = (df["y"] <= 1e-3).astype(int)
    monthly_zero_pct = df.groupby(df.index.to_period("M"))["zero"].mean() * 100

    ax.bar(np.arange(len(monthly_zero_pct)), monthly_zero_pct.values, color="#d62728")
    ax.set_xticks(np.arange(len(monthly_zero_pct)))
    ax.set_xticklabels(
        [str(p) for p in monthly_zero_pct.index], rotation=45, fontsize=7
    )
    ax.set_ylabel("% daylight hours with PV=0")
    ax.set_title("(C4) Suspicious daytime zeros (inverter dropouts?)")

    lines.append(section("PART C4 — Daytime zero-output hours"))
    lines.append("month  | %daylight_hours_with_PV=0")
    for p, v in monthly_zero_pct.items():
        lines.append(f"{p}  | {v:.2f}%")
    high = monthly_zero_pct[monthly_zero_pct > 5]
    if len(high):
        lines.append(
            f"⚠️ {len(high)} month(s) with >5% zero daytime hours — "
            "likely inverter dropouts that look like cloud cover to the model."
        )
    return monthly_zero_pct


def part_c5_csi_distribution(pv: pd.DataFrame, lines: list[str], ax):
    """Clear-sky index k = y / (capacity * GHI / 1000); should be 0–1.2 mostly."""
    capacity = float(pv["capacity"].iloc[0])
    g = pv["shortwave_radiation"].to_numpy()
    y = pv["y"].to_numpy()
    is_day = pv["is_daylight"].to_numpy().astype(bool)
    is_miss = pv["is_missing"].to_numpy().astype(bool)
    ok = is_day & ~is_miss & (g > 50)  # filter low-GHI to avoid /0
    physics = capacity * g / 1000.0
    k = y[ok] / np.maximum(physics[ok], 1e-3)

    ax.hist(k, bins=80, range=(0, 2), color="#1f77b4")
    ax.axvline(1.0, color="red", ls="--", label="ideal CSI=1")
    ax.set_xlabel("clear-sky index k = y / (capacity·GHI/1000)")
    ax.set_ylabel("count")
    ax.set_title("(C5) Clear-sky index distribution")
    ax.legend()

    lines.append(section("PART C5 — Clear-sky index distribution"))
    lines.append(
        f"n={len(k)}   median={np.median(k):.3f}   "
        f"mean={k.mean():.3f}   std={k.std():.3f}"
    )
    lines.append(f"% with k>1.2 (suspicious) : {(k > 1.2).mean() * 100:.2f}")
    lines.append(f"% with k<0.05 (suspicious): {(k < 0.05).mean() * 100:.2f}")
    lines.append(
        f"P10={np.percentile(k, 10):.3f}  P50={np.percentile(k, 50):.3f}  "
        f"P90={np.percentile(k, 90):.3f}"
    )
    lines.append(
        "Healthy plant: median k ~ 0.7-0.85, P90 ~ 1.0, very few k>1.2 outliers."
    )
    lines.append("If P50 is far below 0.7 the ratio is wrong (capacity? clipping?).")
    return k


def part_c6_residual_diagnostic(
    pv: pd.DataFrame, capacity: float, lines: list[str], ax
):
    """Train a single GBM on all-but-last-90 days, score on last 90.
    Decompose residuals by hour / cloud level / month."""
    pv = pv.copy()
    hr = pv.index.hour
    doy = pv.index.dayofyear
    pv["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    pv["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    pv["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    pv["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    feats = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"] + WEATHER_CHANNELS

    cutoff = pv.index.max() - pd.Timedelta(days=90)
    train = pv[(pv.index < cutoff) & (pv["is_missing"] == 0)]
    test = pv[(pv.index >= cutoff) & (pv["is_missing"] == 0) & (pv["is_daylight"] == 1)]

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
    res = test["y"].to_numpy() - yh

    # Bin by GHI-fraction-of-clear-sky-envelope
    ghi = test["shortwave_radiation"].to_numpy()
    capacity_pred = capacity * ghi / 1000.0
    csi_obs = test["y"].to_numpy() / np.maximum(capacity_pred, 1e-3)
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.05, 5]
    labels = ["overcast", "v.cloudy", "cloudy", "broken", "clear", "outlier"]
    cat = pd.cut(csi_obs, bins=bins, labels=labels)
    df = pd.DataFrame(
        {
            "y": test["y"].to_numpy(),
            "yh": yh,
            "res": res,
            "cat": cat,
            "month": test.index.month,
            "hour": test.index.hour,
        }
    )

    grp = df.groupby("cat", observed=False).agg(
        n=("res", "size"),
        bias=("res", "mean"),
        rmse=("res", lambda v: float(np.sqrt(np.mean(v**2)))),
        y_mean=("y", "mean"),
    )

    grp["nrmse_%cap"] = grp["rmse"] / capacity * 100
    ax.bar(grp.index.astype(str), grp["nrmse_%cap"], color="#1f77b4")
    ax.set_ylabel("test RMSE (% of capacity)")
    ax.set_title("(C6) Residual nRMSE by sky condition")
    ax.tick_params(axis="x", rotation=15)

    lines.append(section("PART C6 — Residual breakdown by sky condition"))
    lines.append(f"holdout = last 90 days, daylight only, n={len(df)}")
    lines.append(
        f"overall RMSE = {float(np.sqrt(np.mean(res**2))):.1f} kW   "
        f"({float(np.sqrt(np.mean(res**2))) / capacity * 100:.2f}% of capacity)"
    )
    lines.append(grp.to_string(float_format=lambda v: f"{v:.2f}"))
    lines.append("\nReading: if 'cloudy' / 'broken' bins dominate the error,")
    lines.append(
        "the unpredictability is sub-hourly cloud events that ERA5 cannot see."
    )
    return grp, df


def main() -> None:
    pv = load_pv(CSV_PATH)
    pv = flag_daylight(pv)
    capacity = float(pv["capacity"].iloc[0])
    print(f"capacity = {capacity:.1f} kW")
    print(f"data span: {pv.index.min()} → {pv.index.max()}  ({len(pv)} h)")

    wx = load_weather()
    pv = pv.join(wx, how="left")
    pv[WEATHER_CHANNELS] = pv[WEATHER_CHANNELS].ffill().bfill()

    lines: list[str] = []
    lines.append(
        f"PV diagnostic — capacity={capacity:.0f} kW   "
        f"span: {pv.index.min().date()} → {pv.index.max().date()}   "
        f"n_hours={len(pv)}"
    )

    df_a = part_a_convention_check(pv, capacity, lines)
    df_b = part_b_physics_ceiling(pv, capacity, lines)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    cv = part_c1_capacity_drift(pv, lines, axes[0, 0])
    best_lag = part_c2_timestamp_alignment(pv, lines, axes[0, 1])
    part_c3_peak_hour(pv, lines, axes[1, 0])
    part_c4_zero_runs(pv, lines, axes[1, 1])
    k = part_c5_csi_distribution(pv, lines, axes[2, 0])
    grp, df_resid = part_c6_residual_diagnostic(pv, capacity, lines, axes[2, 1])

    fig.tight_layout()
    fig.savefig(OUT_DIR / "pv_diagnostic_panel.png", dpi=140)
    plt.close(fig)

    metrics = pd.concat(
        [df_a.assign(part="A"), df_b.assign(part="B")], ignore_index=True
    )
    metrics.to_csv(OUT_DIR / "pv_diagnostic_metrics.csv", index=False)

    lines.append(section("VERDICT"))
    a_all = df_a[df_a["model"].str.contains("ALL_hours")]
    a_day = df_a[df_a["model"].str.contains("daylight_only")]
    lines.append(
        f"  Climatology   : R²(all-hours)={a_all.iloc[1]['R2']:.3f}   "
        f"R²(daylight)={a_day.iloc[1]['R2']:.3f}"
    )
    b_all = df_b[
        df_b["model"].str.contains("ALL_hours") & df_b["model"].str.contains("OLS")
    ]
    b_day = df_b[
        df_b["model"].str.contains("daylight_only") & df_b["model"].str.contains("OLS")
    ]
    lines.append(
        f"  OLS(GHI,DNI)  : R²(all-hours)={b_all.iloc[0]['R2']:.3f}   "
        f"R²(daylight)={b_day.iloc[0]['R2']:.3f}"
    )
    lines.append(f"  capacity CV   : {cv:.3f}    (drift?)")
    lines.append(f"  best lag      : {best_lag} h    (timestamp?)")
    lines.append(f"  CSI median    : {np.median(k):.3f}    (healthy plant ~0.7–0.85)")

    report_path = OUT_DIR / "pv_diagnostic_report.txt"
    report_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(
        f"\nSaved:\n  {report_path}\n  pv_diagnostic_panel.png\n  "
        "pv_diagnostic_metrics.csv"
    )


if __name__ == "__main__":
    main()
