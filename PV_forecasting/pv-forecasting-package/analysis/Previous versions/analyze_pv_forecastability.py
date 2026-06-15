"""How predictable is PV_data.csv for a 24-h-ahead hourly forecast?

This script *does not* train anything. It measures the upper bound of
forecast performance that can be reached from the PV history alone (no
weather inputs), by running the naive baselines that every serious
PV-forecasting paper benchmarks against:

    B1. Persistence     : y_hat(t) = y(t - 24 h)
    B2. 2-day persist.  : y_hat(t) = y(t - 48 h)
    B3. Weekly persist. : y_hat(t) = y(t - 168 h)
    B4. Clim (hour)     : y_hat(t) = mean(y | hour-of-day)
    B5. Clim (m x h)    : y_hat(t) = mean(y | month, hour-of-day)
    B6. Smart persist.  : y_hat(t) = k(t-24h) * C(t) where k = y / C and
                          C is a data-driven clear-sky envelope.
    B7. Clim (doy x h)  : y_hat(t) = mean(y | day-of-year window, hour)
                          - upper bound of what climatology can do.

Metrics are reported on the full series and on daylight hours only
(night-time zeros trivially inflate R^2 / NRMSE). We also show the
autocorrelation function and power spectrum to visualise which
frequencies actually carry signal.

Outputs (saved next to the script):
    pv_forecastability_metrics.csv   - table of all baselines
    pv_forecastability_acf.png       - autocorrelation 0..200 h
    pv_forecastability_psd.png       - Welch PSD, annotated
    pv_forecastability_week.png      - worst / typical / best week examples
    pv_forecastability_scatter.png   - y vs y_hat for the best naive
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

CSV_PATH = Path(__file__).with_name("PV_data.csv")
OUT_DIR = Path(__file__).parent


def load_pv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#", skip_blank_lines=True)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["_field"] == "PPV"].copy()
    df["_time"] = pd.to_datetime(df["_time"], utc=True)
    df["_value"] = pd.to_numeric(df["_value"], errors="coerce")
    series = df.set_index("_time")["_value"].sort_index().rename("y")
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="1h")
    series = series.reindex(full_index)
    out = series.to_frame()
    out["is_missing"] = out["y"].isna().astype(int)
    out["y"] = out["y"].clip(lower=0)
    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    df["hour"] = idx.hour
    df["month"] = idx.month
    df["doy"] = idx.dayofyear
    df["dow"] = idx.dayofweek
    return df


def flag_daylight(df: pd.DataFrame, pct: float = 0.05) -> pd.DataFrame:
    """A data-driven 'daylight' mask: an hour counts as daylight if the
    median non-missing PV for that (month, hour-of-day) cell is above
    ``pct`` * (capacity)."""
    capacity = df["y"].quantile(0.999)
    med = df.groupby(["month", "hour"])["y"].transform("median")
    df["is_daylight"] = (med > pct * capacity).astype(int)
    df["capacity"] = capacity
    return df


def clear_sky_envelope(
    df: pd.DataFrame, window_days: int = 14, upper_q: float = 0.95
) -> pd.Series:
    """Data-driven clear-sky envelope C_t: rolling high quantile of PV by
    (hour-of-day) within a +/- window_days window of each day of year."""
    y = df["y"].fillna(0.0)
    doy = df.index.dayofyear.values
    hour = df.index.hour.values
    n_days = 366
    env_map = np.zeros((n_days + 1, 24))
    for h in range(24):
        mask_h = hour == h
        vals_h = y.values[mask_h]
        doy_h = doy[mask_h]
        table = np.zeros(n_days + 1)
        for d in range(1, n_days + 1):
            lo = (d - window_days) % n_days
            hi = (d + window_days) % n_days
            if lo < hi:
                sel = (doy_h >= lo) & (doy_h <= hi)
            else:
                sel = (doy_h >= lo) | (doy_h <= hi)
            if sel.any():
                table[d] = np.quantile(vals_h[sel], upper_q)
        if table.max() > 0:
            smooth = pd.Series(np.concatenate([table[-14:], table, table[:14]]))
            smooth = smooth.rolling(15, center=True, min_periods=1).mean()
            smooth = smooth.iloc[14:-14].to_numpy()
            table = smooth
        env_map[:, h] = table
    C = env_map[doy, hour]
    C = np.maximum(C, 1.0)
    return pd.Series(C, index=df.index, name="clear_sky")


@dataclass
class Metrics:
    name: str
    n: int
    r2: float
    mae: float
    rmse: float
    nmae: float
    nrmse: float
    skill_vs_pers: float

    def as_row(self) -> dict:
        return {
            "baseline": self.name,
            "n": self.n,
            "R2": round(self.r2, 4),
            "MAE": round(self.mae, 2),
            "RMSE": round(self.rmse, 2),
            "nMAE_%cap": round(self.nmae * 100, 2),
            "nRMSE_%cap": round(self.nrmse * 100, 2),
            "skill_vs_pers_%": round(self.skill_vs_pers * 100, 2),
        }


def score(
    name: str,
    y: np.ndarray,
    yhat: np.ndarray,
    mask: np.ndarray,
    capacity: float,
    rmse_ref: float | None = None,
) -> Metrics:
    m = mask & np.isfinite(y) & np.isfinite(yhat)
    y_, yhat_ = y[m], yhat[m]
    err = yhat_ - y_
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((y_ - y_.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    skill = (1 - rmse / rmse_ref) if rmse_ref and rmse_ref > 0 else 0.0
    return Metrics(
        name=name,
        n=int(m.sum()),
        r2=r2,
        mae=mae,
        rmse=rmse,
        nmae=mae / capacity,
        nrmse=rmse / capacity,
        skill_vs_pers=skill,
    )


def run_baselines(df: pd.DataFrame) -> pd.DataFrame:
    y = df["y"].to_numpy(dtype=float)
    capacity = float(df["capacity"].iloc[0])
    valid = df["is_missing"].to_numpy() == 0

    yhat_pers1 = np.roll(y, 24)
    yhat_pers1[:24] = np.nan
    yhat_pers2 = np.roll(y, 48)
    yhat_pers2[:48] = np.nan
    yhat_persW = np.roll(y, 168)
    yhat_persW[:168] = np.nan

    clim_h = df.groupby("hour")["y"].transform("mean").to_numpy()
    clim_mh = df.groupby(["month", "hour"])["y"].transform("mean").to_numpy()

    doy = df.index.dayofyear.values
    hod = df.index.hour.values
    clim_doyh = np.zeros_like(y)
    y_series = df["y"].copy()
    y_series[df["is_missing"] == 1] = np.nan
    for h in range(24):
        mh = hod == h
        doy_h = doy[mh]
        vals = y_series.values[mh]
        table = np.full(367, np.nan)
        for d in range(1, 367):
            lo = (d - 7) % 366
            hi = (d + 7) % 366
            sel = (
                (doy_h >= lo) & (doy_h <= hi)
                if lo < hi
                else (doy_h >= lo) | (doy_h <= hi)
            )
            v = vals[sel]
            v = v[~np.isnan(v)]
            if v.size:
                table[d] = v.mean()
        clim_doyh[mh] = table[doy_h]

    C = clear_sky_envelope(df)
    k = (y / C.to_numpy()).clip(0, 1.5)
    k_lag = np.roll(k, 24)
    k_lag[:24] = np.nan
    yhat_smart = k_lag * C.to_numpy()

    mask_all = valid
    mask_day = valid & (df["is_daylight"].to_numpy() == 1)

    rmse_ref_all = float(np.sqrt(np.nanmean((y[mask_all] - yhat_pers1[mask_all]) ** 2)))
    rmse_ref_day = float(np.sqrt(np.nanmean((y[mask_day] - yhat_pers1[mask_day]) ** 2)))

    rows = []
    for tag, mask, ref in [
        ("all_hours", mask_all, rmse_ref_all),
        ("daylight", mask_day, rmse_ref_day),
    ]:
        for name, yh in [
            ("B1_persistence_24h", yhat_pers1),
            ("B2_persistence_48h", yhat_pers2),
            ("B3_persistence_168h", yhat_persW),
            ("B4_clim_hour", clim_h),
            ("B5_clim_month_hour", clim_mh),
            ("B6_smart_persist_CSI", yhat_smart),
            ("B7_clim_doy_hour", clim_doyh),
        ]:
            m = score(name, y, yh, mask, capacity, rmse_ref=ref)
            rows.append({"subset": tag, **m.as_row()})
    return pd.DataFrame(rows)


def compute_acf(y: np.ndarray, mask: np.ndarray, max_lag: int = 200) -> np.ndarray:
    y = y.copy().astype(float)
    y[~mask] = np.nan
    y = y - np.nanmean(y)
    acf = np.zeros(max_lag + 1)
    var = np.nanvar(y)
    for k in range(max_lag + 1):
        a = y[:-k] if k else y
        b = y[k:] if k else y
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 100 and var > 0:
            acf[k] = np.mean(a[m] * b[m]) / var
    return acf


def plot_acf(acf: np.ndarray, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.stem(np.arange(acf.size), acf, basefmt=" ")
    for lag in (24, 48, 72, 168):
        ax.axvline(lag, color="orange", ls="--", alpha=0.7)
        ax.text(lag, 0.95, f"{lag}h", color="orange", ha="center", va="top", fontsize=9)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Autocorrelation of hourly PV (daylight + night)")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("ACF")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_psd(y: np.ndarray, out_png: Path) -> None:
    y = np.nan_to_num(y - np.nanmean(y))
    f, pxx = signal.welch(y, fs=1.0, nperseg=min(24 * 30, len(y)))
    mask = (f > 0) & (pxx > 0)
    period_h = 1.0 / f[mask]
    pxx = pxx[mask]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.loglog(period_h, pxx)
    for p, label in [(12, "12 h"), (24, "24 h"), (168, "week"), (24 * 30, "month")]:
        if period_h.min() <= p <= period_h.max():
            ax.axvline(p, color="orange", ls="--", alpha=0.5)
            ax.text(
                p,
                pxx.max() * 0.5,
                label,
                rotation=90,
                color="orange",
                fontsize=8,
                va="top",
                ha="right",
            )
    ax.set_xlabel("Period (hours)")
    ax.set_ylabel("Spectral density")
    ax.set_title("Welch power spectrum of hourly PV")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_week_examples(df: pd.DataFrame, out_png: Path) -> None:
    daily_err = (df["y"] - df["y"].shift(24)).abs().resample("1D").mean()
    best_day = daily_err.idxmin()
    worst_day = daily_err.idxmax()
    typical_day = daily_err.sub(daily_err.median()).abs().idxmin()
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharey=True)
    for ax, day, tag in zip(
        axes,
        [best_day, typical_day, worst_day],
        ["Easiest week", "Typical week", "Hardest week"],
        strict=False,
    ):
        start = pd.Timestamp(day) - pd.Timedelta(days=3)
        end = pd.Timestamp(day) + pd.Timedelta(days=4)
        seg = df.loc[start:end]
        ax.plot(seg.index, seg["y"], label="y", color="steelblue", lw=1.5)
        ax.plot(
            seg.index,
            seg["y"].shift(24),
            label="persistence (24h)",
            color="orange",
            lw=1.0,
        )
        ax.set_title(f"{tag} — centered on {day.date()}")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, out_png: Path) -> None:
    y = df["y"].to_numpy()
    yhat = df.groupby(["month", "hour"])["y"].transform("mean").to_numpy()
    mask = (df["is_missing"] == 0).to_numpy() & (df["is_daylight"] == 1).to_numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hexbin(yhat[mask], y[mask], gridsize=60, cmap="viridis", mincnt=1)
    lim = max(y[mask].max(), yhat[mask].max())
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlabel("month×hour climatology ŷ")
    ax.set_ylabel("actual y")
    ax.set_title("Climatology forecast vs actual (daylight hours)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> None:
    df = load_pv(CSV_PATH)
    df = add_calendar_features(df)
    df = flag_daylight(df)

    capacity = float(df["capacity"].iloc[0])
    n = len(df)
    n_miss = int(df["is_missing"].sum())
    n_day = int(df["is_daylight"].sum())
    print(
        f"\nSamples: {n}  | missing: {n_miss} ({n_miss / n:.1%})"
        f"  | daylight hours: {n_day} ({n_day / n:.1%})"
    )
    print(f"Capacity proxy (99.9 percentile): {capacity:.1f}")
    print(f"Daylight mean PV: {df.loc[df['is_daylight'] == 1, 'y'].mean():.1f}")
    print(f"Daylight std  PV: {df.loc[df['is_daylight'] == 1, 'y'].std():.1f}")

    acf = compute_acf(
        df["y"].to_numpy(), (df["is_missing"].to_numpy() == 0), max_lag=200
    )
    print(
        f"\nKey autocorrelations:"
        f"\n  lag 1h  : {acf[1]:+.3f}"
        f"\n  lag 24h : {acf[24]:+.3f}"
        f"\n  lag 48h : {acf[48]:+.3f}"
        f"\n  lag 72h : {acf[72]:+.3f}"
        f"\n  lag 168h: {acf[168]:+.3f}"
    )
    plot_acf(acf, OUT_DIR / "pv_forecastability_acf.png")
    plot_psd(df["y"].to_numpy(), OUT_DIR / "pv_forecastability_psd.png")

    results = run_baselines(df)
    results = results.sort_values(["subset", "nRMSE_%cap"])
    out_csv = OUT_DIR / "pv_forecastability_metrics.csv"
    results.to_csv(out_csv, index=False)
    print("\nNaive-baseline performance (sorted by nRMSE):\n")
    with pd.option_context(
        "display.max_rows", 50, "display.max_colwidth", 40, "display.width", 140
    ):
        print(results.to_string(index=False))

    plot_week_examples(df, OUT_DIR / "pv_forecastability_week.png")
    plot_scatter(df, OUT_DIR / "pv_forecastability_scatter.png")

    day = results[results["subset"] == "daylight"].iloc[0]
    all_ = results[results["subset"] == "all_hours"].iloc[0]
    print("\n=== Predictability verdict ===")
    print(
        f"  Best no-weather baseline (daylight)  : {day['baseline']}  "
        f"R²={day['R2']:.3f}  nRMSE={day['nRMSE_%cap']:.1f}% of capacity"
    )
    print(
        f"  Best no-weather baseline (all hours) : {all_['baseline']}  "
        f"R²={all_['R2']:.3f}  nRMSE={all_['nRMSE_%cap']:.1f}% of capacity"
    )
    if day["R2"] >= 0.80:
        print(
            "  -> A trained model should reach R² ≈ 0.85-0.92 day-ahead "
            "without exogenous inputs, and 0.92-0.97 with a decent NWP."
        )
    elif day["R2"] >= 0.60:
        print(
            "  -> Moderate predictability. A model can add 5-15 pp of R² "
            "over naive baselines; getting past ~0.9 will need weather "
            "features."
        )
    else:
        print(
            "  -> Predictability is low. Investigate data quality / gaps "
            "before modelling."
        )


if __name__ == "__main__":
    main()
