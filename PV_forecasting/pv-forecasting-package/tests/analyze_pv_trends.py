"""Analyse PV_data.csv with the lgtd (Local-Global Trend Decomposition) library.

Produces:
    - Printed diagnostics (basic stats, gaps, trend/seasonal strength, detected periods)
    - pv_trend_hourly.png : 4-panel decomposition of the hourly series
    - pv_trend_daily.png  : 4-panel decomposition of the daily-aggregated series
    - pv_monthly_boxplot.png : monthly distribution (helps read seasonality)

Run:  python analyze_pv_trends.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lgtd import lgtd

CSV_PATH = Path(__file__).with_name("PV_data.csv")
OUT_DIR = Path(__file__).parent


def load_pv_series(path: Path) -> pd.Series:
    """Load InfluxDB-Flux-formatted CSV and return an hourly pd.Series (kW)."""
    df = pd.read_csv(path, comment="#", skip_blank_lines=True)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["_field"] == "PPV"].copy()
    df["_time"] = pd.to_datetime(df["_time"], utc=True)
    df["_value"] = pd.to_numeric(df["_value"], errors="coerce")
    series = df.set_index("_time")["_value"].sort_index().rename("PPV")
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="1h")
    series = series.reindex(full_index)
    return series


def strength(component: np.ndarray, residual: np.ndarray) -> float:
    """Hyndman & Athanasopoulos trend/seasonal strength in [0, 1]."""
    denom = np.var(component + residual)
    if denom <= 0:
        return 0.0
    return float(max(0.0, 1.0 - np.var(residual) / denom))


def describe_series(tag: str, s: pd.Series) -> None:
    print(f"\n=== {tag} ===")
    print(f"  samples      : {len(s)}")
    print(f"  missing      : {int(s.isna().sum())}  ({s.isna().mean():.2%})")
    print(f"  zero values  : {int((s == 0).sum())}  ({(s == 0).mean():.2%})")
    print(f"  mean / std   : {s.mean():.2f}  /  {s.std():.2f}")
    print(f"  min / max    : {s.min():.2f}  /  {s.max():.2f}")
    print(f"  range        : {s.index.min()}  ->  {s.index.max()}")


def decompose_and_report(
    tag: str, values: np.ndarray, index: pd.DatetimeIndex, out_png: Path
) -> dict:
    print(f"\n>>> lgtd decomposition: {tag}")
    model = lgtd(trend_selection="auto", verbose=True)
    result = model.fit_transform(values)

    trend_strength = strength(result.trend, result.residual)
    seas_strength = strength(result.seasonal, result.residual)

    slope_per_step = np.polyfit(np.arange(len(result.trend)), result.trend, 1)[0]
    total_trend_change = result.trend[-1] - result.trend[0]

    print(f"  detected periods   : {result.detected_periods}")
    print(f"  trend_info         : {result.trend_info}")
    print(f"  trend strength F_T : {trend_strength:.3f}  (0 = none, 1 = strong)")
    print(f"  season strength F_S: {seas_strength:.3f}")
    print(f"  linear slope       : {slope_per_step:+.4f}  per step")
    print(
        f"  total trend change : {total_trend_change:+.2f}  "
        f"(series mean {values.mean():.2f})"
    )
    print(f"  residual std       : {np.std(result.residual):.2f}")
    print(
        f"  signal-to-noise    : "
        f"{np.std(result.trend + result.seasonal) / max(np.std(result.residual), 1e-9):.2f}"
    )

    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(index, values, lw=0.6, color="steelblue")
    axes[0].set_ylabel("Observed")
    axes[0].set_title(f"lgtd decomposition — {tag}")
    axes[1].plot(index, result.trend, lw=1.4, color="darkorange")
    axes[1].set_ylabel("Trend")
    axes[2].plot(index, result.seasonal, lw=0.6, color="seagreen")
    axes[2].set_ylabel("Seasonal")
    axes[3].plot(index, result.residual, lw=0.5, color="gray")
    axes[3].axhline(0, color="k", lw=0.5)
    axes[3].set_ylabel("Residual")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  saved plot -> {out_png.name}")

    return {
        "trend_strength": trend_strength,
        "seasonal_strength": seas_strength,
        "slope_per_step": slope_per_step,
        "total_trend_change": total_trend_change,
        "detected_periods": result.detected_periods,
        "trend_info": result.trend_info,
    }


def monthly_boxplot(series: pd.Series, out_png: Path) -> None:
    s = series.dropna()
    months = s.index.to_period("M")
    groups = [s[months == m].values for m in sorted(months.unique())]
    labels = [str(m) for m in sorted(months.unique())]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.boxplot(groups, labels=labels, showfliers=False)
    ax.set_title("PV power distribution per month (daylight hours still included)")
    ax.set_ylabel("PPV")
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  saved plot -> {out_png.name}")


def verdict(hourly: dict, daily: dict) -> None:
    print("\n=== Verdict ===")
    ft_h = hourly["trend_strength"]
    fs_h = hourly["seasonal_strength"]
    ft_d = daily["trend_strength"]
    fs_d = daily["seasonal_strength"]

    def label(x: float) -> str:
        if x < 0.15:
            return "negligible"
        if x < 0.40:
            return "weak"
        if x < 0.65:
            return "moderate"
        return "strong"

    print(
        f"  hourly  -> trend {ft_h:.2f} ({label(ft_h)}),  "
        f"seasonality {fs_h:.2f} ({label(fs_h)})"
    )
    print(
        f"  daily   -> trend {ft_d:.2f} ({label(ft_d)}),  "
        f"seasonality {fs_d:.2f} ({label(fs_d)})"
    )

    if fs_h >= 0.4 or fs_d >= 0.4:
        print("  => Data has a clear *seasonal/diurnal structure* (not pure noise).")
    else:
        print("  => Seasonality is weak; the signal is dominated by noise.")

    if ft_d >= 0.4:
        print("  => There is a meaningful long-term trend in the daily aggregate.")
    elif ft_d >= 0.15:
        print("  => Daily trend is weak but present.")
    else:
        print("  => No meaningful long-term trend – level is approximately stationary.")


def main() -> None:
    series = load_pv_series(CSV_PATH)
    describe_series("Hourly PPV", series)

    hourly_filled = series.fillna(0.0)
    hourly_stats = decompose_and_report(
        tag="hourly PPV",
        values=hourly_filled.to_numpy(dtype=float),
        index=hourly_filled.index,
        out_png=OUT_DIR / "pv_trend_hourly.png",
    )

    daily = series.resample("1D").sum(min_count=12).dropna()
    describe_series("Daily energy (kWh-equiv, sum of PPV)", daily)
    daily_stats = decompose_and_report(
        tag="daily aggregated PPV",
        values=daily.to_numpy(dtype=float),
        index=daily.index,
        out_png=OUT_DIR / "pv_trend_daily.png",
    )

    monthly_boxplot(series, OUT_DIR / "pv_monthly_boxplot.png")
    verdict(hourly_stats, daily_stats)


if __name__ == "__main__":
    main()
