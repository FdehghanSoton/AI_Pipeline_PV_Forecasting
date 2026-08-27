"""Does adding weather inputs raise the 24-h-ahead PV forecast ceiling?

Pipeline
--------
1. Pull an explicitly selected Open-Meteo historical weather product for
   the PV site (cached to ``weather_cache.csv`` so we only query once).
2. Merge with ``PV_data.csv`` on the hourly timestamp.
3. Chronological 80/20 split, evaluate on daylight hours of the test set.
4. Compare:
     - climatology baseline (best history-only model from the previous
       analysis),
     - persistence baseline,
     - Ridge regression on calendar + weather,
     - HistGradientBoosting on calendar + weather.

Outputs:
    weather_cache.csv                           - weather history
    pv_weather_model_metrics.csv                - comparison table
    pv_weather_scatter.png                      - y vs y_hat on test
    pv_weather_feature_importance.png           - GBM feature importance
    pv_weather_week.png                         - test-set worst/typical/best
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import paths


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


CSV_PATH = paths.PV_CSV
WEATHER_CACHE = paths.WEATHER_DIR / "weather_cache.csv"
WEATHER_META = paths.WEATHER_DIR / "weather_cache.meta.json"
WEATHER_MODEL = os.environ.get("PV_WEATHER_MODEL", "best_match").strip().lower()
OUT_DIR = paths.results_dir()

LAT = 50.9097
LON = -1.4044
SITE_NAME = "Southampton, UK"

WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "shortwave_radiation",
    "direct_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "precipitation",
    "pressure_msl",
]


def load_pv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#", skip_blank_lines=True)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["_field"] == "PPV"].copy()
    df["_time"] = pd.to_datetime(df["_time"], utc=True)
    df["_value"] = pd.to_numeric(df["_value"], errors="coerce")
    s = df.set_index("_time")["_value"].sort_index().rename("y")
    full = pd.date_range(s.index.min(), s.index.max(), freq="1h", tz="UTC")
    s = s.reindex(full)
    out = s.to_frame()
    out["is_missing"] = out["y"].isna().astype(int)
    out["y"] = out["y"].fillna(0.0).clip(lower=0)
    return out


def fetch_weather(
    start: pd.Timestamp,
    end: pd.Timestamp,
    weather_model: str = WEATHER_MODEL,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch and cache Open-Meteo historical weather.

    Open-Meteo's default ``best_match`` dataset is not guaranteed to be pure
    ERA5. Set ``PV_WEATHER_MODEL=era5`` and remove/refresh the cache when an
    ERA5-only experiment is required. Cache metadata is stored separately so
    the data provenance is auditable.
    """
    requested_model = weather_model or "best_match"
    cached_model = None
    if WEATHER_META.exists():
        try:
            cached_model = json.loads(WEATHER_META.read_text()).get("weather_model")
        except (OSError, ValueError):
            cached_model = None

    cache_compatible = (
        requested_model == "best_match" and cached_model in (None, "best_match")
    ) or cached_model == requested_model

    if WEATHER_CACHE.exists() and not force_refresh and cache_compatible:
        cached = pd.read_csv(WEATHER_CACHE, parse_dates=["time"])
        cached["time"] = pd.to_datetime(cached["time"], utc=True)
        if (
            cached["time"].min() <= start
            and cached["time"].max() >= end
            and set(WEATHER_VARS).issubset(cached.columns)
        ):
            provenance = cached_model or "legacy cache, model not recorded"
            print(
                f"Using cached weather {WEATHER_CACHE.name} "
                f"({len(cached)} rows, model={provenance}, "
                f"{cached['time'].min()} -> {cached['time'].max()})"
            )
            return cached.set_index("time").sort_index()

    if WEATHER_CACHE.exists() and not cache_compatible:
        print(
            f"Ignoring {WEATHER_CACHE.name}: cached model={cached_model!r}, "
            f"requested model={requested_model!r}."
        )

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": ",".join(WEATHER_VARS),
        "timezone": "UTC",
    }
    if requested_model != "best_match":
        params["models"] = requested_model

    print(
        f"Querying Open-Meteo for {SITE_NAME}: "
        f"{params['start_date']} -> {params['end_date']} "
        f"(model={requested_model})..."
    )
    t0 = time.time()
    response = requests.get(url, params=params, timeout=90)
    response.raise_for_status()
    payload = response.json()
    hourly = payload["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    df.to_csv(WEATHER_CACHE, index_label="time")
    WEATHER_META.write_text(
        json.dumps(
            {
                "weather_model": requested_model,
                "latitude": LAT,
                "longitude": LON,
                "start_date": params["start_date"],
                "end_date": params["end_date"],
                "hourly_variables": WEATHER_VARS,
                "source_url": url,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"  fetched {len(df)} rows in {time.time() - t0:.1f}s, "
        f"cached to {WEATHER_CACHE.name}"
    )
    return df


def flag_daylight(df: pd.DataFrame, pct: float = 0.05) -> pd.DataFrame:
    capacity = df["y"].quantile(0.999)
    df["capacity"] = capacity
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df["doy"] = df.index.dayofyear
    df["dow"] = df.index.dayofweek
    med = df.groupby(["month", "hour"])["y"].transform("median")
    df["is_daylight"] = (med > pct * capacity).astype(int)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix. Everything here is either calendar (known
    in the future) or weather (we treat ERA5 as a proxy for a perfect NWP
    forecast; the realistic forecast would be noisier, so the numbers
    here are an *upper bound* on the benefit of weather features)."""
    df = df.copy()
    hr = df.index.hour
    doy = df.index.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    for lag in (24, 48, 168):
        df[f"y_lag{lag}"] = df["y"].shift(lag)
    return df


def chronological_split(df: pd.DataFrame, frac: float = 0.8) -> tuple:
    n = len(df)
    cut = int(n * frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def learn_climatology(train: pd.DataFrame) -> pd.DataFrame:
    """(doy-of-year window x hour-of-day) climatology, learned on train only."""
    y = train["y"].where(train["is_missing"] == 0).to_numpy()
    doy = train["doy"].to_numpy()
    hod = train["hour"].to_numpy()
    table = np.full((367, 24), np.nan)
    for h in range(24):
        mh = hod == h
        v = y[mh]
        d = doy[mh]
        for day in range(1, 367):
            lo = (day - 7) % 366
            hi = (day + 7) % 366
            sel = (d >= lo) & (d <= hi) if lo < hi else (d >= lo) | (d <= hi)
            vals = v[sel]
            vals = vals[~np.isnan(vals)]
            if vals.size:
                table[day, h] = vals.mean()
    return pd.DataFrame(table)


def apply_climatology(df: pd.DataFrame, table: pd.DataFrame) -> np.ndarray:
    doy = df["doy"].to_numpy()
    hod = df["hour"].to_numpy()
    preds = table.to_numpy()[doy, hod]
    if np.isnan(preds).any():
        col_mean = np.nanmean(table.to_numpy(), axis=0)
        nan_mask = np.isnan(preds)
        preds[nan_mask] = col_mean[hod[nan_mask]]
    return preds


@dataclass
class Result:
    name: str
    n: int
    r2: float
    mae: float
    rmse: float
    nmae_pct_cap: float
    nrmse_pct_cap: float
    skill_vs_clim_pct: float

    def row(self) -> dict:
        return {
            "model": self.name,
            "n_test": self.n,
            "R2_test": round(self.r2, 4),
            "MAE_test": round(self.mae, 1),
            "RMSE_test": round(self.rmse, 1),
            "nMAE_%cap": round(self.nmae_pct_cap, 2),
            "nRMSE_%cap": round(self.nrmse_pct_cap, 2),
            "skill_vs_climatology_%": round(self.skill_vs_clim_pct, 2),
        }


def evaluate(
    name: str,
    y: np.ndarray,
    yhat: np.ndarray,
    capacity: float,
    mask: np.ndarray,
    rmse_ref: float | None = None,
) -> Result:
    m = mask & np.isfinite(y) & np.isfinite(yhat)
    y_, yh = y[m], yhat[m]
    mae = mean_absolute_error(y_, yh)
    rmse = float(np.sqrt(mean_squared_error(y_, yh)))
    r2 = r2_score(y_, yh)
    skill = (1 - rmse / rmse_ref) * 100 if rmse_ref else 0.0
    return Result(
        name,
        int(m.sum()),
        float(r2),
        float(mae),
        rmse,
        mae / capacity * 100,
        rmse / capacity * 100,
        skill,
    )


def plot_scatter(
    y_true: np.ndarray, preds: dict, mask: np.ndarray, out_png: Path
) -> None:
    names = list(preds.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), sharey=True)
    if n == 1:
        axes = [axes]
    lim = y_true[mask].max()
    for ax, name in zip(axes, names, strict=False):
        yh = preds[name]
        m = mask & np.isfinite(y_true) & np.isfinite(yh)
        ax.hexbin(yh[m], y_true[m], gridsize=50, cmap="viridis", mincnt=1)
        ax.plot([0, lim], [0, lim], "r--", lw=1)
        ax.set_title(f"{name}\nR²={r2_score(y_true[m], yh[m]):.3f}")
        ax.set_xlabel("forecast ŷ")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    axes[0].set_ylabel("actual y (test daylight)")
    fig.suptitle("24h-ahead forecasts vs actual — test daylight hours")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_importance(
    importances: np.ndarray, feature_names: list, out_png: Path
) -> None:
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(feature_names))))
    ax.barh(np.array(feature_names)[idx][::-1], importances[idx][::-1])
    ax.set_xlabel("permutation feature importance (GBM)")
    ax.set_title("Which features carry the signal?")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_test_weeks(test: pd.DataFrame, preds: dict, out_png: Path) -> None:
    err = (test["y"] - preds["GBM_calendar+weather"]).abs()
    daily = err.resample("1D").mean()
    best = daily.idxmin()
    worst = daily.idxmax()
    typical = (daily - daily.median()).abs().idxmin()
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharey=True)
    for ax, day, tag in zip(
        axes,
        [best, typical, worst],
        ["Easiest test week", "Typical test week", "Hardest test week"],
        strict=False,
    ):
        start = pd.Timestamp(day) - pd.Timedelta(days=3)
        end = pd.Timestamp(day) + pd.Timedelta(days=4)
        seg = test.loc[start:end]
        ax.plot(seg.index, seg["y"], color="steelblue", lw=1.6, label="actual")
        for name, yhat in preds.items():
            ax.plot(
                seg.index,
                pd.Series(yhat, index=test.index).loc[seg.index],
                lw=1.0,
                label=name,
                alpha=0.85,
            )
        ax.set_title(f"{tag} — centered on {pd.Timestamp(day).date()}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def rolling_origin_backtest(
    df: pd.DataFrame,
    capacity: float,
    warmup_days: int = 120,
    step_days: int = 14,
    horizon_hours: int = 24,
) -> pd.DataFrame:
    """Walk-forward: at each origin, train on everything before (growing
    window) and score the next ``step_days`` worth of 24-h-ahead forecasts.
    Everything stays daylight-hour-masked."""
    cal_feats = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    wx_feats = WEATHER_VARS
    feats = cal_feats + wx_feats

    start_origin = df.index.min() + pd.Timedelta(days=warmup_days)
    end_origin = df.index.max() - pd.Timedelta(hours=horizon_hours)
    origins = pd.date_range(start_origin, end_origin, freq=f"{step_days}D", tz="UTC")

    all_rows = []
    for i, origin in enumerate(origins):
        train = df.loc[df.index < origin].dropna(subset=["y"])
        test_slice = df.loc[
            (df.index >= origin) & (df.index < origin + pd.Timedelta(days=step_days))
        ]
        if len(train) < 1000 or len(test_slice) == 0:
            continue

        ridge_pipe = StandardScaler().fit(train[feats])
        X_tr = ridge_pipe.transform(train[feats])
        ridge = Ridge(alpha=1.0).fit(X_tr, train["y"])
        yhat_r = np.clip(
            ridge.predict(ridge_pipe.transform(test_slice[feats])), 0, None
        )

        gbm = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_depth=8,
            min_samples_leaf=30,
            l2_regularization=0.1,
            random_state=0,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.1,
        ).fit(train[feats], train["y"])
        yhat_g = np.clip(gbm.predict(test_slice[feats]), 0, None)

        yhat_p = test_slice["y_lag24"].to_numpy()

        clim_table = learn_climatology(train)
        yhat_c = apply_climatology(test_slice, clim_table)

        for name, yh in [
            ("persistence_24h", yhat_p),
            ("climatology", yhat_c),
            ("Ridge_cal+weather", yhat_r),
            ("GBM_cal+weather", yhat_g),
        ]:
            tmp = test_slice.assign(yhat=yh, model=name, origin=origin)
            all_rows.append(
                tmp[["y", "yhat", "model", "origin", "is_daylight", "is_missing"]]
            )

        if i % 3 == 0:
            _log(
                f"  rolling origin {i + 1}/{len(origins)} "
                f"({origin.date()}) train={len(train)}"
            )

    rolled = pd.concat(all_rows)
    rolled = rolled[
        (rolled["is_daylight"] == 1)
        & (rolled["is_missing"] == 0)
        & np.isfinite(rolled["yhat"])
        & np.isfinite(rolled["y"])
    ]
    rows = []
    for name, g in rolled.groupby("model"):
        y, yh = g["y"].to_numpy(), g["yhat"].to_numpy()
        mae = mean_absolute_error(y, yh)
        rmse = float(np.sqrt(mean_squared_error(y, yh)))
        r2 = r2_score(y, yh)
        rows.append(
            {
                "model": name,
                "n_test": len(g),
                "R2": round(r2, 4),
                "MAE": round(mae, 1),
                "RMSE": round(rmse, 1),
                "nMAE_%cap": round(mae / capacity * 100, 2),
                "nRMSE_%cap": round(rmse / capacity * 100, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("nRMSE_%cap")


def main() -> None:
    pv = load_pv(CSV_PATH)
    pv = flag_daylight(pv)
    capacity = float(pv["capacity"].iloc[0])

    start = pv.index.min().tz_convert("UTC").normalize()
    end = pv.index.max().tz_convert("UTC").normalize()
    weather = fetch_weather(start, end)

    df = pv.join(weather, how="left")
    n_weather_missing = df[WEATHER_VARS].isna().any(axis=1).sum()
    print(
        f"\nJoined PV + weather: {len(df)} rows, "
        f"{n_weather_missing} rows with any missing weather value"
    )
    df[WEATHER_VARS] = df[WEATHER_VARS].ffill().bfill()
    df = add_features(df)

    train, test = chronological_split(df, frac=0.8)
    print(f"\nSplit: train {train.index[0]} -> {train.index[-1]}  ({len(train)} rows)")
    print(f"       test  {test.index[0]} -> {test.index[-1]}  ({len(test)} rows)")

    y_test = test["y"].to_numpy(dtype=float)
    mask_day = ((test["is_daylight"] == 1) & (test["is_missing"] == 0)).to_numpy()
    _log(f"daylight-valid test samples: {mask_day.sum()}")

    _log("learning climatology table...")
    clim_table = learn_climatology(train)
    _log(f"  table NaN cells: {int(clim_table.isna().sum().sum())}/{clim_table.size}")
    yhat_clim = apply_climatology(test, clim_table)
    _log(f"  clim predictions NaN: {int(np.isnan(yhat_clim).sum())}")

    yhat_pers = test["y_lag24"].to_numpy()
    _log(f"persistence NaN: {int(np.isnan(yhat_pers).sum())}")

    cal_feats = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    wx_feats = WEATHER_VARS

    train_fit = train.dropna(subset=["y"] + cal_feats).copy()
    _log(f"train_fit rows: {len(train_fit)} (cal+weather model)")

    _log("fitting Ridge...")
    X_ridge_feats = cal_feats + wx_feats
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train_fit[X_ridge_feats])
    ridge = Ridge(alpha=1.0).fit(X_tr, train_fit["y"])
    X_te = scaler.transform(test[X_ridge_feats])
    yhat_ridge = np.clip(ridge.predict(X_te), 0, None)
    _log("  Ridge done")

    _log("fitting HistGBM (calendar + weather)...")
    gbm_feats = cal_feats + wx_feats
    gbm = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.05,
        max_depth=8,
        min_samples_leaf=40,
        l2_regularization=0.1,
        random_state=0,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.1,
    ).fit(train_fit[gbm_feats], train_fit["y"])
    yhat_gbm = np.clip(gbm.predict(test[gbm_feats]), 0, None)
    _log(f"  GBM+weather done (trees={gbm.n_iter_})")

    _log("fitting HistGBM (calendar only)...")
    gbm2_feats = cal_feats
    train_cal = train_fit.dropna(subset=["y"]).copy()
    gbm_cal_only = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=0,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.1,
    ).fit(train_cal[gbm2_feats], train_cal["y"])
    yhat_gbm_cal = np.clip(gbm_cal_only.predict(test[gbm2_feats]), 0, None)
    _log(f"  GBM-cal-only done (trees={gbm_cal_only.n_iter_})")

    _log("scoring models...")
    _log(
        f"y_test daylight rows: {mask_day.sum()}, "
        f"NaN in y_test[mask]: {int(np.isnan(y_test[mask_day]).sum())}, "
        f"NaN in yhat_clim[mask]: {int(np.isnan(yhat_clim[mask_day]).sum())}"
    )
    _score_mask = mask_day & np.isfinite(y_test) & np.isfinite(yhat_clim)
    rmse_clim = float(
        np.sqrt(mean_squared_error(y_test[_score_mask], yhat_clim[_score_mask]))
    )
    _log(f"  rmse_clim = {rmse_clim:.2f}")

    results = [
        evaluate("persistence_24h", y_test, yhat_pers, capacity, mask_day, rmse_clim),
        evaluate(
            "climatology_doy_x_hour", y_test, yhat_clim, capacity, mask_day, rmse_clim
        ),
        evaluate(
            "GBM_calendar_only", y_test, yhat_gbm_cal, capacity, mask_day, rmse_clim
        ),
        evaluate(
            "Ridge_calendar+weather", y_test, yhat_ridge, capacity, mask_day, rmse_clim
        ),
        evaluate(
            "GBM_calendar+weather", y_test, yhat_gbm, capacity, mask_day, rmse_clim
        ),
    ]
    table = pd.DataFrame([r.row() for r in results])
    table = table.sort_values("nRMSE_%cap")
    print("\n=== Test-set metrics (daylight only) ===\n")
    with pd.option_context("display.max_colwidth", 40, "display.width", 130):
        print(table.to_string(index=False))
    table.to_csv(OUT_DIR / "pv_weather_model_metrics.csv", index=False)

    preds = {
        "persistence_24h": yhat_pers,
        "climatology": yhat_clim,
        "Ridge_calendar+weather": yhat_ridge,
        "GBM_calendar+weather": yhat_gbm,
    }
    plot_scatter(y_test, preds, mask_day, OUT_DIR / "pv_weather_scatter.png")

    try:
        from sklearn.inspection import permutation_importance

        rng = np.random.default_rng(0)
        test_fit = test.dropna(subset=["y"])
        mask_pi = (
            (test_fit["is_daylight"] == 1) & (test_fit["is_missing"] == 0)
        ).to_numpy()
        idx_pi = np.where(mask_pi)[0]
        if idx_pi.size > 800:
            idx_pi = rng.choice(idx_pi, 800, replace=False)
        _log(f"permutation importance on {len(idx_pi)} rows (n_repeats=3)...")
        pi = permutation_importance(
            gbm,
            test_fit[gbm_feats].iloc[idx_pi],
            test_fit["y"].iloc[idx_pi],
            n_repeats=3,
            random_state=0,
            n_jobs=1,
        )
        plot_importance(
            pi.importances_mean,
            gbm_feats,
            OUT_DIR / "pv_weather_feature_importance.png",
        )
        _log("  feature importance done")
    except Exception as exc:
        _log(f"[feature importance skipped: {exc}]")

    plot_test_weeks(test, preds, OUT_DIR / "pv_weather_week.png")

    best = results[int(np.argmin([r.nrmse_pct_cap for r in results]))]
    clim = [r for r in results if r.name == "climatology_doy_x_hour"][0]
    print("\n=== Single-split verdict (20% tail as test) ===")
    print(
        f"  climatology (no weather) : R²={clim.r2:.3f}  "
        f"nRMSE={clim.nrmse_pct_cap:.1f}% of capacity"
    )
    print(
        f"  best model               : {best.name}  R²={best.r2:.3f}  "
        f"nRMSE={best.nrmse_pct_cap:.1f}% of capacity  "
        f"(skill vs clim: {best.skill_vs_clim_pct:+.1f}%)"
    )

    _log("\nrunning rolling-origin backtest across the full year...")
    backtest = rolling_origin_backtest(df, capacity)
    print(
        "\n=== Rolling-origin backtest (representative 24h-ahead "
        "performance across the full year, daylight only) ===\n"
    )
    with pd.option_context("display.max_colwidth", 40, "display.width", 130):
        print(backtest.to_string(index=False))
    backtest.to_csv(OUT_DIR / "pv_weather_rolling_backtest.csv", index=False)
    best_row = backtest.iloc[0]
    clim_row = backtest[backtest["model"] == "climatology"].iloc[0]
    pers_row = backtest[backtest["model"] == "persistence_24h"].iloc[0]
    print(
        f"\n  Rolling climatology  : R²={clim_row['R2']:.3f}  "
        f"nRMSE={clim_row['nRMSE_%cap']:.1f}%"
    )
    print(
        f"  Rolling persistence  : R²={pers_row['R2']:.3f}  "
        f"nRMSE={pers_row['nRMSE_%cap']:.1f}%"
    )
    print(
        f"  Rolling best model   : {best_row['model']}  "
        f"R²={best_row['R2']:.3f}  nRMSE={best_row['nRMSE_%cap']:.1f}%"
    )


if __name__ == "__main__":
    main()
