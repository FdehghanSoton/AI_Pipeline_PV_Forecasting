"""V2 forecaster — applies the empirically selected -1 h PV timestamp shift
and adds solar-position features.

Key changes vs analyze_pv_ensemble.py:
  1. PV timestamps are shifted -1 h in load_pv_v2 because diagnostics on
     this dataset show stronger contemporaneous alignment after the shift.
     The physical explanation must be checked against the inverter export
     convention and the selected Open-Meteo weather product metadata.
  2. Solar position features replace the hour/doy sin/cos pair:
        zenith, elevation, azimuth, airmass, cos_zenith, dni_extra,
        clearness_index = GHI / GHI_TOA_horizontal
     These are the physical inputs of the irradiance-to-power transfer.
  3. Lagged weather (t-1 and t+1 hour) is added to capture cloud motion.
  4. Reports R² on BOTH conventions (all-hours + daylight) at every fold.
  5. A clear-sky-index target is also evaluated as an alternative.

Run with `python analyze_pv_v2.py`.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import paths
from analyze_pv_cnn2d import WEATHER_CHANNELS, flag_daylight, load_weather

CSV_PATH = paths.PV_CSV
OUT_DIR = paths.results_dir()
LAT, LON, ALT = 50.91, -1.40, 30  # Southampton, UK
TILT, AZIMUTH = (
    30.0,
    180.0,
)  # Assumed fixed-tilt orientation; replace with verified plant metadata


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_pv_v2(path: Path, time_shift_hours: int = -1) -> pd.DataFrame:
    """PV loader with the time-label correction.
    PV(t) is relabelled to PV(t + time_shift_hours)."""
    df = pd.read_csv(path, comment="#", skip_blank_lines=True)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["_field"] == "PPV"].copy()
    df["_time"] = pd.to_datetime(df["_time"], utc=True)
    df["_value"] = pd.to_numeric(df["_value"], errors="coerce")
    s = df.set_index("_time")["_value"].sort_index().rename("y")
    if time_shift_hours != 0:
        s.index = s.index + pd.Timedelta(hours=time_shift_hours)
    full = pd.date_range(s.index.min(), s.index.max(), freq="1h", tz="UTC")
    s = s.reindex(full)
    out = s.to_frame()
    out["is_missing"] = out["y"].isna().astype(int)
    out["y"] = out["y"].fillna(0.0).clip(lower=0)
    return out


def flag_daylight_geometric(
    df: pd.DataFrame,
    pct: float = 0.05,
    elevation_deg: float = 5.0,
    lat: float = LAT,
    lon: float = LON,
    alt: float = ALT,
) -> pd.DataFrame:
    """Leakage-free daylight mask defined from solar elevation.

    Unlike :func:`analyze_pv_cnn2d.flag_daylight`, which thresholds the
    per-(month, hour) median of observed PV (and therefore touches held-out
    labels when computed on the full series), this rule uses only the solar
    geometry: an hour is daylight if the solar elevation exceeds
    ``elevation_deg``. The empirical capacity is still recorded for use as a
    fixed normalised-metric denominator, but it does not enter the mask.
    """
    capacity = df["y"].quantile(0.999)
    df["capacity"] = capacity
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    if "sun_elev" in df.columns:
        elevation = df["sun_elev"].to_numpy()
    else:
        sp = pvlib.solarposition.get_solarposition(df.index, lat, lon, altitude=alt)
        elevation = sp["elevation"].to_numpy()
    df["is_daylight"] = (elevation > elevation_deg).astype(int)
    return df


def add_solar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Solar zenith / elevation / azimuth / airmass + clearness index.
    Uses pvlib's NREL SPA — accurate to <0.01°."""
    sp = pvlib.solarposition.get_solarposition(df.index, LAT, LON, altitude=ALT)
    df["sun_zenith"] = sp["zenith"].to_numpy()
    df["sun_elev"] = sp["elevation"].to_numpy()
    df["sun_az"] = sp["azimuth"].to_numpy()
    df["cos_zenith"] = np.cos(np.radians(df["sun_zenith"]))
    df["airmass"] = pvlib.atmosphere.get_relative_airmass(df["sun_zenith"]).fillna(40.0)

    dni_extra = pvlib.irradiance.get_extra_radiation(df.index)
    if isinstance(dni_extra, pd.Series):
        dni_extra = dni_extra.to_numpy()
    df["dni_extra"] = dni_extra
    ghi_toa_horizontal = dni_extra * np.maximum(df["cos_zenith"], 0.0)
    df["ghi_toa_horiz"] = ghi_toa_horizontal
    df["clearness_kt"] = np.where(
        ghi_toa_horizontal > 50,
        df["shortwave_radiation"] / np.maximum(ghi_toa_horizontal, 1.0),
        0.0,
    )
    df["clearness_kt"] = df["clearness_kt"].clip(0, 1.5)

    aoi = pvlib.irradiance.aoi(TILT, AZIMUTH, df["sun_zenith"], df["sun_az"])
    df["aoi"] = aoi.fillna(90.0)
    df["cos_aoi"] = np.cos(np.radians(df["aoi"])).clip(0, None)

    poa_global = pvlib.irradiance.get_total_irradiance(
        TILT,
        AZIMUTH,
        df["sun_zenith"],
        df["sun_az"],
        dni=df["direct_normal_irradiance"],
        ghi=df["shortwave_radiation"],
        dhi=np.maximum(
            df["shortwave_radiation"]
            - df["direct_normal_irradiance"] * np.maximum(df["cos_zenith"], 0),
            0,
        ),
    )
    df["poa_global"] = poa_global["poa_global"].fillna(0.0)
    return df


def add_weather_lags(df: pd.DataFrame) -> pd.DataFrame:
    for c in [
        "shortwave_radiation",
        "direct_normal_irradiance",
        "cloud_cover",
        "poa_global",
        "clearness_kt",
    ]:
        df[f"{c}_lag1"] = df[c].shift(1)
        df[f"{c}_lead1"] = df[c].shift(-1)
        df[f"{c}_roll3"] = df[c].rolling(3, center=True, min_periods=1).mean()
    df = df.ffill().bfill()
    return df


def metric_set(y: np.ndarray, yh: np.ndarray, capacity: float) -> dict:
    m = np.isfinite(y) & np.isfinite(yh)
    y, yh = y[m], yh[m]
    if len(y) < 5 or np.var(y) < 1e-9:
        return dict(
            R2=np.nan,
            MAE=np.nan,
            RMSE=np.nan,
            nMAE_pct=np.nan,
            nRMSE_pct=np.nan,
            n=len(y),
        )
    return dict(
        n=int(len(y)),
        R2=float(r2_score(y, yh)),
        MAE=float(mean_absolute_error(y, yh)),
        RMSE=float(np.sqrt(mean_squared_error(y, yh))),
        nMAE_pct=float(mean_absolute_error(y, yh) / capacity * 100),
        nRMSE_pct=float(np.sqrt(mean_squared_error(y, yh)) / capacity * 100),
    )


@dataclass
class FoldResult:
    origin: pd.Timestamp
    n_train: int
    n_val: int
    n_test: int
    test_idx: pd.DatetimeIndex
    y_test: np.ndarray
    is_day_test: np.ndarray
    is_miss_test: np.ndarray
    preds: dict[str, np.ndarray]


def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    base_wx = WEATHER_CHANNELS.copy()
    solar_feats = [
        "sun_zenith",
        "sun_elev",
        "cos_zenith",
        "airmass",
        "dni_extra",
        "ghi_toa_horiz",
        "clearness_kt",
        "aoi",
        "cos_aoi",
        "poa_global",
    ]
    lag_feats = []
    for c in [
        "shortwave_radiation",
        "direct_normal_irradiance",
        "cloud_cover",
        "poa_global",
        "clearness_kt",
    ]:
        lag_feats += [f"{c}_lag1", f"{c}_lead1", f"{c}_roll3"]
    cal_feats = []
    df = df.copy()
    doy = df.index.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    cal_feats += ["doy_sin", "doy_cos"]
    feats = base_wx + solar_feats + lag_feats + cal_feats
    return df, feats


def rolling_backtest(
    df: pd.DataFrame,
    capacity: float,
    feats: list[str],
    n_folds: int = 4,
    first_test_days: int = 120,
) -> list[FoldResult]:
    days = df.index.normalize().unique()
    n_days = len(days)
    fold_len = (n_days - first_test_days) // n_folds
    folds: list[FoldResult] = []

    for fold in range(n_folds):
        test_start_day = days[first_test_days + fold * fold_len]
        if fold < n_folds - 1:
            test_end_day = days[first_test_days + (fold + 1) * fold_len]
        else:
            test_end_day = days[-1] + pd.Timedelta(days=1)

        train_mask = df.index < test_start_day
        test_mask = (df.index >= test_start_day) & (df.index < test_end_day)
        train_full = df[train_mask & (df["is_missing"] == 0)]
        if len(train_full) < 200:
            continue
        val_size = max(7 * 24, int(0.15 * len(train_full)))
        train = train_full.iloc[:-val_size]
        val = train_full.iloc[-val_size:]
        test = df[test_mask]

        _log(
            f"fold {fold + 1}/{n_folds}  origin={test_start_day.date()}  "
            f"train={len(train)}  val={len(val)}  test={len(test)}"
        )

        X_tr = train[feats].to_numpy()
        y_tr = train["y"].to_numpy()
        X_val = val[feats].to_numpy()
        y_val = val["y"].to_numpy()
        X_te = test[feats].to_numpy()

        scaler = StandardScaler().fit(X_tr)
        ridge = Ridge(alpha=1.0).fit(scaler.transform(X_tr), y_tr)
        ridge_val = np.clip(ridge.predict(scaler.transform(X_val)), 0, None)
        ridge_te = np.clip(ridge.predict(scaler.transform(X_te)), 0, None)

        gbm = HistGradientBoostingRegressor(
            max_iter=600,
            learning_rate=0.05,
            max_depth=8,
            min_samples_leaf=30,
            l2_regularization=0.1,
            random_state=0,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1,
        )
        gbm.fit(X_tr, y_tr)
        gbm_val = np.clip(gbm.predict(X_val), 0, None)
        gbm_te = np.clip(gbm.predict(X_te), 0, None)

        # Clear-sky-index regressor: y/(cap*POA/1000) target, then back-transform
        poa_tr = train["poa_global"].to_numpy()
        poa_val = val["poa_global"].to_numpy()
        poa_te = test["poa_global"].to_numpy()
        denom_tr = np.maximum(capacity * poa_tr / 1000.0, 1.0)
        denom_val = np.maximum(capacity * poa_val / 1000.0, 1.0)
        denom_te = np.maximum(capacity * poa_te / 1000.0, 1.0)
        kt_target = np.clip(y_tr / denom_tr, 0, 1.5)
        ok_kt = poa_tr > 50  # only fit during daylight
        gbm_kt = HistGradientBoostingRegressor(
            max_iter=600,
            learning_rate=0.05,
            max_depth=8,
            min_samples_leaf=30,
            l2_regularization=0.1,
            random_state=0,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1,
        )
        if ok_kt.sum() > 200:
            gbm_kt.fit(X_tr[ok_kt], kt_target[ok_kt])
            kt_val_pred = np.clip(gbm_kt.predict(X_val), 0, 1.5)
            kt_te_pred = np.clip(gbm_kt.predict(X_te), 0, 1.5)
            kt_val_pred[poa_val < 50] = 0
            kt_te_pred[poa_te < 50] = 0
            kt_te_yhat = kt_te_pred * denom_te
            kt_val_yhat = kt_val_pred * denom_val
        else:
            kt_te_yhat = gbm_te
            kt_val_yhat = gbm_val

        is_day_val = val["is_daylight"].to_numpy().astype(bool)
        is_miss_val = val["is_missing"].to_numpy().astype(bool)
        val_mask_eval = is_day_val & ~is_miss_val
        val_preds = {"Ridge": ridge_val, "GBM": gbm_val, "GBM_kt": kt_val_yhat}
        ensembles_test = build_ensembles(
            val_preds,
            y_val,
            val_mask_eval,
            {"Ridge": ridge_te, "GBM": gbm_te, "GBM_kt": kt_te_yhat},
        )

        preds = {
            "Ridge": ridge_te,
            "GBM": gbm_te,
            "GBM_kt": kt_te_yhat,
            **ensembles_test,
        }

        folds.append(
            FoldResult(
                origin=test_start_day,
                n_train=len(train),
                n_val=len(val),
                n_test=len(test),
                test_idx=test.index,
                y_test=test["y"].to_numpy(),
                is_day_test=test["is_daylight"].to_numpy().astype(bool),
                is_miss_test=test["is_missing"].to_numpy().astype(bool),
                preds=preds,
            )
        )
    return folds


def build_ensembles(val_preds, y_val, val_mask, test_preds):
    """Fit fusion weights on held-out validation rows and apply them to test.

    The ``NNLSStack`` member solves an unconstrained non-negative least-squares
    problem on the validation predictions and then renormalises the resulting
    weights to sum to one. The renormalisation step means the reported weights
    form a convex combination, but they are not the exact solution of a
    least-squares problem with a sum-to-one equality constraint imposed during
    optimisation. The paper text is worded to reflect this precisely.
    """
    names = list(val_preds.keys())
    V = np.stack([val_preds[n] for n in names], axis=1)
    T = np.stack([test_preds[n] for n in names], axis=1)
    y_m, V_m = y_val[val_mask], V[val_mask]

    out = {"Mean": T.mean(axis=1)}
    rmses = {
        n: float(np.sqrt(mean_squared_error(y_m, V_m[:, i])))
        for i, n in enumerate(names)
    }
    inv = np.array([1.0 / rmses[n] for n in names])
    inv /= inv.sum()
    out["InvRMSE"] = T @ inv
    try:
        ridge = Ridge(alpha=1.0, fit_intercept=True).fit(V_m, y_m)
        out["RidgeStack"] = np.clip(ridge.predict(T), 0, None)
    except Exception:
        out["RidgeStack"] = out["Mean"]
    try:
        w, _ = nnls(V_m, y_m)
        if w.sum() > 1e-6:
            w = w / w.sum()
        else:
            w = np.ones(len(names)) / len(names)
        out["NNLSStack"] = T @ w
    except Exception:
        out["NNLSStack"] = out["Mean"]
    best = min(names, key=lambda n: rmses[n])
    out["BestSingleByVal"] = test_preds[best]
    return out


def aggregate(folds: list[FoldResult], capacity: float) -> pd.DataFrame:
    """Score each model on (a) all hours and (b) daylight only, across folds."""
    names = list(folds[0].preds.keys())
    all_y = np.concatenate([f.y_test[~f.is_miss_test] for f in folds])
    all_day = np.concatenate([f.is_day_test[~f.is_miss_test] for f in folds])
    rows = []
    for name in names:
        all_yh = np.concatenate([f.preds[name][~f.is_miss_test] for f in folds])
        for tag, mask in [
            ("ALL", np.ones(len(all_y), dtype=bool)),
            ("daylight", all_day),
        ]:
            m = metric_set(all_y[mask], all_yh[mask], capacity)
            rows.append({"model": name, "subset": tag, **m})
    return pd.DataFrame(rows)


def per_month_scores(
    folds: list[FoldResult], capacity: float, model: str
) -> pd.DataFrame:
    """Pool predictions across folds first, then group by month so each
    (month, subset) appears exactly once."""
    rows = []
    chunks = []
    for f in folds:
        yh = f.preds[model]
        ok = ~f.is_miss_test
        chunks.append(
            pd.DataFrame(
                {"y": f.y_test[ok], "yh": yh[ok], "is_day": f.is_day_test[ok]},
                index=f.test_idx[ok],
            )
        )
    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    months = pd.PeriodIndex(df.index.tz_convert("UTC").tz_localize(None), freq="M")
    for ts in months.unique():
        g = df[months == ts]
        for tag, sub in [("ALL", g), ("daylight", g[g["is_day"]])]:
            if len(sub) < 5:
                continue
            rows.append(
                {
                    "month": str(ts),
                    "subset": tag,
                    "n": len(sub),
                    "R2": r2_score(sub["y"], sub["yh"]),
                    "nRMSE_%cap": np.sqrt(mean_squared_error(sub["y"], sub["yh"]))
                    / capacity
                    * 100,
                }
            )
    return pd.DataFrame(rows)


def plot_summary(agg: pd.DataFrame, out_png: Path) -> None:
    pivot_r2 = agg.pivot(index="model", columns="subset", values="R2")
    pivot_nrmse = agg.pivot(index="model", columns="subset", values="nRMSE_pct")
    pivot_r2 = pivot_r2.sort_values("daylight", ascending=False)
    pivot_nrmse = pivot_nrmse.loc[pivot_r2.index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    pivot_r2.plot(kind="barh", ax=ax1)
    ax1.set_xlabel("R²")
    ax1.set_title("R² (higher = better)")
    ax1.axvline(0.80, color="red", ls="--", lw=1, label="target 0.80")
    ax1.legend()
    pivot_nrmse.plot(kind="barh", ax=ax2)
    ax2.set_xlabel("nRMSE (% of capacity)")
    ax2.set_title("nRMSE (lower = better)")
    fig.suptitle("V2 (time-fixed + solar features) — rolling backtest")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> None:
    pv = load_pv_v2(CSV_PATH, time_shift_hours=-1)
    pv = flag_daylight(pv)
    capacity = float(pv["capacity"].iloc[0])
    _log(
        f"capacity={capacity:.0f} kW   span: {pv.index.min().date()} → {pv.index.max().date()}"
    )

    wx = load_weather()
    pv = pv.join(wx, how="left")
    pv[WEATHER_CHANNELS] = pv[WEATHER_CHANNELS].ffill().bfill()

    pv = add_solar_features(pv)
    pv = add_weather_lags(pv)
    pv, feats = make_features(pv)
    _log(f"features ({len(feats)}): {feats}")

    day_counts = pv.groupby(pv.index.normalize()).size()
    full_days_idx = day_counts[day_counts == 24].index
    pv = pv[pv.index.normalize().isin(full_days_idx)]

    folds = rolling_backtest(pv, capacity, feats, n_folds=4, first_test_days=120)
    agg = aggregate(folds, capacity)
    print("\n=== V2 — Rolling backtest (BOTH conventions) ===")
    with pd.option_context("display.max_colwidth", 40, "display.width", 130):
        for sub in ["ALL", "daylight"]:
            sub_agg = agg[agg["subset"] == sub].sort_values("nRMSE_pct")
            print(f"\n--- subset = {sub} ---")
            print(sub_agg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    agg.to_csv(OUT_DIR / "pv_v2_metrics.csv", index=False)

    pm = per_month_scores(folds, capacity, "RidgeStack")
    print("\n=== Per-month R² for RidgeStack ===")
    pivot_pm = pm.pivot(index="month", columns="subset", values="R2")
    print(pivot_pm.to_string(float_format=lambda v: f"{v:.3f}"))
    pm.to_csv(OUT_DIR / "pv_v2_per_month.csv", index=False)

    plot_summary(agg, OUT_DIR / "pv_v2_summary.png")
    print("\nSaved pv_v2_metrics.csv, pv_v2_per_month.csv, pv_v2_summary.png")

    print("\n=== Verdict (V2) ===")
    daylight_best = (
        agg[agg["subset"] == "daylight"].sort_values("R2", ascending=False).iloc[0]
    )
    all_best = agg[agg["subset"] == "ALL"].sort_values("R2", ascending=False).iloc[0]
    print(
        f"  Best model on DAYLIGHT only :  {daylight_best['model']}  "
        f"R²={daylight_best['R2']:.3f}  nRMSE={daylight_best['nRMSE_pct']:.2f}%"
    )
    print(
        f"  Best model on ALL hours     :  {all_best['model']}  "
        f"R²={all_best['R2']:.3f}  nRMSE={all_best['nRMSE_pct']:.2f}%"
    )


if __name__ == "__main__":
    main()
