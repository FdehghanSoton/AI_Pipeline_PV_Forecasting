"""V3 — adds the full 15-channel weather panel and contrasts two evaluation
protocols, since the per-month breakdown of V2 showed the autumn collapse is
caused by the rolling-origin protocol asking the model to predict a season
it has never seen, not by data quality.

Two evaluation modes
--------------------
  TEMPORAL  : strict rolling-origin (same as V2). Honest, but penalises the
              first-year setup because folds 2–4 may have to predict an
              out-of-distribution season.

  KFOLD     : random 5-fold CV across the whole year. Mixes train/test
              seasons exactly the way most '1-year R² benchmark' papers
              implicitly do. This is the upper bound for an in-distribution
              forecast assuming you had multi-year history.

Outputs:
  pv_v3_metrics.csv             – per-mode aggregated metrics
  pv_v3_per_month.csv           – per-month R² for the best model
  pv_v3_summary.png             – bar chart
  pv_v3_pred_actual.png         – scatter + timeline of best model
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import paths
from analyze_pv_cnn2d import flag_daylight
from analyze_pv_v2 import (
    add_solar_features,
    build_ensembles,
    load_pv_v2,
    metric_set,
)

CSV_PATH = paths.PV_CSV
WEATHER_CACHE = paths.WEATHER_DIR / "weather_cache.csv"
OUT_DIR = paths.results_dir()

EXTENDED_WEATHER = [
    "shortwave_radiation",
    "direct_normal_irradiance",
    "direct_radiation",
    "diffuse_radiation",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "precipitation",
    "pressure_msl",
]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_weather_full(
    source: str = "ifs", variable_set: str = "full"
) -> pd.DataFrame:
    """Load one weather product from its provenance-tracked cache.

    ``source`` selects the Open-Meteo product (see :mod:`weather_sources`).
    ``variable_set`` is ``"full"`` for all fifteen variables or
    ``"forecast_matched"`` for the twelve that the previous-runs forecast
    product also serves, which keeps analysis and forecast runs comparable.

    Feature construction downstream is column-driven, so dropping the three
    sub-level cloud fields automatically drops their lag, lead and rolling
    derivatives from the model input as well.
    """
    from weather_sources import FORECAST_VARS, SOURCES

    if source not in SOURCES:
        raise ValueError(
            f"unknown weather source {source!r}; expected one of "
            f"{sorted(SOURCES)}"
        )
    definition = SOURCES[source]
    if not definition.cache_path.exists():
        raise FileNotFoundError(
            f"{definition.cache_path.name} not found. Run "
            "`python weather_sources.py` to populate the weather caches."
        )
    if not definition.meta_path.exists():
        warnings.warn(
            f"{definition.cache_path.name} has no provenance metadata; "
            "refetch it so the weather product is auditable.",
            RuntimeWarning,
            stacklevel=2,
        )

    w = pd.read_csv(definition.cache_path, parse_dates=["time"])
    w["time"] = pd.to_datetime(w["time"], utc=True)
    w = w.set_index("time").sort_index()

    wanted = list(EXTENDED_WEATHER)
    if variable_set == "forecast_matched":
        wanted = [c for c in wanted if c in set(FORECAST_VARS)]
    elif variable_set != "full":
        raise ValueError(
            f"unknown weather variable set {variable_set!r}; expected 'full' "
            "or 'forecast_matched'"
        )
    have = [c for c in wanted if c in w.columns]
    return w[have]


def add_full_lags(df: pd.DataFrame) -> pd.DataFrame:
    for c in [
        "shortwave_radiation",
        "direct_normal_irradiance",
        "direct_radiation",
        "diffuse_radiation",
        "cloud_cover",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "wind_speed_10m",
        "poa_global",
        "clearness_kt",
    ]:
        if c not in df.columns:
            continue
        df[f"{c}_lag1"] = df[c].shift(1)
        df[f"{c}_lead1"] = df[c].shift(-1)
        df[f"{c}_roll3"] = df[c].rolling(3, center=True, min_periods=1).mean()
    df = df.ffill().bfill()
    return df


def make_features_v3(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    doy = df.index.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    base = [c for c in EXTENDED_WEATHER if c in df.columns]
    solar = [
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
    lag = [
        f"{c}_{s}"
        for c in [
            "shortwave_radiation",
            "direct_normal_irradiance",
            "direct_radiation",
            "diffuse_radiation",
            "cloud_cover",
            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",
            "wind_speed_10m",
            "poa_global",
            "clearness_kt",
        ]
        for s in ("lag1", "lead1", "roll3")
        if f"{c}_{s}" in df.columns
    ]
    cal = ["doy_sin", "doy_cos"]
    feats = base + solar + lag + cal
    feats = [f for f in feats if f in df.columns]
    return df, feats


@dataclass
class FoldRes:
    name: str
    test_idx: pd.DatetimeIndex
    y_test: np.ndarray
    is_day_test: np.ndarray
    is_miss_test: np.ndarray
    preds: dict[str, np.ndarray]


def fit_one_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feats: list[str],
    capacity: float,
) -> dict[str, np.ndarray]:
    X_tr, y_tr = train_df[feats].to_numpy(), train_df["y"].to_numpy()
    X_val, y_val = val_df[feats].to_numpy(), val_df["y"].to_numpy()
    X_te = test_df[feats].to_numpy()

    scaler = StandardScaler().fit(X_tr)
    ridge = Ridge(alpha=1.0).fit(scaler.transform(X_tr), y_tr)
    ridge_val = np.clip(ridge.predict(scaler.transform(X_val)), 0, None)
    ridge_te = np.clip(ridge.predict(scaler.transform(X_te)), 0, None)

    gbm = HistGradientBoostingRegressor(
        max_iter=800,
        learning_rate=0.04,
        max_depth=8,
        min_samples_leaf=30,
        l2_regularization=0.1,
        random_state=0,
        early_stopping=True,
        n_iter_no_change=25,
        validation_fraction=0.12,
    )
    gbm.fit(X_tr, y_tr)
    gbm_val = np.clip(gbm.predict(X_val), 0, None)
    gbm_te = np.clip(gbm.predict(X_te), 0, None)

    poa_tr = train_df["poa_global"].to_numpy()
    poa_val = val_df["poa_global"].to_numpy()
    poa_te = test_df["poa_global"].to_numpy()
    denom_tr = np.maximum(capacity * poa_tr / 1000.0, 1.0)
    denom_val = np.maximum(capacity * poa_val / 1000.0, 1.0)
    denom_te = np.maximum(capacity * poa_te / 1000.0, 1.0)
    kt = np.clip(y_tr / denom_tr, 0, 1.5)
    okkt = poa_tr > 50
    if okkt.sum() > 200:
        gbmkt = HistGradientBoostingRegressor(
            max_iter=800,
            learning_rate=0.04,
            max_depth=8,
            min_samples_leaf=30,
            l2_regularization=0.1,
            random_state=0,
            early_stopping=True,
            n_iter_no_change=25,
            validation_fraction=0.12,
        )
        gbmkt.fit(X_tr[okkt], kt[okkt])
        kt_val = np.clip(gbmkt.predict(X_val), 0, 1.5)
        kt_te = np.clip(gbmkt.predict(X_te), 0, 1.5)
        kt_val[poa_val < 50] = 0
        kt_te[poa_te < 50] = 0
        gbmkt_val_y = kt_val * denom_val
        gbmkt_te_y = kt_te * denom_te
    else:
        gbmkt_val_y, gbmkt_te_y = gbm_val, gbm_te

    is_day_val = val_df["is_daylight"].to_numpy().astype(bool)
    is_miss_val = val_df["is_missing"].to_numpy().astype(bool)
    val_mask_eval = is_day_val & ~is_miss_val
    val_preds = {"Ridge": ridge_val, "GBM": gbm_val, "GBM_kt": gbmkt_val_y}
    test_preds = {"Ridge": ridge_te, "GBM": gbm_te, "GBM_kt": gbmkt_te_y}
    ens = build_ensembles(val_preds, y_val, val_mask_eval, test_preds)
    out = {**test_preds, **ens}
    return out


def temporal_backtest(
    df: pd.DataFrame,
    capacity: float,
    feats: list[str],
    n_folds: int = 4,
    first_test_days: int = 120,
) -> list[FoldRes]:
    days = df.index.normalize().unique()
    n_days = len(days)
    fold_len = (n_days - first_test_days) // n_folds
    out: list[FoldRes] = []
    for fold in range(n_folds):
        ts = days[first_test_days + fold * fold_len]
        te = (
            days[first_test_days + (fold + 1) * fold_len]
            if fold < n_folds - 1
            else days[-1] + pd.Timedelta(days=1)
        )
        train_full = df[(df.index < ts) & (df["is_missing"] == 0)]
        if len(train_full) < 200:
            continue
        val_size = max(7 * 24, int(0.15 * len(train_full)))
        train, val = train_full.iloc[:-val_size], train_full.iloc[-val_size:]
        test = df[(df.index >= ts) & (df.index < te)]
        _log(
            f"  T fold {fold + 1}/{n_folds}  origin={ts.date()}  "
            f"train={len(train)}  val={len(val)}  test={len(test)}"
        )
        preds = fit_one_fold(train, val, test, feats, capacity)
        out.append(
            FoldRes(
                name=f"T{fold + 1}",
                test_idx=test.index,
                y_test=test["y"].to_numpy(),
                is_day_test=test["is_daylight"].to_numpy().astype(bool),
                is_miss_test=test["is_missing"].to_numpy().astype(bool),
                preds=preds,
            )
        )
    return out


def kfold_backtest(
    df: pd.DataFrame, capacity: float, feats: list[str], n_folds: int = 5, seed: int = 0
) -> list[FoldRes]:
    """Random K-fold by day (keep all 24 hours of a day in the same split)."""
    days = df.index.normalize().unique()
    n_days = len(days)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_days)
    fold_size = n_days // n_folds
    out: list[FoldRes] = []
    for k in range(n_folds):
        idx = perm[k * fold_size : (k + 1) * fold_size if k < n_folds - 1 else n_days]
        test_days = days[idx]
        is_test = df.index.normalize().isin(test_days)
        train_full = df[~is_test & (df["is_missing"] == 0)]
        val_size = max(7 * 24, int(0.10 * len(train_full)))
        rng2 = np.random.default_rng(seed + k)
        val_perm = rng2.permutation(len(train_full))
        val_pick = val_perm[:val_size]
        val_mask = np.zeros(len(train_full), dtype=bool)
        val_mask[val_pick] = True
        train, val = train_full[~val_mask], train_full[val_mask]
        test = df[is_test]
        _log(
            f"  K fold {k + 1}/{n_folds}  test_days={len(test_days)}  "
            f"train={len(train)}  val={len(val)}  test={len(test)}"
        )
        preds = fit_one_fold(train, val, test, feats, capacity)
        out.append(
            FoldRes(
                name=f"K{k + 1}",
                test_idx=test.index,
                y_test=test["y"].to_numpy(),
                is_day_test=test["is_daylight"].to_numpy().astype(bool),
                is_miss_test=test["is_missing"].to_numpy().astype(bool),
                preds=preds,
            )
        )
    return out


def aggregate(folds: list[FoldRes], capacity: float, mode: str) -> pd.DataFrame:
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
            rows.append({"mode": mode, "model": name, "subset": tag, **m})
    return pd.DataFrame(rows)


def per_month_scores(
    folds: list[FoldRes], capacity: float, model: str, mode: str
) -> pd.DataFrame:
    chunks = []
    for f in folds:
        ok = ~f.is_miss_test
        chunks.append(
            pd.DataFrame(
                {
                    "y": f.y_test[ok],
                    "yh": f.preds[model][ok],
                    "is_day": f.is_day_test[ok],
                },
                index=f.test_idx[ok],
            )
        )
    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    months = pd.PeriodIndex(df.index.tz_convert("UTC").tz_localize(None), freq="M")
    rows = []
    for ts in sorted(months.unique()):
        g = df[months == ts]
        for tag, sub in [("ALL", g), ("daylight", g[g["is_day"]])]:
            if len(sub) < 5:
                continue
            rows.append(
                {
                    "mode": mode,
                    "month": str(ts),
                    "subset": tag,
                    "n": len(sub),
                    "R2": r2_score(sub["y"], sub["yh"]),
                    "nRMSE_pct": np.sqrt(mean_squared_error(sub["y"], sub["yh"]))
                    / capacity
                    * 100,
                }
            )
    return pd.DataFrame(rows)


def plot_summary(metrics: pd.DataFrame, out_png: Path) -> None:
    pivot = metrics.pivot_table(index="model", columns=["mode", "subset"], values="R2")
    fig, ax = plt.subplots(figsize=(11, 6))
    pivot = pivot.sort_values(("TEMPORAL", "daylight"), ascending=False)
    pivot.plot(kind="barh", ax=ax)
    ax.axvline(0.80, color="red", ls="--", label="target 0.80")
    ax.set_xlabel("R²")
    ax.set_title("V3 — both eval modes  ×  both subsets")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_pred_actual(
    folds: list[FoldRes], capacity: float, model: str, mode: str, out_png: Path
) -> None:
    chunks = []
    for f in folds:
        ok = ~f.is_miss_test
        chunks.append(
            pd.DataFrame(
                {
                    "y": f.y_test[ok],
                    "yh": f.preds[model][ok],
                    "is_day": f.is_day_test[ok],
                },
                index=f.test_idx[ok],
            )
        )
    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    ax = axes[0]
    ax.scatter(
        df["y"], df["yh"], s=2, alpha=0.3, c=df["is_day"].astype(int), cmap="coolwarm"
    )
    lim = capacity * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set_xlabel("Observed (kW)")
    ax.set_ylabel("Predicted (kW)")
    ax.set_xlim(-10, lim)
    ax.set_ylim(-10, lim)
    ax.set_title(
        f"{model} — {mode}    "
        f"R²={r2_score(df['y'], df['yh']):.3f} (all)   "
        f"daylight R²={r2_score(df[df.is_day]['y'], df[df.is_day]['yh']):.3f}"
    )

    sample = df.last("21D")
    ax2 = axes[1]
    ax2.plot(sample.index, sample["y"], label="actual", lw=1)
    ax2.plot(sample.index, sample["yh"], label="predicted", lw=1, alpha=0.8)
    ax2.set_ylabel("kW")
    ax2.legend()
    ax2.set_title("last 21 days of pooled test set")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> None:
    pv = load_pv_v2(CSV_PATH, time_shift_hours=-1)
    pv = flag_daylight(pv)
    capacity = float(pv["capacity"].iloc[0])
    _log(
        f"capacity={capacity:.0f} kW   "
        f"span={pv.index.min().date()} → {pv.index.max().date()}"
    )
    wx = load_weather_full()
    pv = pv.join(wx, how="left")
    pv[wx.columns] = pv[wx.columns].ffill().bfill()
    pv = add_solar_features(pv)
    pv = add_full_lags(pv)
    pv, feats = make_features_v3(pv)
    _log(f"features ({len(feats)}): {feats}")

    day_counts = pv.groupby(pv.index.normalize()).size()
    full_days_idx = day_counts[day_counts == 24].index
    pv = pv[pv.index.normalize().isin(full_days_idx)]

    print("\n=== TEMPORAL (rolling-origin, strict) ===")
    t_folds = temporal_backtest(pv, capacity, feats, n_folds=4, first_test_days=120)
    t_metrics = aggregate(t_folds, capacity, "TEMPORAL")

    print("\n=== KFOLD (random 5-fold by day, in-distribution) ===")
    k_folds = kfold_backtest(pv, capacity, feats, n_folds=5, seed=0)
    k_metrics = aggregate(k_folds, capacity, "KFOLD")

    metrics = pd.concat([t_metrics, k_metrics], ignore_index=True)
    metrics.to_csv(OUT_DIR / "pv_v3_metrics.csv", index=False)

    for mode, df_m in metrics.groupby("mode"):
        for sub in ["ALL", "daylight"]:
            print(f"\n--- {mode}  ({sub}) ---")
            df_s = df_m[df_m["subset"] == sub].sort_values("nRMSE_pct")
            print(df_s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    pm_rows = []
    for mode, fld in [("TEMPORAL", t_folds), ("KFOLD", k_folds)]:
        pm_rows.append(per_month_scores(fld, capacity, "RidgeStack", mode))
    pm = pd.concat(pm_rows, ignore_index=True)
    pm.to_csv(OUT_DIR / "pv_v3_per_month.csv", index=False)

    print("\n=== Per-month R² for RidgeStack (both modes) ===")
    pivot_pm = pm.pivot_table(index="month", columns=["mode", "subset"], values="R2")
    print(pivot_pm.to_string(float_format=lambda v: f"{v:.3f}"))

    plot_summary(metrics, OUT_DIR / "pv_v3_summary.png")
    plot_pred_actual(
        k_folds,
        capacity,
        "RidgeStack",
        "KFOLD",
        OUT_DIR / "pv_v3_pred_actual_kfold.png",
    )
    plot_pred_actual(
        t_folds,
        capacity,
        "RidgeStack",
        "TEMPORAL",
        OUT_DIR / "pv_v3_pred_actual_temporal.png",
    )

    print("\n=== Verdict ===")
    for mode in ["TEMPORAL", "KFOLD"]:
        for sub in ["ALL", "daylight"]:
            best = (
                metrics[(metrics["mode"] == mode) & (metrics["subset"] == sub)]
                .sort_values("R2", ascending=False)
                .iloc[0]
            )
            print(
                f"  {mode:8s} {sub:8s}  best={best['model']:14s}  "
                f"R²={best['R2']:.3f}   nRMSE={best['nRMSE_pct']:.2f}%"
            )


if __name__ == "__main__":
    main()
