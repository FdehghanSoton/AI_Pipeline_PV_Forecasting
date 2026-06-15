"""V4 — adds per-hour GBM + reintegrates the 2D CNN on top of V3's stack.

Two new base learners
---------------------
  GBM_per_hour : 24 separate HistGradientBoostingRegressor models, one for
                 each hour-of-day target. Each sub-model only sees rows
                 where index.hour == h, so it can specialise on the
                 morning/noon/afternoon irradiance-to-power transfer.

  CNN2D        : the 2D ConvNet from analyze_pv_cnn2d.py, but re-trained on
                 the time-shift-corrected PV series. Provides spatial-
                 temporal diversity (intra-day and inter-day cloud patterns).

Everything else (features, rolling-origin + random K-fold protocols, ensembles,
per-month breakdown) is identical to V3 so V3 ↔ V4 numbers are directly
comparable.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from analyze_pv_cnn2d import (
    WEATHER_CHANNELS,
    build_supervised_tensors,
    flag_daylight,
    set_global_seed,
    to_daily_matrix,
    train_cnn,
)
from analyze_pv_v2 import (
    add_solar_features,
    build_ensembles,
    flag_daylight_geometric,
    load_pv_v2,
    metric_set,
)
from analyze_pv_v3 import (
    add_full_lags,
    load_weather_full,
    make_features_v3,
    plot_summary,
)
from baselines import add_skill_columns, baseline_predictions
from config import RunConfig, load_config
from stats_tests import diebold_mariano, ensemble_gain

CSV_PATH = Path(__file__).with_name("PV_data.csv")
OUT_DIR = Path(__file__).parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(4)

BASE_LEARNERS = ["Ridge", "GBM", "GBM_kt", "GBM_per_hour", "CNN"]
ENSEMBLE_NAMES = ["Mean", "InvRMSE", "RidgeStack", "NNLSStack", "BestSingleByVal"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass
class FoldRes:
    name: str
    test_idx: pd.DatetimeIndex
    y_test: np.ndarray
    is_day_test: np.ndarray
    is_miss_test: np.ndarray
    preds: dict[str, np.ndarray]
    clearness_test: np.ndarray | None = None


def fit_per_hour_gbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feats: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """24 hour-specialised GBMs. Returns (val_pred, test_pred) hourly arrays."""
    val_pred = np.zeros(len(val_df))
    test_pred = np.zeros(len(test_df))
    train_h = train_df.index.hour.to_numpy()
    val_h = val_df.index.hour.to_numpy()
    test_h = test_df.index.hour.to_numpy()
    for h in range(24):
        m_tr = train_h == h
        m_va = val_h == h
        m_te = test_h == h
        if m_tr.sum() < 30 or m_te.sum() == 0:
            continue
        gbm = HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=15,
            l2_regularization=0.2,
            random_state=0,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.15,
        )
        gbm.fit(
            train_df.loc[m_tr, feats].to_numpy(), train_df.loc[m_tr, "y"].to_numpy()
        )
        if m_va.any():
            val_pred[m_va] = np.clip(
                gbm.predict(val_df.loc[m_va, feats].to_numpy()), 0, None
            )
        test_pred[m_te] = np.clip(
            gbm.predict(test_df.loc[m_te, feats].to_numpy()), 0, None
        )
    return val_pred, test_pred


def fit_cnn_for_fold(
    pv_full: pd.DataFrame,
    train_days: pd.DatetimeIndex,
    val_days: pd.DatetimeIndex,
    test_days: pd.DatetimeIndex,
    capacity: float,
    seed: int = 0,
) -> tuple[dict[pd.Timestamp, np.ndarray], dict[pd.Timestamp, np.ndarray]]:
    """Train CNN on `train_days`, return per-day 24-hour predictions for
    val and test days. Uses the time-shift-corrected PV via pv_full."""
    pv_mat, day_idx = to_daily_matrix(pv_full["y"])
    is_day_mat, _ = to_daily_matrix(pv_full["is_daylight"])
    is_miss_mat, _ = to_daily_matrix(pv_full["is_missing"])
    pv_mat = np.nan_to_num(pv_mat, nan=0.0).astype(np.float32)
    is_day_mat = np.nan_to_num(is_day_mat, nan=0).astype(int)
    is_miss_mat = np.nan_to_num(is_miss_mat, nan=1).astype(int)
    wx_mats = []
    for ch in WEATHER_CHANNELS:
        m, _ = to_daily_matrix(pv_full[ch])
        wx_mats.append(np.nan_to_num(m, nan=0.0))
    wx_mat_raw = np.stack(wx_mats, axis=0).astype(np.float32)

    # Normalise weather using train-day stats only (no leakage)
    train_mask = day_idx.isin(train_days)
    if train_mask.sum() < 10:
        return {}, {}
    train_slice = wx_mat_raw[:, train_mask, :]
    mean = np.nanmean(train_slice, axis=(1, 2), keepdims=True)
    std = np.nanstd(train_slice, axis=(1, 2), keepdims=True) + 1e-6
    wx_mat = (wx_mat_raw - mean) / std
    wx_mat = np.nan_to_num(wx_mat, nan=0.0).astype(np.float32)

    hidden_days = val_days.append(test_days).unique()
    X, Y, W, _E, target_days = build_supervised_tensors(
        pv_mat,
        wx_mat,
        day_idx,
        is_day_mat,
        is_miss_mat,
        capacity,
        hidden_pv_days=hidden_days,
    )
    is_train = target_days.isin(train_days)
    is_val = target_days.isin(val_days)
    is_test = target_days.isin(test_days)
    tr_idx = np.where(is_train)[0]
    val_idx = np.where(is_val)[0]
    te_idx = np.where(is_test)[0]
    if len(tr_idx) < 20 or len(te_idx) == 0:
        return {}, {}
    if len(val_idx) == 0:
        # carve off the last 15% of training days for CNN val
        n_val = max(7, int(0.15 * len(tr_idx)))
        val_idx = tr_idx[-n_val:]
        tr_idx = tr_idx[:-n_val]

    cnn = train_cnn(
        X[tr_idx],
        Y[tr_idx],
        W[tr_idx],
        X[val_idx],
        Y[val_idx],
        W[val_idx],
        seed=seed,
    )
    with torch.no_grad():
        val_pred = cnn(X[val_idx].to(DEVICE)).cpu().numpy() * capacity
        te_pred = cnn(X[te_idx].to(DEVICE)).cpu().numpy() * capacity
    val_pred = np.nan_to_num(val_pred, nan=0.0)
    te_pred = np.nan_to_num(te_pred, nan=0.0)

    # Map back to dict keyed by day
    val_map = {
        pd.Timestamp(target_days[i]).normalize(): val_pred[j]
        for j, i in enumerate(val_idx)
    }
    te_map = {
        pd.Timestamp(target_days[i]).normalize(): te_pred[j]
        for j, i in enumerate(te_idx)
    }
    return val_map, te_map


def expand_daily_to_hourly(
    daily_map: dict[pd.Timestamp, np.ndarray], hourly_index: pd.DatetimeIndex
) -> np.ndarray:
    out = np.zeros(len(hourly_index), dtype=np.float32)
    for j, ts in enumerate(hourly_index):
        d = ts.normalize()
        v = daily_map.get(d)
        if v is not None:
            out[j] = v[ts.hour]
    return out


def fit_one_fold(
    pv_full: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feats: list[str],
    capacity: float,
    fold_tag: str,
    train_capacity: float | None = None,
    cnn_seeds: tuple[int, ...] = (0,),
    include_baselines: bool = True,
) -> dict[str, np.ndarray]:
    """Fit all base learners and ensembles for one fold.

    ``capacity`` is the fixed (full-series) value used only as a metric and
    smart-persistence denominator. ``train_capacity`` is the leakage-safe
    capacity estimated from this fold's training rows; it drives the clearness
    target and CNN normalisation so that no test-day magnitude enters the
    learned signal. With several ``cnn_seeds`` the CNN prediction is the mean
    across seeds, which reduces single-seed variance.
    """
    if train_capacity is None:
        train_capacity = capacity
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
    denom_tr = np.maximum(train_capacity * poa_tr / 1000.0, 1.0)
    denom_val = np.maximum(train_capacity * poa_val / 1000.0, 1.0)
    denom_te = np.maximum(train_capacity * poa_te / 1000.0, 1.0)
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
        kt_val_p = np.clip(gbmkt.predict(X_val), 0, 1.5)
        kt_te_p = np.clip(gbmkt.predict(X_te), 0, 1.5)
        kt_val_p[poa_val < 50] = 0
        kt_te_p[poa_te < 50] = 0
        gbmkt_val_y = kt_val_p * denom_val
        gbmkt_te_y = kt_te_p * denom_te
    else:
        gbmkt_val_y, gbmkt_te_y = gbm_val, gbm_te

    t0 = time.time()
    perh_val, perh_te = fit_per_hour_gbm(train_df, val_df, test_df, feats)
    _log(f"  [{fold_tag}] per-hour GBM trained in {time.time() - t0:.1f}s")

    t0 = time.time()
    train_days = pd.DatetimeIndex(train_df.index.normalize().unique())
    val_days = pd.DatetimeIndex(val_df.index.normalize().unique())
    test_days = pd.DatetimeIndex(test_df.index.normalize().unique())
    base_seed = sum(ord(char) for char in fold_tag)
    cnn_val_runs: list[np.ndarray] = []
    cnn_te_runs: list[np.ndarray] = []
    for offset in cnn_seeds:
        cnn_val_map, cnn_te_map = fit_cnn_for_fold(
            pv_full,
            train_days,
            val_days,
            test_days,
            train_capacity,
            seed=base_seed + int(offset),
        )
        cnn_val_runs.append(
            np.nan_to_num(expand_daily_to_hourly(cnn_val_map, val_df.index), nan=0.0)
        )
        cnn_te_runs.append(
            np.nan_to_num(expand_daily_to_hourly(cnn_te_map, test_df.index), nan=0.0)
        )
    cnn_val = np.mean(cnn_val_runs, axis=0)
    cnn_te = np.mean(cnn_te_runs, axis=0)
    _log(
        f"  [{fold_tag}] CNN trained ({len(cnn_seeds)} seed(s)) "
        f"in {time.time() - t0:.1f}s"
    )

    is_day_val = val_df["is_daylight"].to_numpy().astype(bool)
    is_miss_val = val_df["is_missing"].to_numpy().astype(bool)
    val_mask_eval = is_day_val & ~is_miss_val

    val_preds = {
        "Ridge": ridge_val,
        "GBM": gbm_val,
        "GBM_kt": gbmkt_val_y,
        "GBM_per_hour": perh_val,
        "CNN": cnn_val,
    }
    test_preds = {
        "Ridge": ridge_te,
        "GBM": gbm_te,
        "GBM_kt": gbmkt_te_y,
        "GBM_per_hour": perh_te,
        "CNN": cnn_te,
    }
    ens = build_ensembles(val_preds, y_val, val_mask_eval, test_preds)
    # Reference baselines are scored alongside the models but are deliberately
    # excluded from the ensemble (they never enter build_ensembles), so they
    # cannot influence the stacking weights.
    base_preds: dict[str, np.ndarray] = {}
    if include_baselines:
        base_preds = baseline_predictions(
            pv_full, train_df, test_df.index, capacity
        )
    return {**test_preds, **ens, **base_preds}


def _fold_train_capacity(
    train_df: pd.DataFrame, capacity: float, policy: str
) -> float:
    """Capacity for the learned signal: fold-internal under ``fold_train``."""
    if policy == "fold_train" and len(train_df) > 50:
        return float(train_df["y"].quantile(0.999))
    return capacity


def temporal_backtest(
    pv_full: pd.DataFrame,
    df: pd.DataFrame,
    capacity: float,
    feats: list[str],
    n_folds: int = 4,
    first_test_days: int = 120,
    cfg: RunConfig | None = None,
) -> list[FoldRes]:
    cfg = cfg or load_config()
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
            f"T fold {fold + 1}/{n_folds}  origin={ts.date()}  "
            f"train={len(train)}  val={len(val)}  test={len(test)}"
        )
        train_cap = _fold_train_capacity(train, capacity, cfg.capacity_policy)
        preds = fit_one_fold(
            pv_full,
            train,
            val,
            test,
            feats,
            capacity,
            fold_tag=f"T{fold + 1}",
            train_capacity=train_cap,
            cnn_seeds=cfg.cnn_seeds,
            include_baselines=cfg.include_baselines,
        )
        out.append(
            FoldRes(
                name=f"T{fold + 1}",
                test_idx=test.index,
                y_test=test["y"].to_numpy(),
                is_day_test=test["is_daylight"].to_numpy().astype(bool),
                is_miss_test=test["is_missing"].to_numpy().astype(bool),
                preds=preds,
                clearness_test=test.get("clearness_kt", pd.Series(index=test.index))
                .to_numpy(),
            )
        )
    return out


def kfold_backtest(
    pv_full: pd.DataFrame,
    df: pd.DataFrame,
    capacity: float,
    feats: list[str],
    n_folds: int = 5,
    seed: int = 0,
    cfg: RunConfig | None = None,
) -> list[FoldRes]:
    cfg = cfg or load_config()
    days = df.index.normalize().unique()
    n_days = len(days)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_days)
    fold_size = n_days // n_folds
    out: list[FoldRes] = []
    for k in range(n_folds):
        idx = perm[k * fold_size : (k + 1) * fold_size if k < n_folds - 1 else n_days]
        test_days = days[idx]
        is_test_h = df.index.normalize().isin(test_days)
        train_full = df[~is_test_h & (df["is_missing"] == 0)]
        train_days_all = pd.DatetimeIndex(train_full.index.normalize().unique())
        rng2 = np.random.default_rng(seed + k)
        train_days_perm = train_days_all[rng2.permutation(len(train_days_all))]
        n_val_days = max(7, int(0.10 * len(train_days_all)))
        val_days = train_days_perm[:n_val_days]
        train_days = train_days_perm[n_val_days:]
        train = train_full[train_full.index.normalize().isin(train_days)]
        val = train_full[train_full.index.normalize().isin(val_days)]
        test = df[is_test_h]
        _log(
            f"K fold {k + 1}/{n_folds}  test_days={len(test_days)}  "
            f"train={len(train)}  val={len(val)}  test={len(test)}"
        )
        train_cap = _fold_train_capacity(train, capacity, cfg.capacity_policy)
        preds = fit_one_fold(
            pv_full,
            train,
            val,
            test,
            feats,
            capacity,
            fold_tag=f"K{k + 1}",
            train_capacity=train_cap,
            cnn_seeds=cfg.cnn_seeds,
            include_baselines=cfg.include_baselines,
        )
        out.append(
            FoldRes(
                name=f"K{k + 1}",
                test_idx=test.index,
                y_test=test["y"].to_numpy(),
                is_day_test=test["is_daylight"].to_numpy().astype(bool),
                is_miss_test=test["is_missing"].to_numpy().astype(bool),
                preds=preds,
                clearness_test=test.get("clearness_kt", pd.Series(index=test.index))
                .to_numpy(),
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


def folds_to_frame(folds: list[FoldRes], mode: str) -> pd.DataFrame:
    """Return one auditable row per held-out timestamp and model forecast."""
    parts: list[pd.DataFrame] = []
    for fold in folds:
        frame = pd.DataFrame(
            {
                "mode": mode,
                "fold": fold.name,
                "timestamp": fold.test_idx,
                "y_actual": fold.y_test,
                "is_daylight": fold.is_day_test.astype(int),
                "is_missing": fold.is_miss_test.astype(int),
            }
        )
        if fold.clearness_test is not None:
            frame["clearness_kt"] = fold.clearness_test
        for model_name, values in fold.preds.items():
            frame[model_name] = values
        parts.append(frame)
    return pd.concat(parts, ignore_index=True).sort_values("timestamp")


def residual_correlation(folds: list[FoldRes], base_names: list[str]) -> pd.DataFrame:
    all_y = np.concatenate([f.y_test[~f.is_miss_test & f.is_day_test] for f in folds])
    res = {}
    for n in base_names:
        all_yh = np.concatenate(
            [f.preds[n][~f.is_miss_test & f.is_day_test] for f in folds]
        )
        res[n] = all_y - all_yh
    return pd.DataFrame(res).corr()


SOLAR_FEATURES = {
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
}


def apply_feature_ablation(
    pv: pd.DataFrame, feats: list[str], cfg: RunConfig
) -> tuple[pd.DataFrame, list[str]]:
    """Filter the feature list according to the ablation switches in ``cfg``.

    Dropping physics features removes the solar-geometry and clearness columns
    (and their temporal derivatives) from the model inputs, and substitutes a
    calendar-only diurnal encoding so the tabular models still receive some
    time-of-day signal. The underlying ``poa_global`` column is retained in the
    frame for the clearness target, so this ablates the input representation
    only, not the clearness-target machinery.
    """
    pv = pv.copy()
    out = list(feats)

    if not cfg.use_physics_features:
        out = [
            f
            for f in out
            if f not in SOLAR_FEATURES
            and not f.startswith(("clearness_kt", "poa_global"))
        ]
        hour = pv.index.hour
        pv["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        pv["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        out += ["hour_sin", "hour_cos"]

    if not cfg.use_temporal_context:
        out = [f for f in out if not f.endswith(("_lag1", "_lead1", "_roll3"))]

    seen: set[str] = set()
    deduped = [f for f in out if not (f in seen or seen.add(f))]
    return pv, deduped


def aggregate_per_fold(folds: list[FoldRes], capacity: float, mode: str) -> pd.DataFrame:
    """Per-fold metrics (not pooled), so fold-to-fold spread can be reported."""
    names = list(folds[0].preds.keys())
    rows = []
    for fold in folds:
        keep = ~fold.is_miss_test
        y = fold.y_test[keep]
        day = fold.is_day_test[keep]
        for name in names:
            yh = fold.preds[name][keep]
            for tag, mask in [
                ("ALL", np.ones(len(y), dtype=bool)),
                ("daylight", day),
            ]:
                m = metric_set(y[mask], yh[mask], capacity)
                rows.append(
                    {"mode": mode, "fold": fold.name, "model": name, "subset": tag, **m}
                )
    return pd.DataFrame(rows)


def significance_table(
    predictions: pd.DataFrame, metrics: pd.DataFrame
) -> pd.DataFrame:
    """Diebold-Mariano test of the best ensemble vs the best base learner.

    For each (mode, subset) the lowest-RMSE ensemble and the lowest-RMSE base
    learner are selected from the pooled metrics, and their daylight/all-hours
    residuals are compared on the pooled held-out predictions.
    """
    rows = []
    for mode in predictions["mode"].unique():
        pred_mode = predictions[predictions["mode"] == mode]
        valid = pred_mode[pred_mode["is_missing"] == 0]
        for subset in ("ALL", "daylight"):
            sub = valid if subset == "ALL" else valid[valid["is_daylight"] == 1]
            met = metrics[(metrics["mode"] == mode) & (metrics["subset"] == subset)]
            ens = met[met["model"].isin(ENSEMBLE_NAMES)].sort_values("RMSE")
            base = met[met["model"].isin(BASE_LEARNERS)].sort_values("RMSE")
            if ens.empty or base.empty:
                continue
            best_ens = ens.iloc[0]["model"]
            best_base = base.iloc[0]["model"]
            gain = ensemble_gain(
                float(ens.iloc[0]["RMSE"]), float(base.iloc[0]["RMSE"])
            )
            dm = diebold_mariano(
                sub["y_actual"].to_numpy(),
                sub[best_ens].to_numpy(),
                sub[best_base].to_numpy(),
                loss="squared",
                name_a=best_ens,
                name_b=best_base,
            )
            rows.append(
                {
                    "mode": mode,
                    "subset": subset,
                    "best_ensemble": best_ens,
                    "best_base": best_base,
                    **gain,
                    **dm.as_dict(),
                }
            )
    return pd.DataFrame(rows)


def build_dataset(cfg: RunConfig) -> tuple[pd.DataFrame, list[str], float]:
    """Load and feature-engineer the PV panel under the active configuration."""
    pv = load_pv_v2(CSV_PATH, time_shift_hours=cfg.time_shift_hours)
    wx = load_weather_full()
    pv = pv.join(wx, how="left")
    pv[wx.columns] = pv[wx.columns].ffill().bfill()
    pv = add_solar_features(pv)

    if cfg.daylight_policy == "geometric":
        pv = flag_daylight_geometric(
            pv,
            pct=cfg.daylight_pct,
            elevation_deg=cfg.daylight_elevation_deg,
            lat=cfg.site.lat,
            lon=cfg.site.lon,
            alt=cfg.site.alt,
        )
    else:
        pv = flag_daylight(pv, pct=cfg.daylight_pct)
    capacity = float(pv["capacity"].iloc[0])

    pv = add_full_lags(pv)
    pv, feats = make_features_v3(pv)
    pv, feats = apply_feature_ablation(pv, feats, cfg)

    # Keep only days that have all 24 hourly slots present.
    day_counts = pv.groupby(pv.index.normalize()).size()
    full_days_idx = day_counts[day_counts == 24].index
    pv = pv[pv.index.normalize().isin(full_days_idx)]

    if cfg.drop_missing_days:
        day_missing = pv.groupby(pv.index.normalize())["is_missing"].transform("sum")
        before = len(pv)
        pv = pv[day_missing == 0]
        _log(
            f"drop_missing_days: removed {before - len(pv)} rows on days with "
            f"any missing PV ({pv.index.normalize().nunique()} days remain)"
        )

    _log(
        f"capacity={capacity:.0f} kW   span={pv.index.min().date()} -> "
        f"{pv.index.max().date()}   features={len(feats)}   "
        f"daylight={cfg.daylight_policy}   capacity_policy={cfg.capacity_policy}"
    )
    return pv, feats, capacity


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg.seed)
    _log(f"device: {DEVICE}")
    _log(f"config: {json.dumps(cfg.describe())}")
    cfg.tagged("pv_v4_run_config.json").write_text(
        json.dumps(cfg.describe(), indent=2) + "\n", encoding="utf-8"
    )

    pv, feats, capacity = build_dataset(cfg)

    print("\n=== TEMPORAL ===")
    t_folds = temporal_backtest(
        pv,
        pv,
        capacity,
        feats,
        n_folds=cfg.temporal_n_folds,
        first_test_days=cfg.first_test_days,
        cfg=cfg,
    )
    t_metrics = aggregate(t_folds, capacity, "TEMPORAL")

    print("\n=== KFOLD ===")
    k_folds = kfold_backtest(
        pv, pv, capacity, feats, n_folds=cfg.kfold_n_folds, seed=cfg.seed, cfg=cfg
    )
    k_metrics = aggregate(k_folds, capacity, "KFOLD")

    metrics = pd.concat([t_metrics, k_metrics], ignore_index=True)
    if cfg.include_baselines:
        metrics = add_skill_columns(metrics, reference="SmartPersistence")
    metrics.to_csv(cfg.tagged("pv_v4_metrics.csv"), index=False)

    per_fold = pd.concat(
        [
            aggregate_per_fold(t_folds, capacity, "TEMPORAL"),
            aggregate_per_fold(k_folds, capacity, "KFOLD"),
        ],
        ignore_index=True,
    )
    per_fold.to_csv(cfg.tagged("pv_v4_per_fold_metrics.csv"), index=False)

    predictions = pd.concat(
        [folds_to_frame(t_folds, "TEMPORAL"), folds_to_frame(k_folds, "KFOLD")],
        ignore_index=True,
    )
    predictions.to_csv(cfg.tagged("pv_v4_predictions.csv"), index=False)

    for mode, df_m in metrics.groupby("mode"):
        for sub in ["ALL", "daylight"]:
            print(f"\n--- {mode} ({sub}) ---")
            df_s = df_m[df_m["subset"] == sub].sort_values("nRMSE_pct")
            print(df_s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    pm_rows = []
    for mode, fld in [("TEMPORAL", t_folds), ("KFOLD", k_folds)]:
        # Pick the best ensemble for this mode (lowest nRMSE on daylight)
        sub = metrics[
            (metrics["mode"] == mode)
            & (metrics["subset"] == "daylight")
            & (metrics["model"].isin(ENSEMBLE_NAMES))
        ]
        best_model = sub.sort_values("nRMSE_pct").iloc[0]["model"]
        pm_rows.append(per_month_scores(fld, capacity, best_model, mode))
        print(f"\n=== Per-month R2 for {best_model} [{mode}] ===")
        df_pm = pm_rows[-1]
        pivot = df_pm.pivot_table(index="month", columns="subset", values="R2")
        print(pivot.to_string(float_format=lambda v: f"{v:.3f}"))
    pm = pd.concat(pm_rows, ignore_index=True)
    pm.to_csv(cfg.tagged("pv_v4_per_month.csv"), index=False)

    corr_temporal = residual_correlation(t_folds, BASE_LEARNERS)
    corr_kfold = residual_correlation(k_folds, BASE_LEARNERS)
    print("\n=== Residual correlation between base models (TEMPORAL, daylight) ===")
    print(corr_temporal.round(3).to_string())
    print("\n=== Residual correlation between base models (KFOLD, daylight) ===")
    print(corr_kfold.round(3).to_string())
    corr_temporal.to_csv(cfg.tagged("pv_v4_residual_corr_temporal.csv"))
    corr_kfold.to_csv(cfg.tagged("pv_v4_residual_corr_kfold.csv"))

    significance = significance_table(predictions, metrics)
    if not significance.empty:
        significance.to_csv(cfg.tagged("pv_v4_significance.csv"), index=False)
        print("\n=== Ensemble vs best base learner (Diebold-Mariano) ===")
        print(significance.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    plot_summary(metrics, cfg.tagged("pv_v4_summary.png"))
    print(
        "\nSaved metrics, per-fold metrics, predictions, per-month, "
        "residual-correlation, significance, and summary outputs"
        + (f" (run tag: {cfg.run_tag})" if cfg.run_tag else "")
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
                f"R2={best['R2']:.3f}   nRMSE={best['nRMSE_pct']:.2f}%"
            )


if __name__ == "__main__":
    main()
