"""Ensemble of GBM + Ridge + 2D-CNN for 24 h-ahead hourly PV forecasting.

Protocol
--------
Rolling-origin backtest with 4 folds (same split as analyze_pv_cnn2d.py).
In every fold:
  1. Split prior history chronologically into train (70%) / in-fold val (30%).
  2. Fit Ridge, GBM and CNN on train.
  3. Predict each model on val AND on test.
  4. Learn ensemble weights on the *val* predictions (never the test).
  5. Combine the test predictions using the learnt weights.

Metrics on daylight hours only, aggregated across all four fold test sets.
Also report the residual-correlation matrix between base models — this is
the ceiling for how much ensembling can possibly help.

Ensemble strategies compared
----------------------------
    mean                 : equal 1/K weights
    inv_rmse             : weights proportional to 1 / val_RMSE_i, normalised
    ridge_stack          : Ridge regression y_val ~ [yhat_i] with alpha=1
    nnls_stack           : non-negative LS with sum(w)=1 constraint
    hourly_stack         : separate NNLS weights per hour-of-day
    best_single          : picks the base model with the best val RMSE
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import paths
from analyze_pv_cnn2d import (
    WEATHER_CHANNELS,
    build_supervised_tensors,
    flag_daylight,
    load_pv,
    load_weather,
    to_daily_matrix,
    train_cnn,
)

CSV_PATH = paths.PV_CSV
OUT_DIR = paths.results_dir()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(4)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_hourly_panel(pv: pd.DataFrame) -> pd.DataFrame:
    """Hourly tabular panel for GBM / Ridge: calendar + weather + past PV."""
    df = pv.copy()
    idx = df.index
    hr = idx.hour
    doy = idx.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


@dataclass
class FoldPreds:
    origin: pd.Timestamp
    days: pd.DatetimeIndex
    y: np.ndarray
    mask: np.ndarray
    preds: dict[str, np.ndarray]


def fit_and_predict(
    pv_mat: np.ndarray,
    wx_mat_raw: np.ndarray,
    day_idx: pd.DatetimeIndex,
    is_day_mat: np.ndarray,
    is_miss_mat: np.ndarray,
    panel: pd.DataFrame,
    capacity: float,
    n_folds: int = 4,
) -> list[FoldPreds]:
    n_days = pv_mat.shape[0]
    first_test = 120 + 7
    fold_len = (n_days - first_test) // n_folds

    cal_feats = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    wx_feats = WEATHER_CHANNELS
    tab_feats = cal_feats + wx_feats

    fold_preds: list[FoldPreds] = []

    for fold in range(n_folds):
        test_start = first_test + fold * fold_len
        test_end = test_start + fold_len if fold < n_folds - 1 else n_days

        mean = np.nanmean(wx_mat_raw[:, :test_start, :], axis=(1, 2), keepdims=True)
        std = (
            np.nanstd(wx_mat_raw[:, :test_start, :], axis=(1, 2), keepdims=True) + 1e-6
        )
        wx_mat = (wx_mat_raw - mean) / std
        wx_mat = np.nan_to_num(wx_mat, nan=0.0).astype(np.float32)

        X, Y, W, _E, target_days = build_supervised_tensors(
            pv_mat,
            wx_mat,
            day_idx,
            is_day_mat,
            is_miss_mat,
            capacity,
        )
        n_samples = len(target_days)
        train_mask = np.arange(n_samples) < test_start - 7
        test_mask = (~train_mask) & (np.arange(n_samples) < test_end - 7)

        train_idx_all = np.where(train_mask)[0]
        val_size = max(14, int(0.2 * len(train_idx_all)))
        val_idx_cnn = train_idx_all[-val_size:]
        tr_idx_cnn = train_idx_all[:-val_size]
        te_idx = np.where(test_mask)[0]

        _log(
            f"fold {fold + 1}/{n_folds} origin={target_days[te_idx[0]].date()} "
            f"train={len(tr_idx_cnn)} val={val_size} test={len(te_idx)}"
        )

        t0 = time.time()
        cnn = train_cnn(
            X[tr_idx_cnn],
            Y[tr_idx_cnn],
            W[tr_idx_cnn],
            X[val_idx_cnn],
            Y[val_idx_cnn],
            W[val_idx_cnn],
            seed=fold,
        )
        with torch.no_grad():
            cnn_val = cnn(X[val_idx_cnn].to(DEVICE)).cpu().numpy() * capacity
            cnn_te = cnn(X[te_idx].to(DEVICE)).cpu().numpy() * capacity
        cnn_val = np.nan_to_num(cnn_val, nan=0.0)
        cnn_te = np.nan_to_num(cnn_te, nan=0.0)
        _log(f"  CNN trained in {time.time() - t0:.1f}s")

        val_days = target_days[val_idx_cnn]
        test_days = target_days[te_idx]
        train_days = target_days[tr_idx_cnn]

        train_h = panel[panel.index.normalize().isin(train_days)]
        val_h = panel[panel.index.normalize().isin(val_days)]
        test_h = panel[panel.index.normalize().isin(test_days)]
        train_h = train_h.dropna(subset=["y"])

        scaler = StandardScaler().fit(train_h[tab_feats])
        ridge = Ridge(alpha=1.0).fit(scaler.transform(train_h[tab_feats]), train_h["y"])
        ridge_val = np.clip(ridge.predict(scaler.transform(val_h[tab_feats])), 0, None)
        ridge_te = np.clip(ridge.predict(scaler.transform(test_h[tab_feats])), 0, None)

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
        ).fit(train_h[tab_feats], train_h["y"])
        gbm_val = np.clip(gbm.predict(val_h[tab_feats]), 0, None)
        gbm_te = np.clip(gbm.predict(test_h[tab_feats]), 0, None)

        val_h = val_h.assign(
            yhat_cnn=_expand_daily_to_hourly(cnn_val, val_days, val_h.index),
            yhat_ridge=ridge_val,
            yhat_gbm=gbm_val,
        )
        test_h = test_h.assign(
            yhat_cnn=_expand_daily_to_hourly(cnn_te, test_days, test_h.index),
            yhat_ridge=ridge_te,
            yhat_gbm=gbm_te,
        )

        val_daily_mask = (
            (val_h["is_daylight"] == 1) & (val_h["is_missing"] == 0)
        ).to_numpy()
        test_daily_mask = (
            (test_h["is_daylight"] == 1) & (test_h["is_missing"] == 0)
        ).to_numpy()

        test_y_hourly = test_h["y"].to_numpy()
        test_preds = {
            "CNN": test_h["yhat_cnn"].to_numpy(),
            "Ridge": test_h["yhat_ridge"].to_numpy(),
            "GBM": test_h["yhat_gbm"].to_numpy(),
        }
        val_y_hourly = val_h["y"].to_numpy()
        val_preds = {
            "CNN": val_h["yhat_cnn"].to_numpy(),
            "Ridge": val_h["yhat_ridge"].to_numpy(),
            "GBM": val_h["yhat_gbm"].to_numpy(),
        }

        ensembles = build_ensembles(
            val_preds,
            val_y_hourly,
            val_daily_mask,
            test_preds,
            test_h.index.hour.to_numpy(),
        )
        all_preds = {**test_preds, **ensembles}

        fold_preds.append(
            FoldPreds(
                origin=target_days[te_idx[0]],
                days=test_h.index,
                y=test_y_hourly,
                mask=test_daily_mask,
                preds=all_preds,
            )
        )
    return fold_preds


def _expand_daily_to_hourly(
    daily_preds: np.ndarray,
    target_days: pd.DatetimeIndex,
    hourly_index: pd.DatetimeIndex,
) -> np.ndarray:
    """daily_preds: (n_days, 24) from CNN.  Expand to hourly index values."""
    out = np.zeros(len(hourly_index), dtype=np.float32)
    day_to_row = {pd.Timestamp(d).normalize(): i for i, d in enumerate(target_days)}
    for j, ts in enumerate(hourly_index):
        row = day_to_row.get(ts.normalize())
        if row is not None:
            out[j] = daily_preds[row, ts.hour]
    return out


def build_ensembles(
    val_preds: dict,
    y_val: np.ndarray,
    val_mask: np.ndarray,
    test_preds: dict,
    test_hours: np.ndarray,
) -> dict:
    names = list(val_preds.keys())
    V = np.stack([val_preds[n] for n in names], axis=1)
    T = np.stack([test_preds[n] for n in names], axis=1)
    y_m = y_val[val_mask]
    V_m = V[val_mask]

    val_rmse = {
        n: float(np.sqrt(mean_squared_error(y_m, V_m[:, i])))
        for i, n in enumerate(names)
    }

    out = {}
    out["Mean"] = T.mean(axis=1)

    inv = np.array([1.0 / val_rmse[n] for n in names])
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

    hourly_pred = np.zeros(T.shape[0])
    default_w = np.ones(len(names)) / len(names)
    for h in range(24):
        val_mask & (val_preds[names[0]] == val_preds[names[0]])
        hr_val_idx = np.where(
            val_mask & (np.repeat(np.arange(len(y_val)) % 24 == h, 1))
        )[0]
        if len(hr_val_idx) < 8:
            w = default_w
        else:
            try:
                w, _ = nnls(V[hr_val_idx], y_val[hr_val_idx])
                if w.sum() > 1e-6:
                    w = w / w.sum()
                else:
                    w = default_w
            except Exception:
                w = default_w
        hr_test_idx = np.where(test_hours == h)[0]
        hourly_pred[hr_test_idx] = T[hr_test_idx] @ w
    out["HourlyNNLS"] = hourly_pred

    best = min(names, key=lambda n: val_rmse[n])
    out["BestSingleByVal"] = test_preds[best]

    return out


def aggregate(fold_preds: list[FoldPreds], capacity: float) -> pd.DataFrame:
    all_y = np.concatenate([fp.y[fp.mask.astype(bool)] for fp in fold_preds])
    names = list(fold_preds[0].preds.keys())
    rows = []
    for name in names:
        all_yh = np.concatenate(
            [fp.preds[name][fp.mask.astype(bool)] for fp in fold_preds]
        )
        m = np.isfinite(all_y) & np.isfinite(all_yh)
        y = all_y[m]
        yh = all_yh[m]
        rows.append(
            {
                "model": name,
                "n": len(y),
                "R2": round(r2_score(y, yh), 4),
                "MAE": round(mean_absolute_error(y, yh), 1),
                "RMSE": round(float(np.sqrt(mean_squared_error(y, yh))), 1),
                "nMAE_%cap": round(mean_absolute_error(y, yh) / capacity * 100, 2),
                "nRMSE_%cap": round(
                    float(np.sqrt(mean_squared_error(y, yh))) / capacity * 100, 2
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("nRMSE_%cap")


def residual_correlation(fold_preds: list[FoldPreds]) -> pd.DataFrame:
    base = ["CNN", "Ridge", "GBM"]
    all_y = np.concatenate([fp.y[fp.mask.astype(bool)] for fp in fold_preds])
    res = {}
    for n in base:
        all_yh = np.concatenate(
            [fp.preds[n][fp.mask.astype(bool)] for fp in fold_preds]
        )
        res[n] = all_y - all_yh
    return pd.DataFrame(res).corr()


def plot_bar(agg: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    colors = [
        "#d62728"
        if m in ("CNN", "Ridge", "GBM")
        else ("#2ca02c" if m == "BestSingleByVal" else "#1f77b4")
        for m in agg["model"]
    ]
    ax.barh(agg["model"], agg["nRMSE_%cap"], color=colors)
    for i, (_, r) in enumerate(agg.iterrows()):
        ax.text(
            r["nRMSE_%cap"] + 0.05, i, f" R²={r['R2']:.3f}", va="center", fontsize=9
        )
    ax.set_xlabel("Test daylight nRMSE (% of capacity)")
    ax.set_title("Base models vs. ensembles (lower is better)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> None:
    _log(f"device: {DEVICE}")
    pv = load_pv(CSV_PATH)
    pv = flag_daylight(pv)
    capacity = float(pv["capacity"].iloc[0])

    wx = load_weather()
    pv = pv.join(wx, how="left")
    pv[WEATHER_CHANNELS] = pv[WEATHER_CHANNELS].ffill().bfill()

    first_full = pv.index.normalize().min() + pd.Timedelta(days=1)
    last_full = pv.index.normalize().max()
    pv = pv.loc[(pv.index >= first_full) & (pv.index < last_full)]

    pv_mat, day_idx = to_daily_matrix(pv["y"])
    is_day_mat, _ = to_daily_matrix(pv["is_daylight"])
    is_miss_mat, _ = to_daily_matrix(pv["is_missing"])
    pv_mat = np.nan_to_num(pv_mat, nan=0.0).astype(np.float32)
    is_day_mat = np.nan_to_num(is_day_mat, nan=0).astype(int)
    is_miss_mat = np.nan_to_num(is_miss_mat, nan=1).astype(int)
    wx_mats = []
    for ch in WEATHER_CHANNELS:
        m, _ = to_daily_matrix(pv[ch])
        wx_mats.append(np.nan_to_num(m, nan=0.0))
    wx_mat_raw = np.stack(wx_mats, axis=0).astype(np.float32)

    panel = build_hourly_panel(pv)

    fold_preds = fit_and_predict(
        pv_mat,
        wx_mat_raw,
        day_idx,
        is_day_mat,
        is_miss_mat,
        panel,
        capacity,
        n_folds=4,
    )

    agg = aggregate(fold_preds, capacity)
    print("\n=== Base models and ensembles (rolling backtest, daylight only) ===\n")
    with pd.option_context("display.max_colwidth", 40, "display.width", 130):
        print(agg.to_string(index=False))
    agg.to_csv(OUT_DIR / "pv_ensemble_metrics.csv", index=False)

    corr = residual_correlation(fold_preds)
    print("\n=== Residual correlation matrix between base models ===")
    print("(values close to 1 => ensembling helps little; close to 0 => big help)\n")
    print(corr.round(3).to_string())
    corr.to_csv(OUT_DIR / "pv_ensemble_residual_corr.csv")

    plot_bar(agg, OUT_DIR / "pv_ensemble_bar.png")

    best = agg.iloc[0]
    base_rows = agg[agg["model"].isin(["CNN", "Ridge", "GBM"])]
    best_single = base_rows.sort_values("nRMSE_%cap").iloc[0]
    print("\n=== Verdict ===")
    print(
        f"  best base model  : {best_single['model']}  "
        f"R²={best_single['R2']:.3f}  nRMSE={best_single['nRMSE_%cap']:.2f}%"
    )
    print(
        f"  best ensemble    : {best['model']}  "
        f"R²={best['R2']:.3f}  nRMSE={best['nRMSE_%cap']:.2f}%"
    )
    r2_gain = (best["R2"] - best_single["R2"]) * 100
    rmse_gain = best_single["nRMSE_%cap"] - best["nRMSE_%cap"]
    print(f"  absolute gain    : +{r2_gain:.2f} pp R²  /  -{rmse_gain:.2f} pp nRMSE")


if __name__ == "__main__":
    main()
