"""2D CNN on a (day x hour x channel) reshape of the PV + weather data.

Input tensor layout
-------------------
For each target day d we build an image-like input of shape
    (channels, days, hours) = (C, 8, 24)
where the 8 days are the 7 days preceding d and day d itself, and
channels are:
    0: PV (kW, normalised), zero on day d (unknown at forecast time)
    1: is_past mask           (1 for days 0..6, 0 for day 7)
    2: PV availability mask  (0 where historical PV is missing/held out)
    3..8: 6 weather variables from the configured Open-Meteo archive
          product (known for all 8 days; retrospective weather upper bound)

Target: 24 hourly PV values for day d, normalised.

Training protocol
-----------------
Rolling-origin: the whole year is split into 4 windows. For each window
we retrain on all prior history and evaluate the next ~60 days. Metrics
are reported on test daylight hours only, exactly like the GBM backtest.
"""

from __future__ import annotations

import os
import random

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

import paths

CSV_PATH = paths.PV_CSV
WEATHER_CACHE = paths.WEATHER_DIR / "weather_cache.csv"
OUT_DIR = paths.results_dir()

WEATHER_CHANNELS = [
    "shortwave_radiation",
    "direct_normal_irradiance",
    "cloud_cover",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(4)


def set_global_seed(seed: int = 0) -> None:
    """Seed Python, NumPy and PyTorch for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def flag_daylight(df: pd.DataFrame, pct: float = 0.05) -> pd.DataFrame:
    capacity = df["y"].quantile(0.999)
    df["capacity"] = capacity
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    med = df.groupby(["month", "hour"])["y"].transform("median")
    df["is_daylight"] = (med > pct * capacity).astype(int)
    return df


def load_weather() -> pd.DataFrame:
    if not WEATHER_CACHE.exists():
        raise FileNotFoundError(
            f"{WEATHER_CACHE.name} not found. Run analyze_pv_with_weather.py "
            "first to populate the cache."
        )
    w = pd.read_csv(WEATHER_CACHE, parse_dates=["time"])
    w["time"] = pd.to_datetime(w["time"], utc=True)
    w = w.set_index("time").sort_index()
    return w[WEATHER_CHANNELS]


def to_daily_matrix(series: pd.Series) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Reshape an hourly series to (n_days, 24). Only keeps whole UTC days."""
    s = series.copy()
    days = s.index.normalize()
    df = s.to_frame("v")
    df["day"] = days
    df["hour"] = s.index.hour
    wide = df.pivot_table(index="day", columns="hour", values="v", aggfunc="first")
    wide = wide.reindex(columns=range(24))
    mat = wide.to_numpy(dtype=np.float32)
    return mat, wide.index


def build_supervised_tensors(
    pv_mat: np.ndarray,
    wx_mat: np.ndarray,
    day_index: pd.DatetimeIndex,
    is_daylight_mat: np.ndarray,
    is_missing_mat: np.ndarray,
    capacity: float,
    history_days: int = 7,
    hidden_pv_days: pd.DatetimeIndex | None = None,
    include_availability_channel: bool = True,
    weight_missing_targets: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    pd.DatetimeIndex,
]:
    """Create supervised day-by-hour tensors.

    ``hidden_pv_days`` masks PV observations from held-out days wherever they
    appear inside another sample's historical context. This prevents random
    day-fold labels from leaking into CNN training inputs.

    Missing PV observations are filled with zero, which is indistinguishable
    from a genuine night-time zero unless the network is told otherwise. It is
    told in two separate ways, and both can be switched off here so that their
    contributions can be measured independently:

    ``include_availability_channel``
        Adds a binary channel marking which historical PV values were actually
        observed, so a filled zero in the input is distinguishable from a real
        one. Switching this off removes the network's only way to tell them
        apart on the input side.
    ``weight_missing_targets``
        By default a missing target hour gets loss weight zero, so imputed
        zeros never contribute a training gradient. Setting this to ``True``
        gives them the same weight as observed hours, which is the naive
        fill-with-zero approach.

    Returns
    -------
    X
        Inputs with channels: normalised PV, past/future indicator, the
        optional PV-availability indicator, and the weather channels.
    Y
        Normalised target-day PV.
    W
        Loss weights: 0 for missing targets, 1 for valid night targets and 4
        for valid daylight targets.
    E
        Daylight-and-valid evaluation mask.
    target_days
        Calendar day corresponding to each sample.
    """
    n_days, n_hours = pv_mat.shape
    window = history_days + 1

    pv_input = np.array(pv_mat, copy=True)
    pv_available = (np.asarray(is_missing_mat) == 0).astype(np.float32)
    if hidden_pv_days is not None and len(hidden_pv_days):
        hidden = day_index.isin(pd.DatetimeIndex(hidden_pv_days).normalize())
        pv_input[hidden] = 0.0
        pv_available[hidden] = 0.0

    pv_norm = np.nan_to_num(pv_input / capacity, nan=0.0).astype(np.float32)
    target_norm = np.nan_to_num(pv_mat / capacity, nan=0.0).astype(np.float32)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    eval_masks: list[np.ndarray] = []
    target_days: list[pd.Timestamp] = []

    for d in range(history_days, n_days):
        past_pv = pv_norm[d - history_days : d]
        pv_block = np.concatenate(
            [past_pv, np.zeros((1, n_hours), dtype=np.float32)], axis=0
        )

        is_past = np.ones((window, n_hours), dtype=np.float32)
        is_past[-1] = 0.0

        past_available = pv_available[d - history_days : d]
        availability_block = np.concatenate(
            [past_available, np.zeros((1, n_hours), dtype=np.float32)], axis=0
        )

        wx_block = wx_mat[:, d - history_days : d + 1, :]
        blocks = [pv_block[None], is_past[None]]
        if include_availability_channel:
            blocks.append(availability_block[None])
        blocks.append(wx_block)
        channels = np.concatenate(blocks, axis=0).astype(np.float32)

        valid = is_missing_mat[d] == 0
        daylight = is_daylight_mat[d] == 1
        observed = (
            np.ones_like(valid, dtype=np.float32)
            if weight_missing_targets
            else valid.astype(np.float32)
        )
        loss_weight = observed * (1.0 + 3.0 * daylight.astype(np.float32))
        eval_mask = valid & daylight

        xs.append(channels)
        ys.append(target_norm[d])
        weights.append(loss_weight)
        eval_masks.append(eval_mask.astype(np.float32))
        target_days.append(pd.Timestamp(day_index[d]))

    X = torch.from_numpy(np.stack(xs))
    Y = torch.from_numpy(np.stack(ys))
    W = torch.from_numpy(np.stack(weights))
    E = torch.from_numpy(np.stack(eval_masks))
    return X, Y, W, E, pd.DatetimeIndex(target_days)


def build_weather_tensor(
    wx_df: pd.DataFrame,
    day_index: pd.DatetimeIndex,
    train_stats: tuple | None = None,
) -> tuple[np.ndarray, tuple]:
    mats = []
    for ch in WEATHER_CHANNELS:
        mat, _ = to_daily_matrix(wx_df[ch])
        mat = mat.reshape(len(day_index), 24)
        mats.append(mat)
    wx = np.stack(mats, axis=0).astype(np.float32)

    if train_stats is None:
        mean = wx.mean(axis=(1, 2), keepdims=True)
        std = wx.std(axis=(1, 2), keepdims=True) + 1e-6
    else:
        mean, std = train_stats
    wx = (wx - mean) / std
    return wx, (mean, std)


class PVCNN2D(nn.Module):
    def __init__(self, n_channels: int, n_days: int, n_hours: int = 24):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, (3, 5), padding=(1, 2)),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 32, (3, 5), padding=(1, 2)),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 16, (3, 3), padding=(1, 1)),
            nn.GroupNorm(4, 16),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Conv1d(16, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 1, 1),
        )

    def forward(self, x):
        h = self.cnn(x)
        h = h[:, :, -1, :]
        y = self.head(h).squeeze(1)
        return y


def weighted_mse(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Weighted MSE with zero weight for missing target measurements."""
    return ((pred - target) ** 2 * weight).sum() / (weight.sum() + 1e-6)


def train_cnn(
    X_tr: torch.Tensor,
    Y_tr: torch.Tensor,
    W_tr: torch.Tensor,
    X_va: torch.Tensor,
    Y_va: torch.Tensor,
    W_va: torch.Tensor,
    epochs: int = 120,
    batch_size: int = 32,
    lr: float = 2e-3,
    patience: int = 15,
    seed: int = 0,
) -> PVCNN2D:
    set_global_seed(seed)
    model = PVCNN2D(n_channels=X_tr.shape[1], n_days=X_tr.shape[2]).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ds = TensorDataset(X_tr, Y_tr, W_tr)
    generator = torch.Generator().manual_seed(seed)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )

    best_val = float("inf")
    best_state = None
    bad = 0
    for _epoch in range(epochs):
        model.train()
        for xb, yb, wb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            wb = wb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = weighted_mse(pred, yb, wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            pv = model(X_va.to(DEVICE)).cpu()
            vloss = weighted_mse(pv, Y_va, W_va).item()
        if vloss < best_val - 1e-5:
            best_val = vloss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@dataclass
class FoldResult:
    origin: pd.Timestamp
    model: str
    n_daylight: int
    r2: float
    mae: float
    rmse: float


def evaluate_fold(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray, capacity: float
) -> tuple[float, float, float]:
    m = mask.astype(bool) & np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 5:
        return float("nan"), float("nan"), float("nan")
    y, yh = y_true[m], y_pred[m]
    return (
        float(r2_score(y, yh)),
        float(mean_absolute_error(y, yh)),
        float(np.sqrt(mean_squared_error(y, yh))),
    )


def rolling_backtest(
    pv_mat,
    wx_mat_raw,
    day_idx,
    is_daylight_mat,
    is_missing_mat,
    capacity,
    n_folds=3,
    val_frac=0.15,
):
    n_days = pv_mat.shape[0]
    # first test window starts after ~4 months of training, so the CNN
    # actually has enough data to avoid degenerate BN/GroupNorm behaviour.
    first_test = 120 + 7
    fold_len = (n_days - first_test) // n_folds
    results: list[FoldResult] = []
    all_preds = []

    for fold in range(n_folds):
        test_start = first_test + fold * fold_len
        test_end = test_start + fold_len if fold < n_folds - 1 else n_days

        mean = np.nanmean(wx_mat_raw[:, :test_start, :], axis=(1, 2), keepdims=True)
        std = (
            np.nanstd(wx_mat_raw[:, :test_start, :], axis=(1, 2), keepdims=True) + 1e-6
        )
        wx_mat = (wx_mat_raw - mean) / std
        wx_mat = np.nan_to_num(wx_mat, nan=0.0).astype(np.float32)

        X, Y, W, E, target_days = build_supervised_tensors(
            pv_mat,
            wx_mat,
            day_idx,
            is_daylight_mat,
            is_missing_mat,
            capacity,
        )

        is_train = np.arange(len(target_days)) < test_start - 7
        is_test = (~is_train) & (np.arange(len(target_days)) < test_end - 7)

        train_idx = np.where(is_train)[0]
        rng = np.random.default_rng(0)
        perm = rng.permutation(train_idx)
        n_val = max(5, int(len(train_idx) * val_frac))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        te_idx = np.where(is_test)[0]

        X_tr, Y_tr, W_tr = X[tr_idx], Y[tr_idx], W[tr_idx]
        X_va, Y_va, W_va = X[val_idx], Y[val_idx], W[val_idx]
        X_te, Y_te, E_te = X[te_idx], Y[te_idx], E[te_idx]
        te_days = target_days[te_idx]

        _log(
            f"fold {fold + 1}/{n_folds}: origin~{te_days[0].date()} "
            f"train={len(tr_idx)} val={len(val_idx)} test={len(te_idx)}"
        )
        t0 = time.time()
        model = train_cnn(X_tr, Y_tr, W_tr, X_va, Y_va, W_va, seed=fold)
        _log(f"  fold {fold + 1} trained in {time.time() - t0:.1f}s")

        with torch.no_grad():
            pred_raw = model(X_te.to(DEVICE)).cpu().numpy()
        if not np.isfinite(pred_raw).all():
            bad = int(np.isnan(pred_raw).sum())
            _log(f"  WARNING fold {fold + 1}: {bad} NaN predictions — zeroing")
            pred_raw = np.nan_to_num(pred_raw, nan=0.0, posinf=1.0, neginf=0.0)
        pred = pred_raw * capacity
        actual = Y_te.numpy() * capacity
        masks = E_te.numpy()

        r2, mae, rmse = evaluate_fold(
            actual.reshape(-1),
            pred.reshape(-1),
            masks.reshape(-1),
            capacity,
        )
        results.append(FoldResult(te_days[0], "CNN2D", int(masks.sum()), r2, mae, rmse))
        _log(
            f"  fold {fold + 1} daylight R²={r2:.3f} "
            f"nRMSE={rmse / capacity * 100:.2f}% nMAE={mae / capacity * 100:.2f}%"
        )

        for i, d in enumerate(te_days):
            for h in range(24):
                if masks[i, h]:
                    all_preds.append(
                        {
                            "day": d,
                            "hour": h,
                            "y": float(actual[i, h]),
                            "yhat": float(pred[i, h]),
                        }
                    )

    preds_df = pd.DataFrame(all_preds)
    preds_df = preds_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["y", "yhat"])
    y, yh = preds_df["y"].to_numpy(), preds_df["yhat"].to_numpy()
    overall = {
        "model": "CNN2D",
        "n_test": len(preds_df),
        "R2": round(r2_score(y, yh), 4) if len(preds_df) else float("nan"),
        "MAE": round(mean_absolute_error(y, yh), 1) if len(preds_df) else float("nan"),
        "RMSE": round(float(np.sqrt(mean_squared_error(y, yh))), 1)
        if len(preds_df)
        else float("nan"),
        "nMAE_%cap": round(mean_absolute_error(y, yh) / capacity * 100, 2)
        if len(preds_df)
        else float("nan"),
        "nRMSE_%cap": round(
            float(np.sqrt(mean_squared_error(y, yh))) / capacity * 100, 2
        )
        if len(preds_df)
        else float("nan"),
    }
    return results, overall, preds_df


def plot_days(preds_df: pd.DataFrame, out_png: Path) -> None:
    err = (preds_df["y"] - preds_df["yhat"]).abs()
    daily = preds_df.assign(abs_err=err).groupby("day")["abs_err"].mean()
    best = daily.idxmin()
    worst = daily.idxmax()
    typical = (daily - daily.median()).abs().idxmin()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, day, tag in zip(
        axes,
        [best, typical, worst],
        ["Easiest test day", "Typical test day", "Hardest test day"],
        strict=False,
    ):
        seg = preds_df[preds_df["day"] == day].sort_values("hour")
        ax.plot(seg["hour"], seg["y"], "o-", color="steelblue", label="actual")
        ax.plot(seg["hour"], seg["yhat"], "o-", color="orange", label="CNN2D forecast")
        ax.set_title(f"{tag}\n{pd.Timestamp(day).date()}")
        ax.set_xlabel("hour of day")
        ax.legend()
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("PV (kW)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_scatter(preds_df: pd.DataFrame, capacity: float, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hexbin(preds_df["yhat"], preds_df["y"], gridsize=40, cmap="viridis", mincnt=1)
    lim = max(preds_df["y"].max(), preds_df["yhat"].max())
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    r2 = r2_score(preds_df["y"], preds_df["yhat"])
    rmse = float(np.sqrt(mean_squared_error(preds_df["y"], preds_df["yhat"])))
    ax.set_title(
        f"CNN2D test daylight: R²={r2:.3f}, "
        f"nRMSE={rmse / capacity * 100:.1f}% cap"
    )
    ax.set_xlabel("forecast ŷ")
    ax.set_ylabel("actual y")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> None:
    _log(f"device: {DEVICE}")
    pv = load_pv(CSV_PATH)
    pv = flag_daylight(pv)
    capacity = float(pv["capacity"].iloc[0])
    _log(f"loaded {len(pv)} hourly rows, capacity={capacity:.1f}")

    wx = load_weather()
    pv = pv.join(wx, how="left")
    pv[WEATHER_CHANNELS] = pv[WEATHER_CHANNELS].ffill().bfill()

    day_counts = pv.groupby(pv.index.normalize()).size()
    full_days_idx = day_counts[day_counts == 24].index
    pv = pv[pv.index.normalize().isin(full_days_idx)]
    _log(
        f"trimmed to full UTC days: {pv.index.min()} -> {pv.index.max()} "
        f"({len(pv)} rows, {len(full_days_idx)} days)"
    )

    pv_mat, day_idx = to_daily_matrix(pv["y"])
    is_day_mat, _ = to_daily_matrix(pv["is_daylight"])
    is_miss_mat, _ = to_daily_matrix(pv["is_missing"])

    pv_mat = np.nan_to_num(pv_mat, nan=0.0).astype(np.float32)
    is_day_mat = np.nan_to_num(is_day_mat, nan=0).astype(int)
    is_miss_mat = np.nan_to_num(is_miss_mat, nan=1).astype(int)

    wx_mats = []
    for ch in WEATHER_CHANNELS:
        m, _ = to_daily_matrix(pv[ch])
        m = np.nan_to_num(m, nan=0.0)
        wx_mats.append(m)
    wx_mat_raw = np.stack(wx_mats, axis=0).astype(np.float32)
    assert np.isfinite(pv_mat).all(), "pv_mat has NaN after trim"
    assert np.isfinite(wx_mat_raw).all(), "wx_mat has NaN after trim"
    _log(
        f"tensors: pv {pv_mat.shape}, wx {wx_mat_raw.shape}, "
        f"daylight cells {int(is_day_mat.sum())}"
    )

    folds, overall, preds_df = rolling_backtest(
        pv_mat,
        wx_mat_raw,
        day_idx,
        is_day_mat,
        is_miss_mat,
        capacity=capacity,
        n_folds=4,
    )

    print("\n=== CNN2D rolling-origin backtest (daylight only) ===\n")
    per_fold = pd.DataFrame(
        [
            {
                "origin": f.origin.date(),
                "n_daylight": f.n_daylight,
                "R2": round(f.r2, 4),
                "MAE": round(f.mae, 1),
                "RMSE": round(f.rmse, 1),
                "nMAE_%cap": round(f.mae / capacity * 100, 2),
                "nRMSE_%cap": round(f.rmse / capacity * 100, 2),
            }
            for f in folds
        ]
    )
    with pd.option_context("display.max_colwidth", 40, "display.width", 130):
        print(per_fold.to_string(index=False))
    print("\n-- aggregate over all folds --")
    print(pd.DataFrame([overall]).to_string(index=False))

    preds_df.to_csv(OUT_DIR / "pv_cnn2d_predictions.csv", index=False)
    per_fold.to_csv(OUT_DIR / "pv_cnn2d_per_fold.csv", index=False)
    plot_days(preds_df, OUT_DIR / "pv_cnn2d_example_days.png")
    plot_scatter(preds_df, capacity, OUT_DIR / "pv_cnn2d_scatter.png")

    gbm_rolling = (
        pd.read_csv(OUT_DIR / "pv_weather_rolling_backtest.csv")
        if (OUT_DIR / "pv_weather_rolling_backtest.csv").exists()
        else None
    )
    print("\n=== Comparison vs. the GBM rolling baseline ===")
    if gbm_rolling is not None:
        combined = pd.concat(
            [gbm_rolling, pd.DataFrame([overall])], ignore_index=True
        ).sort_values("nRMSE_%cap")
        print(combined.to_string(index=False))
    else:
        print(
            "(run analyze_pv_with_weather.py first to generate "
            "pv_weather_rolling_backtest.csv)"
        )


if __name__ == "__main__":
    main()
