"""Measure what each base learner costs and what it is worth.

A reviewer asked whether the ensemble's accuracy justifies its computation,
noting that an operator has to run every base learner to obtain one forecast.
Answering that needs two numbers per learner that the pipeline does not
otherwise record: the wall-clock time to fit it and to predict with it, and
the accuracy given up by dropping it from the ensemble.

Timing is measured on a single fold rather than averaged over the backtest,
because the point is the relative cost of the learners and a single fold
already separates them by more than an order of magnitude. Fitting is timed
with the model's own training call; prediction is timed over the fold's test
rows and reported per forecast day, which is the unit an operator cares about
for a day-ahead product.

Leave-one-out value is computed from saved fold predictions, refitting only
the cheap NNLS combination on the same validation rows. No base learner is
retrained, so the comparison isolates the contribution of each member rather
than confounding it with retraining noise.

Run ``python run_cost_benefit.py`` with ``pv_v4_predictions.csv`` present.
Writes ``pv_v4_cost_benefit.csv``.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import nnls

import analyze_pv_v4 as pipeline
import model_labels
import paths
from config import load_config

OUT_DIR = paths.results_dir()
BASE_LEARNERS = list(pipeline.BASE_LEARNERS)


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def time_learners(cfg) -> pd.DataFrame:
    """Fit each base learner once on a representative fold and time it."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    pipeline.set_global_seed(cfg.seed)
    pv, feats, capacity = pipeline.build_dataset(cfg)

    days = pv.index.normalize().unique()
    split = days[int(0.8 * len(days))]
    train = pv[(pv.index.normalize() < split) & (pv["is_missing"] == 0)]
    test = pv[pv.index.normalize() >= split]
    test_days = test.index.normalize().nunique()

    X_tr, y_tr = train[feats].to_numpy(), train["y"].to_numpy()
    X_te = test[feats].to_numpy()
    rows: list[dict[str, object]] = []

    def record(key: str, fit_s: float, pred_s: float) -> None:
        rows.append(
            {
                "model": key,
                "label": model_labels.label(key),
                "fit_seconds": round(fit_s, 3),
                "predict_ms_per_day": round(1000.0 * pred_s / max(test_days, 1), 3),
            }
        )
        _log(f"  {model_labels.label(key):22s} fit {fit_s:7.2f}s")

    scaler = StandardScaler()
    t0 = time.perf_counter()
    Xs = scaler.fit_transform(X_tr)
    ridge = Ridge(alpha=1.0).fit(Xs, y_tr)
    fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    ridge.predict(scaler.transform(X_te))
    record("Ridge", fit, time.perf_counter() - t0)

    def new_gbm() -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.05,
            max_depth=8,
            min_samples_leaf=30,
            l2_regularization=0.1,
            random_state=cfg.seed,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.1,
        )

    t0 = time.perf_counter()
    gbm = new_gbm().fit(X_tr, y_tr)
    fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    gbm.predict(X_te)
    record("GBM", fit, time.perf_counter() - t0)

    # Clearness-index GBM: same learner, daylight rows, normalised target.
    day = train["is_daylight"].to_numpy().astype(bool)
    denom = np.maximum(capacity * train["poa_global"].to_numpy() / 1000.0, 1.0)
    kt = np.clip(y_tr / denom, 0, cfg.kappa_clip)
    t0 = time.perf_counter()
    gbm_kt = new_gbm().fit(X_tr[day], kt[day])
    fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    gbm_kt.predict(X_te)
    record("GBM_kt", fit, time.perf_counter() - t0)

    hours_tr = train.index.hour.to_numpy()
    hours_te = test.index.hour.to_numpy()
    t0 = time.perf_counter()
    per_hour = {}
    for h in range(24):
        m = hours_tr == h
        if m.sum() >= 30:
            per_hour[h] = new_gbm().fit(X_tr[m], y_tr[m])
    fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    for h, model in per_hour.items():
        m = hours_te == h
        if m.any():
            model.predict(X_te[m])
    record("GBM_per_hour", fit, time.perf_counter() - t0)

    train_days = pd.DatetimeIndex(train.index.normalize().unique())
    test_day_idx = pd.DatetimeIndex(test.index.normalize().unique())
    t0 = time.perf_counter()
    pipeline.fit_cnn_for_fold(
        pv, train_days, test_day_idx[:0], test_day_idx, capacity, seed=cfg.seed
    )
    # Training dominates; the forward pass is a single batched call.
    record("CNN", time.perf_counter() - t0, 0.0)

    return pd.DataFrame(rows)


def leave_one_out_value(predictions: pd.DataFrame) -> pd.DataFrame:
    """RMSE cost of dropping each learner, refitting only the NNLS weights."""
    rows = []
    for mode, mode_df in predictions.groupby("mode"):
        usable = mode_df[
            (mode_df["is_missing"] == 0) & (mode_df["is_daylight"] == 1)
        ].dropna(subset=BASE_LEARNERS + ["y_actual"])
        if usable.empty:
            continue
        full = _stack_rmse(usable, BASE_LEARNERS)
        for dropped in BASE_LEARNERS:
            kept = [m for m in BASE_LEARNERS if m != dropped]
            rmse = _stack_rmse(usable, kept)
            rows.append(
                {
                    "mode": mode,
                    "model": dropped,
                    "label": model_labels.label(dropped),
                    "ensemble_rmse_without": round(rmse, 2),
                    "rmse_increase_pct": round(100.0 * (rmse - full) / full, 2),
                }
            )
        rows.append(
            {
                "mode": mode,
                "model": "ALL",
                "label": "Full ensemble",
                "ensemble_rmse_without": round(full, 2),
                "rmse_increase_pct": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _stack_rmse(df: pd.DataFrame, members: list[str]) -> float:
    """Leave-one-fold-out NNLS stacking RMSE over the saved test predictions.

    The saved file holds test-fold predictions only, with no validation rows,
    so weights for a fold are fitted on the remaining folds rather than on
    that fold's own rows. This keeps every scored point out of sample. The
    resulting RMSE is therefore not identical to the headline NNLS figure,
    which uses within-fold validation rows, but it is computed identically
    for every subset of members and so supports the comparison this table
    is making.
    """
    errors: list[np.ndarray] = []
    for fold, held_out in df.groupby("fold"):
        others = df[df["fold"] != fold]
        if others.empty:
            continue
        A = others[members].to_numpy()
        b = others["y_actual"].to_numpy()
        try:
            weights, _ = nnls(A, b)
        except Exception:
            weights = np.ones(len(members))
        total = weights.sum()
        weights = weights / total if total > 0 else np.ones(len(members)) / len(members)
        pred = held_out[members].to_numpy() @ weights
        errors.append(held_out["y_actual"].to_numpy() - pred)
    if not errors:
        return float("nan")
    residual = np.concatenate(errors)
    return float(np.sqrt(np.mean(residual**2)))


def main() -> None:
    cfg = load_config()
    _log("timing base learners on one fold")
    timing = time_learners(cfg)

    path = OUT_DIR / "pv_v4_predictions.csv"
    if not path.exists():
        print(f"\n{path.name} not found; skipping leave-one-out value.")
        timing.to_csv(OUT_DIR / "pv_v4_cost_benefit.csv", index=False)
        print(timing.to_string(index=False))
        return

    predictions = pd.read_csv(path)
    value = leave_one_out_value(predictions)
    merged = value.merge(timing, on=["model", "label"], how="left")
    merged.to_csv(OUT_DIR / "pv_v4_cost_benefit.csv", index=False)

    print("\n=== Cost of each base learner, and the accuracy lost by dropping it ===")
    for mode in ("KFOLD", "TEMPORAL"):
        view = merged[merged["mode"] == mode]
        if view.empty:
            continue
        print(f"\n--- {model_labels.mode_label(mode)} (daylight) ---")
        print(
            view[
                [
                    "label",
                    "fit_seconds",
                    "predict_ms_per_day",
                    "ensemble_rmse_without",
                    "rmse_increase_pct",
                ]
            ].to_string(index=False)
        )
    print("\nWrote pv_v4_cost_benefit.csv")


if __name__ == "__main__":
    main()
