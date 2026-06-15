"""Reference forecasting baselines and forecast-skill scores.

Solar-forecasting studies are normally judged against trivial reference
forecasts. Without them, headline error metrics cannot be interpreted, and a
complex model that merely matches persistence looks deceptively strong. This
module provides three standard references and a skill-score helper:

* ``persistence_24h`` - yesterday's measured value at the same hour.
* ``climatology`` - the training-set mean for each (month, hour), fitted only
  on training rows (no test leakage).
* ``smart_persistence`` - clear-sky (clearness) persistence: yesterday's
  clearness ratio applied to today's clear-sky envelope.

Each forecaster returns an array aligned to ``test_index``. Baselines are kept
separate from the ensemble base learners so they never enter the stacking
weights; they are merged into the per-fold prediction dictionary only for
scoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASELINE_NAMES = ("Persistence", "Climatology", "SmartPersistence")


def persistence_24h(pv_full: pd.DataFrame, test_index: pd.DatetimeIndex) -> np.ndarray:
    """Forecast PV(t) with the measured value 24 hours earlier.

    Missing or out-of-range lookups fall back to zero. This is the naive
    diurnal-persistence reference and uses only information available one day
    before the target hour.
    """
    y = pv_full["y"]
    lagged = y.reindex(test_index - pd.Timedelta(hours=24))
    out = np.nan_to_num(lagged.to_numpy(), nan=0.0)
    return np.clip(out, 0.0, None)


def climatology(
    train_df: pd.DataFrame, test_index: pd.DatetimeIndex
) -> np.ndarray:
    """Training-set mean PV for each (month, hour), evaluated on the test index.

    The lookup table is fitted only on training rows, so this reference is
    leakage-free. Unseen (month, hour) combinations fall back to the overall
    training mean.
    """
    train = train_df.copy()
    months = train.index.month
    hours = train.index.hour
    table = (
        pd.DataFrame({"y": train["y"].to_numpy(), "month": months, "hour": hours})
        .groupby(["month", "hour"])["y"]
        .mean()
    )
    overall = float(train["y"].mean())
    keys = list(zip(test_index.month, test_index.hour, strict=False))
    values = [float(table.get((m, h), overall)) for m, h in keys]
    return np.clip(np.asarray(values, dtype=float), 0.0, None)


def smart_persistence(
    pv_full: pd.DataFrame,
    test_index: pd.DatetimeIndex,
    capacity: float,
    poa_col: str = "poa_global",
) -> np.ndarray:
    """Clear-sky (clearness) persistence baseline.

    The clear-sky power envelope is approximated by ``capacity * POA / 1000``.
    Yesterday's clearness ratio (measured power divided by that envelope) is
    carried forward and multiplied by today's envelope. This removes the
    deterministic solar cycle from the persistence forecast and is a stronger
    reference than naive persistence around sunrise and sunset.
    """
    envelope_full = np.maximum(capacity * pv_full[poa_col] / 1000.0, 1.0)
    ratio_full = (pv_full["y"] / envelope_full).clip(0.0, 1.5)

    ratio_prev = ratio_full.reindex(test_index - pd.Timedelta(hours=24))
    ratio_prev = np.nan_to_num(ratio_prev.to_numpy(), nan=0.0)

    envelope_today = pv_full[poa_col].reindex(test_index)
    envelope_today = np.maximum(
        capacity * np.nan_to_num(envelope_today.to_numpy(), nan=0.0) / 1000.0, 0.0
    )
    return np.clip(ratio_prev * envelope_today, 0.0, None)


def baseline_predictions(
    pv_full: pd.DataFrame,
    train_df: pd.DataFrame,
    test_index: pd.DatetimeIndex,
    capacity: float,
) -> dict[str, np.ndarray]:
    """Compute all reference baselines for one fold's test index."""
    return {
        "Persistence": persistence_24h(pv_full, test_index),
        "Climatology": climatology(train_df, test_index),
        "SmartPersistence": smart_persistence(pv_full, test_index, capacity),
    }


def skill_score(rmse_model: float, rmse_reference: float) -> float:
    """Forecast skill score ``s = 1 - RMSE_model / RMSE_reference``.

    ``s = 0`` means no improvement over the reference, ``s = 1`` is a perfect
    forecast, and ``s < 0`` means the model is worse than the reference.
    """
    if not np.isfinite(rmse_reference) or rmse_reference <= 0:
        return float("nan")
    return float(1.0 - rmse_model / rmse_reference)


def add_skill_columns(
    metrics: pd.DataFrame, reference: str = "SmartPersistence"
) -> pd.DataFrame:
    """Append a skill-score column to a metrics table.

    Skill is computed per (mode, subset) against the RMSE of ``reference``.
    Rows for modes/subsets where the reference is absent receive NaN.
    """
    out = metrics.copy()
    out["skill_vs_" + reference] = np.nan
    group_cols = [c for c in ("mode", "subset") if c in out.columns]
    if not group_cols:
        return out
    for _, group in out.groupby(group_cols):
        ref_rows = group[group["model"] == reference]
        if ref_rows.empty:
            continue
        ref_rmse = float(ref_rows["RMSE"].iloc[0])
        for idx in group.index:
            out.loc[idx, "skill_vs_" + reference] = skill_score(
                float(out.loc[idx, "RMSE"]), ref_rmse
            )
    return out
