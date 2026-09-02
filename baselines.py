"""Reference forecasting baselines and forecast-skill scores.

Solar-forecasting studies are normally judged against trivial reference
forecasts. Without them, headline error metrics cannot be interpreted, and a
complex model that merely matches persistence looks deceptively strong. This
module provides three standard references and a skill-score helper:

* ``persistence_24h`` - yesterday's measured value at the same hour.
* ``climatology`` - the training-set mean for each (month, hour), fitted only
  on training rows (no test leakage).
* ``smart_persistence`` - persistence of the clearness ratio, rescaled by the
  target hour's plane-of-array irradiance.

None of them is allowed to see a held-out measurement: the capacity that scales
the clearness ratio is the fold's training capacity, and every fallback used
when an input is unavailable is estimated from training rows.

Each forecaster returns an array aligned to ``test_index``. Baselines are kept
separate from the ensemble base learners so they never enter the stacking
weights; they are merged into the per-fold prediction dictionary only for
scoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASELINE_NAMES = ("Persistence", "Climatology", "SmartPersistence")


LOOKBACK_DAYS = 7


def _same_hour_lookback(
    series: pd.Series, test_index: pd.DatetimeIndex, max_days: int = LOOKBACK_DAYS
) -> np.ndarray:
    """Most recent available value at the same hour, searching back day by day.

    Returns NaN where nothing is available within ``max_days``. Every value it
    returns precedes the target hour by a whole number of days, so it is known
    when a day-ahead forecast is issued.
    """
    out = np.full(len(test_index), np.nan)
    for day in range(1, max_days + 1):
        need = ~np.isfinite(out)
        if not need.any():
            break
        candidate = series.reindex(
            test_index[need] - pd.Timedelta(days=day)
        ).to_numpy()
        filled = out[need]
        found = np.isfinite(candidate)
        filled[found] = candidate[found]
        out[need] = filled
    return out


def _hour_table(values: pd.Series, keys: pd.Index) -> pd.Series:
    return pd.DataFrame({"v": values.to_numpy(), "k": keys}).groupby("k")["v"].mean()


def climatology(
    train_df: pd.DataFrame, test_index: pd.DatetimeIndex
) -> np.ndarray:
    """Training-set mean PV for each (month, hour), evaluated on the test index.

    The lookup table is fitted only on training rows, so this reference is
    leakage-free. A (month, hour) pair absent from training falls back to the
    training mean for that hour across the months that are present, and only
    then to the overall training mean. The hour-of-day fallback matters under
    rolling-origin evaluation, where early folds are routinely asked about
    calendar months they have never seen but always know the diurnal shape.
    """
    train = train_df.copy()
    by_month_hour = (
        pd.DataFrame(
            {
                "y": train["y"].to_numpy(),
                "month": train.index.month,
                "hour": train.index.hour,
            }
        )
        .groupby(["month", "hour"])["y"]
        .mean()
    )
    by_hour = _hour_table(train["y"], train.index.hour)
    overall = float(train["y"].mean())
    values = [
        float(by_month_hour.get((m, h), by_hour.get(h, overall)))
        for m, h in zip(test_index.month, test_index.hour, strict=False)
    ]
    return np.clip(np.asarray(values, dtype=float), 0.0, None)


def persistence_24h(
    pv_full: pd.DataFrame,
    test_index: pd.DatetimeIndex,
    train_df: pd.DataFrame | None = None,
) -> np.ndarray:
    """Forecast PV(t) with the measured value 24 hours earlier.

    When that hour was not recorded, the forecast falls back to the most recent
    observation at the same hour within ``LOOKBACK_DAYS``, and then to the
    training climatology. Substituting zero instead would put a confident
    zero-generation forecast into daylight hours and inflate the reference
    error, which flatters every model scored against it.
    """
    out = _same_hour_lookback(pv_full["y"], test_index)
    if train_df is not None:
        out = np.where(np.isfinite(out), out, climatology(train_df, test_index))
    return np.clip(np.nan_to_num(out, nan=0.0), 0.0, None)


def smart_persistence(
    pv_full: pd.DataFrame,
    test_index: pd.DatetimeIndex,
    capacity: float,
    train_df: pd.DataFrame | None = None,
    poa_col: str = "poa_global",
) -> np.ndarray:
    """Persistence of the clearness ratio, rescaled by target-day irradiance.

    The generation envelope is approximated by ``capacity * POA / 1000``. The
    most recent available ratio of measured power to that envelope at the same
    hour is carried forward and multiplied by the target hour's envelope, which
    removes the deterministic solar cycle from the persistence forecast and
    makes it a stronger reference than naive persistence near sunrise and
    sunset. A ratio that is unavailable within ``LOOKBACK_DAYS`` falls back to
    the mean training ratio for that hour.

    ``capacity`` should be the fold's training capacity: it does not cancel
    between the ratio and the rescaling once the ratio is bounded, so passing a
    full-record value would let held-out hours influence the reference.
    """
    envelope_full = np.maximum(capacity * pv_full[poa_col] / 1000.0, 1.0)
    ratio_full = (pv_full["y"] / envelope_full).clip(0.0, 1.5)

    ratio_prev = _same_hour_lookback(ratio_full, test_index)
    if train_df is not None:
        train_ratio = ratio_full.reindex(train_df.index)
        by_hour = _hour_table(train_ratio, train_df.index.hour)
        overall = float(train_ratio.mean())
        fallback = np.asarray(
            [float(by_hour.get(h, overall)) for h in test_index.hour], dtype=float
        )
        ratio_prev = np.where(np.isfinite(ratio_prev), ratio_prev, fallback)
    ratio_prev = np.nan_to_num(ratio_prev, nan=0.0)

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
    """Compute all reference baselines for one fold's test index.

    ``capacity`` is the fold's training capacity, so no reference forecast is
    informed by a held-out measurement.
    """
    return {
        "Persistence": persistence_24h(pv_full, test_index, train_df),
        "Climatology": climatology(train_df, test_index),
        "SmartPersistence": smart_persistence(
            pv_full, test_index, capacity, train_df
        ),
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
