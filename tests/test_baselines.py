from __future__ import annotations

import numpy as np
import pandas as pd

import baselines as b


def _frame(n: int = 96) -> pd.DataFrame:
    idx = pd.date_range("2025-06-01", periods=n, freq="h", tz="UTC")
    hour = idx.hour
    y = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None) * 1000.0
    poa = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None) * 900.0
    return pd.DataFrame({"y": y, "poa_global": poa}, index=idx)


def test_persistence_uses_value_24h_earlier() -> None:
    df = _frame()
    test_index = df.index[48:]
    pred = b.persistence_24h(df, test_index)
    expected = df["y"].reindex(test_index - pd.Timedelta(hours=24)).to_numpy()
    assert np.allclose(pred, np.clip(expected, 0, None))


def test_persistence_falls_back_to_an_earlier_day_not_zero() -> None:
    df = _frame()
    df.loc[df.index[24:48], "y"] = np.nan
    test_index = df.index[48:72]
    pred = b.persistence_24h(df, test_index)
    # The 24 h lag is missing, so the forecast comes from two days earlier.
    expected = df["y"].reindex(test_index - pd.Timedelta(days=2)).to_numpy()
    assert np.allclose(pred, np.clip(expected, 0, None))


def test_persistence_falls_back_to_climatology_when_lookback_is_exhausted() -> None:
    df = _frame()
    train = df.iloc[:48]
    # No prior day exists at all for the first 24 hours.
    pred = b.persistence_24h(df, df.index[:24], train)
    assert np.allclose(pred, b.climatology(train, df.index[:24]))


def test_climatology_is_fitted_on_train_only() -> None:
    df = _frame()
    train = df.iloc[:48]
    test_index = df.index[48:]
    pred = b.climatology(train, test_index)
    # Prediction for a given hour equals the train mean for that hour.
    table = train.groupby(train.index.hour)["y"].mean()
    for value, hour in zip(pred, test_index.hour, strict=False):
        assert np.isclose(value, table.loc[hour])


def test_climatology_unseen_month_uses_hour_not_overall_mean() -> None:
    df = _frame()
    train = df.iloc[:48]
    future = pd.date_range("2025-11-01", periods=24, freq="h", tz="UTC")
    pred = b.climatology(train, future)
    by_hour = train.groupby(train.index.hour)["y"].mean()
    assert np.allclose(pred, np.clip(by_hour.loc[future.hour].to_numpy(), 0, None))
    assert not np.allclose(pred, float(train["y"].mean()))


def test_smart_persistence_non_negative_and_shaped() -> None:
    df = _frame()
    test_index = df.index[48:]
    pred = b.smart_persistence(df, test_index, capacity=1000.0)
    assert pred.shape == (len(test_index),)
    assert np.all(pred >= 0.0)


def test_smart_persistence_uses_the_capacity_it_is_given() -> None:
    df = _frame()
    test_index = df.index[48:]
    # A capacity low enough to push the carried ratio onto its 1.5 bound stops
    # the capacity from cancelling, which is why the fold-specific value has to
    # be passed in rather than a full-record one.
    bound = b.smart_persistence(df, test_index, capacity=500.0)
    free = b.smart_persistence(df, test_index, capacity=1600.0)
    assert not np.allclose(bound, free)


def test_baseline_predictions_never_forecast_zero_in_daylight() -> None:
    df = _frame()
    df.loc[df.index[24:72], "y"] = np.nan
    train = df.iloc[:24]
    test_index = df.index[48:]
    preds = b.baseline_predictions(df, train, test_index, capacity=1000.0)
    daylight = df["poa_global"].reindex(test_index).to_numpy() > 50
    for name, values in preds.items():
        assert np.all(values[daylight] > 0.0), name


def test_skill_score_signs() -> None:
    assert b.skill_score(50.0, 100.0) == 0.5
    assert b.skill_score(100.0, 100.0) == 0.0
    assert b.skill_score(150.0, 100.0) < 0.0
    assert np.isnan(b.skill_score(50.0, 0.0))


def test_add_skill_columns() -> None:
    metrics = pd.DataFrame(
        {
            "mode": ["KFOLD", "KFOLD"],
            "subset": ["daylight", "daylight"],
            "model": ["NNLSStack", "SmartPersistence"],
            "RMSE": [200.0, 400.0],
        }
    )
    out = b.add_skill_columns(metrics, reference="SmartPersistence")
    col = "skill_vs_SmartPersistence"
    assert col in out.columns
    nnls = out[out["model"] == "NNLSStack"][col].iloc[0]
    assert np.isclose(nnls, 0.5)
