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


def test_persistence_missing_lookup_is_zero() -> None:
    df = _frame()
    # First day has no 24 h-earlier value -> zero fallback.
    pred = b.persistence_24h(df, df.index[:24])
    assert np.all(pred == 0.0)


def test_climatology_is_fitted_on_train_only() -> None:
    df = _frame()
    train = df.iloc[:48]
    test_index = df.index[48:]
    pred = b.climatology(train, test_index)
    # Prediction for a given hour equals the train mean for that hour.
    table = train.groupby(train.index.hour)["y"].mean()
    for value, hour in zip(pred, test_index.hour, strict=False):
        assert np.isclose(value, table.loc[hour])


def test_smart_persistence_non_negative_and_shaped() -> None:
    df = _frame()
    test_index = df.index[48:]
    pred = b.smart_persistence(df, test_index, capacity=1000.0)
    assert pred.shape == (len(test_index),)
    assert np.all(pred >= 0.0)


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
