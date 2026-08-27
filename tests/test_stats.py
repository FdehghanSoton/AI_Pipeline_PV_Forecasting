from __future__ import annotations

import numpy as np

from stats_tests import diebold_mariano, ensemble_gain


def test_identical_forecasts_have_no_difference() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    pred = y + rng.normal(scale=0.5, size=200)
    res = diebold_mariano(y, pred, pred.copy())
    assert res.better in {"tie", "no significant difference"}


def test_clearly_better_forecast_is_detected() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(size=400)
    good = y + rng.normal(scale=0.1, size=400)
    bad = y + rng.normal(scale=2.0, size=400)
    res = diebold_mariano(y, good, bad, name_a="good", name_b="bad")
    assert res.statistic < 0  # good has lower loss
    assert res.p_value < 0.05
    assert res.better == "good"


def test_short_series_returns_undetermined() -> None:
    y = np.arange(4.0)
    res = diebold_mariano(y, y, y)
    assert res.better == "undetermined"


def test_ensemble_gain() -> None:
    gain = ensemble_gain(rmse_ensemble=90.0, rmse_best_base=100.0)
    assert np.isclose(gain["abs_rmse_gain"], 10.0)
    assert np.isclose(gain["rel_rmse_gain_pct"], 10.0)
    nan_gain = ensemble_gain(90.0, 0.0)
    assert np.isnan(nan_gain["rel_rmse_gain_pct"])
