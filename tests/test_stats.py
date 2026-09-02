from __future__ import annotations

import numpy as np
import pandas as pd

from stats_tests import diebold_mariano, ensemble_gain, paired_day_bootstrap


def _days(n_days: int, per_day: int) -> np.ndarray:
    return np.repeat(pd.date_range("2025-01-01", periods=n_days), per_day)


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


def test_bootstrap_detects_a_better_forecast() -> None:
    rng = np.random.default_rng(2)
    n_days, per_day = 120, 12
    days = _days(n_days, per_day)
    y = rng.normal(size=n_days * per_day)
    # A shared per-day shock makes hours within a day dependent.
    shock = np.repeat(rng.normal(scale=0.5, size=n_days), per_day)
    good = y + shock + rng.normal(scale=0.2, size=y.size)
    bad = y + shock + rng.normal(scale=0.8, size=y.size)
    res = paired_day_bootstrap(y, good, bad, days)
    assert res.n_days == n_days
    assert res.rel_rmse_gain_pct > 0
    assert res.ci_low_pct > 0
    assert res.p_value < 0.01


def test_bootstrap_does_not_separate_equivalent_forecasts() -> None:
    rng = np.random.default_rng(3)
    n_days, per_day = 120, 12
    days = _days(n_days, per_day)
    y = rng.normal(size=n_days * per_day)
    a = y + rng.normal(scale=0.5, size=y.size)
    b = y + rng.normal(scale=0.5, size=y.size)
    res = paired_day_bootstrap(y, a, b, days)
    assert res.ci_low_pct < 0 < res.ci_high_pct
    assert res.p_value > 0.05


def test_bootstrap_p_value_stays_a_probability() -> None:
    rng = np.random.default_rng(4)
    days = _days(50, 10)
    y = rng.normal(size=500)
    pred = y + rng.normal(scale=0.5, size=500)
    res = paired_day_bootstrap(y, pred, pred.copy(), days)
    assert res.rel_rmse_gain_pct == 0.0
    assert 0.0 <= res.p_value <= 1.0


def test_ensemble_gain() -> None:
    gain = ensemble_gain(rmse_ensemble=90.0, rmse_best_base=100.0)
    assert np.isclose(gain["abs_rmse_gain"], 10.0)
    assert np.isclose(gain["rel_rmse_gain_pct"], 10.0)
    nan_gain = ensemble_gain(90.0, 0.0)
    assert np.isnan(nan_gain["rel_rmse_gain_pct"])
