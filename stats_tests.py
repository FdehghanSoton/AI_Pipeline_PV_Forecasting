"""Statistical comparison of forecasts.

The ensemble gains in this study are small, so they need a significance
statement rather than a bare ranking. Hourly PV errors are strongly dependent
within a day, and the forecasts being compared are produced one target day at
a time, so the day is the natural independent unit here.

``paired_day_bootstrap`` therefore resamples whole target days.
``diebold_mariano`` is kept because it is the conventional test, but its hourly
form assumes serially uncorrelated loss differentials; pass loss differentials
already aggregated to the day, or prefer the bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class DMResult:
    statistic: float
    p_value: float
    n: int
    better: str
    loss: str

    def as_dict(self) -> dict[str, object]:
        return {
            "dm_statistic": self.statistic,
            "p_value": self.p_value,
            "n": self.n,
            "better": self.better,
            "loss": self.loss,
        }


def diebold_mariano(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    loss: str = "squared",
    name_a: str = "A",
    name_b: str = "B",
) -> DMResult:
    """Diebold-Mariano test of equal predictive accuracy of two forecasts.

    A negative statistic means ``pred_a`` has the lower average loss. The
    two-sided p-value uses a Student-t reference distribution with ``n - 1``
    degrees of freedom, which is a small-sample correction over the asymptotic
    normal form. Forecasts are assumed to share the same target series.

    Parameters
    ----------
    loss
        ``"squared"`` or ``"absolute"`` loss differential.
    """
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(pred_a) & np.isfinite(pred_b)
    y_true, pred_a, pred_b = y_true[mask], pred_a[mask], pred_b[mask]
    n = int(len(y_true))
    if n < 8:
        return DMResult(float("nan"), float("nan"), n, "undetermined", loss)

    if loss == "squared":
        err_a = (y_true - pred_a) ** 2
        err_b = (y_true - pred_b) ** 2
    elif loss == "absolute":
        err_a = np.abs(y_true - pred_a)
        err_b = np.abs(y_true - pred_b)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    diff = err_a - err_b
    mean_diff = float(np.mean(diff))
    # Newey-West style variance with a single lag is unnecessary for pooled,
    # roughly serially-uncorrelated daily residuals; use the sample variance.
    var_diff = float(np.var(diff, ddof=1))
    if var_diff <= 0:
        return DMResult(float("nan"), float("nan"), n, "tie", loss)

    statistic = mean_diff / np.sqrt(var_diff / n)
    p_value = float(2.0 * stats.t.sf(abs(statistic), df=n - 1))
    if p_value < 0.05:
        better = name_a if mean_diff < 0 else name_b
    else:
        better = "no significant difference"
    return DMResult(float(statistic), p_value, n, better, loss)


@dataclass(frozen=True)
class BootstrapResult:
    rel_rmse_gain_pct: float
    ci_low_pct: float
    ci_high_pct: float
    p_value: float
    n_days: int
    n: int

    def as_dict(self) -> dict[str, object]:
        return {
            "rel_rmse_gain_pct": self.rel_rmse_gain_pct,
            "ci_low_pct": self.ci_low_pct,
            "ci_high_pct": self.ci_high_pct,
            "p_value": self.p_value,
            "n_days": self.n_days,
            "n": self.n,
        }


def paired_day_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    days: np.ndarray,
    n_boot: int = 10000,
    seed: int = 0,
    block_days: int = 1,
) -> BootstrapResult:
    """Paired block bootstrap of the RMSE difference on the saved forecasts.

    Both forecasts are scored on exactly the same hours. The procedure
    resamples already-generated out-of-fold predictions: it captures
    variability from which target days are drawn, not from retraining. Each
    replicate draws contiguous blocks of ``block_days`` calendar days with
    replacement, so dependence inside a day, and across neighbouring days when
    ``block_days > 1``, is carried through the resampling.

    The reported quantity is the percentage RMSE reduction of ``pred_a``
    relative to ``pred_b``, with a percentile interval. The p-value is the
    two-sided percentile bootstrap value for the mean squared-loss
    differential, bounded below by ``1 / n_boot``.
    """
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(pred_a) & np.isfinite(pred_b)
    y_true, pred_a, pred_b = y_true[mask], pred_a[mask], pred_b[mask]
    days = np.asarray(days)[mask]

    loss_a = (y_true - pred_a) ** 2
    loss_b = (y_true - pred_b) ** 2
    codes, uniques = pd.factorize(days)
    n_days = int(codes.max()) + 1 if len(codes) else 0
    if n_days < 8:
        nan = float("nan")
        return BootstrapResult(nan, nan, nan, nan, n_days, int(len(y_true)))

    order = np.argsort(pd.to_datetime(uniques))
    inv = np.empty_like(order)
    inv[order] = np.arange(n_days)
    codes = inv[codes]

    sum_a = np.bincount(codes, weights=loss_a, minlength=n_days)
    sum_b = np.bincount(codes, weights=loss_b, minlength=n_days)
    counts = np.bincount(codes, minlength=n_days).astype(float)

    width = max(int(block_days), 1)
    n_blocks = int(np.ceil(n_days / width))
    pad = n_blocks * width - n_days
    if pad:
        sum_a = np.concatenate([sum_a, np.zeros(pad)])
        sum_b = np.concatenate([sum_b, np.zeros(pad)])
        counts = np.concatenate([counts, np.zeros(pad)])
    block_a = sum_a.reshape(n_blocks, width).sum(axis=1)
    block_b = sum_b.reshape(n_blocks, width).sum(axis=1)
    block_n = counts.reshape(n_blocks, width).sum(axis=1)

    def rel_gain(sa: float, sb: float, n: float) -> float:
        rmse_a, rmse_b = np.sqrt(sa / n), np.sqrt(sb / n)
        return float((rmse_b - rmse_a) / rmse_b * 100.0)

    observed = rel_gain(sum_a.sum(), sum_b.sum(), counts.sum())

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_blocks, size=(n_boot, n_blocks))
    boot_a = block_a[draws].sum(axis=1)
    boot_b = block_b[draws].sum(axis=1)
    boot_n = block_n[draws].sum(axis=1)
    gains = (1.0 - np.sqrt(boot_a / boot_n) / np.sqrt(boot_b / boot_n)) * 100.0
    mean_diff = (boot_a - boot_b) / boot_n

    ci_low, ci_high = np.percentile(gains, [2.5, 97.5])
    tail = min(float(np.mean(mean_diff >= 0.0)), float(np.mean(mean_diff <= 0.0)))
    p_value = min(max(2.0 * tail, 1.0 / n_boot), 1.0)
    return BootstrapResult(
        observed, float(ci_low), float(ci_high), p_value, n_days, int(len(y_true))
    )


def ensemble_gain(
    rmse_ensemble: float, rmse_best_base: float
) -> dict[str, float]:
    """Relative RMSE reduction of the ensemble over the best single base model."""
    if not np.isfinite(rmse_best_base) or rmse_best_base <= 0:
        return {"abs_rmse_gain": float("nan"), "rel_rmse_gain_pct": float("nan")}
    return {
        "abs_rmse_gain": float(rmse_best_base - rmse_ensemble),
        "rel_rmse_gain_pct": float(
            (rmse_best_base - rmse_ensemble) / rmse_best_base * 100.0
        ),
    }
