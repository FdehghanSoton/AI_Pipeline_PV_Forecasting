"""Statistical comparison of forecasts.

The ensemble gains over the best single base learner in this study are small,
so they need a significance test rather than a bare ranking. This module
implements the Diebold-Mariano test for equal predictive accuracy and a small
helper to summarise the gain of one model over another.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
