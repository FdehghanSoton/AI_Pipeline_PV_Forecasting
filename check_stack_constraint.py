"""Compare three ways of turning validation predictions into stacking weights.

``build_ensembles`` forecasts with the fitted NNLS coefficients unchanged.
Two alternatives would instead produce weights that sum to one, and this
script scores all three on the same folds so the choice can be justified:

``NNLSStack``          the fitted coefficients, used unchanged (the default);
``NNLSRenormalised``   the same coefficients divided by their sum, which is a
                       rescaling of the solution to a different problem;
``NNLSConvex``         the least-squares solution obtained with non-negativity
                       and sum-to-one imposed together during fitting.

Run ``python check_stack_constraint.py``. Writes
``pv_v4_stack_constraint.csv`` and prints the per-fold coefficient sums, which
show how far the fitted solution sits from the simplex.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

import analyze_pv_v4 as m
import paths
from config import load_config

OUT_PATH = paths.results_dir() / "pv_v4_stack_constraint.csv"

_orig_build_ensembles = m.build_ensembles
_weight_sums: dict[str, list[float]] = {"KFOLD": [], "TEMPORAL": []}
_mode = {"tag": "KFOLD"}


def convex_least_squares(V: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares over the simplex: ``w >= 0`` and ``sum(w) == 1``.

    Both constraints are imposed during fitting, so the result is the
    minimiser of the stated objective rather than a rescaled solution of a
    different one.
    """
    n = V.shape[1]
    start = np.full(n, 1.0 / n)

    def objective(w: np.ndarray) -> float:
        residual = y - V @ w
        return float(residual @ residual)

    def gradient(w: np.ndarray) -> np.ndarray:
        return -2.0 * V.T @ (y - V @ w)

    result = minimize(
        objective,
        start,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, None)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        return start
    w = np.clip(result.x, 0.0, None)
    return w / w.sum() if w.sum() > 1e-12 else start


def _wrapped(val_preds, y_val, val_mask, test_preds):
    out = _orig_build_ensembles(val_preds, y_val, val_mask, test_preds)
    names = list(val_preds.keys())
    V = np.stack([val_preds[n] for n in names], axis=1)
    T = np.stack([test_preds[n] for n in names], axis=1)
    y_m, V_m = y_val[val_mask], V[val_mask]

    raw, _ = nnls(V_m, y_m)
    total = float(raw.sum())
    _weight_sums[_mode["tag"]].append(total)
    out["NNLSRenormalised"] = np.clip(T @ (raw / total), 0.0, None)
    out["NNLSConvex"] = np.clip(T @ convex_least_squares(V_m, y_m), 0.0, None)
    return out


def _daylight_rmse(folds, name: str) -> float:
    y = np.concatenate([f.y_test[~f.is_miss_test & f.is_day_test] for f in folds])
    yh = np.concatenate(
        [f.preds[name][~f.is_miss_test & f.is_day_test] for f in folds]
    )
    return float(np.sqrt(np.mean((y - yh) ** 2)))


def main() -> None:
    m.build_ensembles = _wrapped
    cfg = load_config()
    pv, feats, capacity = m.build_dataset(cfg)

    _mode["tag"] = "KFOLD"
    kfold = m.kfold_backtest(
        pv, pv, capacity, feats, n_folds=cfg.kfold_n_folds, seed=cfg.seed, cfg=cfg
    )
    _mode["tag"] = "TEMPORAL"
    temporal = m.temporal_backtest(
        pv,
        pv,
        capacity,
        feats,
        n_folds=cfg.temporal_n_folds,
        first_test_days=cfg.first_test_days,
        cfg=cfg,
    )

    rows = []
    for tag, folds in (("KFOLD", kfold), ("TEMPORAL", temporal)):
        for name in ("NNLSStack", "NNLSRenormalised", "NNLSConvex", "RidgeStack"):
            rmse = _daylight_rmse(folds, name)
            rows.append(
                {
                    "mode": tag,
                    "variant": name,
                    "daylight_RMSE": rmse,
                    "daylight_nRMSE_pct": 100.0 * rmse / capacity,
                    "mean_raw_weight_sum": float(np.mean(_weight_sums[tag])),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(OUT_PATH, index=False)
    print(table.to_string(index=False))
    for tag in ("KFOLD", "TEMPORAL"):
        sums = ", ".join(f"{s:.4f}" for s in _weight_sums[tag])
        print(f"{tag} fitted NNLS coefficient sums per fold: {sums}")


if __name__ == "__main__":
    main()
