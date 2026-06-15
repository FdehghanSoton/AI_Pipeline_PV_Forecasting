"""Full-pipeline regression guard.

Two layers:

1. An always-on check that the committed reference metrics
   (``pv_v4_metrics.csv``) still contain the documented headline values within
   the configured tolerance. This catches accidental corruption of the
   committed results.
2. An opt-in check (set ``PV_RUN_REGRESSION=1``) that reruns the pipeline with
   the legacy policies and compares the regenerated headline metrics against the
   committed reference within the same tolerance. This is skipped by default
   because it retrains every model, including the CNN.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from config import REGRESSION_ABS_TOL_NRMSE_PCT, REGRESSION_ABS_TOL_R2

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "pv_v4_metrics.csv"

# Documented committed headline values (CHANGELOG.md baseline block).
EXPECTED = {
    ("KFOLD", "NNLSStack", "daylight"): {"R2": 0.736, "nRMSE_pct": 13.14},
    ("KFOLD", "NNLSStack", "ALL"): {"R2": 0.858, "nRMSE_pct": 8.42},
    ("TEMPORAL", "RidgeStack", "daylight"): {"R2": 0.570, "nRMSE_pct": 16.88},
    ("TEMPORAL", "RidgeStack", "ALL"): {"R2": 0.754, "nRMSE_pct": 10.65},
}


def _lookup(df: pd.DataFrame, mode: str, model: str, subset: str) -> pd.Series:
    row = df[
        (df["mode"] == mode) & (df["model"] == model) & (df["subset"] == subset)
    ]
    assert len(row) == 1, f"expected one row for {mode}/{model}/{subset}"
    return row.iloc[0]


def test_committed_reference_matches_documented_headline() -> None:
    if not METRICS.exists():
        pytest.skip("pv_v4_metrics.csv not present")
    df = pd.read_csv(METRICS)
    for (mode, model, subset), want in EXPECTED.items():
        row = _lookup(df, mode, model, subset)
        assert abs(row["R2"] - want["R2"]) <= REGRESSION_ABS_TOL_R2
        assert abs(row["nRMSE_pct"] - want["nRMSE_pct"]) <= REGRESSION_ABS_TOL_NRMSE_PCT


@pytest.mark.skipif(
    os.environ.get("PV_RUN_REGRESSION") != "1",
    reason="set PV_RUN_REGRESSION=1 to run the full-pipeline regression (slow)",
)
def test_full_pipeline_reproduces_reference(monkeypatch, tmp_path) -> None:
    # Reproduce the committed numbers using the legacy policies.
    monkeypatch.setenv("PV_CAPACITY_POLICY", "global")
    monkeypatch.setenv("PV_DAYLIGHT_POLICY", "pv_median")
    monkeypatch.setenv("PV_INCLUDE_BASELINES", "0")

    import analyze_pv_v4 as v4

    cfg = v4.load_config()
    pv, feats, capacity = v4.build_dataset(cfg)
    t_folds = v4.temporal_backtest(
        pv, pv, capacity, feats, n_folds=cfg.temporal_n_folds,
        first_test_days=cfg.first_test_days, cfg=cfg,
    )
    k_folds = v4.kfold_backtest(
        pv, pv, capacity, feats, n_folds=cfg.kfold_n_folds, seed=cfg.seed, cfg=cfg
    )
    metrics = pd.concat(
        [v4.aggregate(t_folds, capacity, "TEMPORAL"),
         v4.aggregate(k_folds, capacity, "KFOLD")],
        ignore_index=True,
    )
    for (mode, model, subset), want in EXPECTED.items():
        row = _lookup(metrics, mode, model, subset)
        assert abs(row["R2"] - want["R2"]) <= REGRESSION_ABS_TOL_R2
        assert abs(row["nRMSE_pct"] - want["nRMSE_pct"]) <= REGRESSION_ABS_TOL_NRMSE_PCT
