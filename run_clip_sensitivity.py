"""Sensitivity of the results to the clearness-ratio upper bound.

First, the bound on the clearness-index *feature* never binds: the largest
value observed in the record is 1.0045, so that bound cannot have influenced
any reported number (see ``audit_clipping.py``).

Second, the bound on the clearness-index GBM's *target* does bind, on about a
tenth of daylight hours. Rather than defend 1.5 as a principled constant, which
it is not, this script sweeps it and reports how much the results move. If the
results are stable across a wide range of bounds, the specific value is not
load-bearing.

The sweep reports the clearness-index GBM itself, which the bound acts on
directly, alongside the best ensemble, which is what the paper's headline
numbers come from.

Run ``python run_clip_sensitivity.py``. Output goes to
``pv_v4_clip_sensitivity.csv``.
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pandas as pd

import analyze_pv_v4 as pipeline
import paths
from baselines import add_skill_columns
from config import RunConfig, load_config

OUT_DIR = paths.results_dir()

# 1.0 forbids any enhancement above the clear-sky envelope, which is physically
# wrong but a useful lower anchor. Infinity leaves the target unbounded.
BOUNDS: tuple[float, ...] = (1.0, 1.25, 1.5, 2.0, 3.0, float("inf"))

MODE_LABEL = {"KFOLD": "Random day-fold", "TEMPORAL": "Rolling-origin"}


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def run_bound(bound: float, base: RunConfig) -> pd.DataFrame:
    """Both backtests at one value of the clearness-ratio bound."""
    cfg = replace(base, kappa_clip=bound, run_tag=f"clip_{bound}")
    pipeline.set_global_seed(cfg.seed)
    pv, feats, capacity = pipeline.build_dataset(cfg)

    temporal = pipeline.temporal_backtest(
        pv,
        pv,
        capacity,
        feats,
        n_folds=cfg.temporal_n_folds,
        first_test_days=cfg.first_test_days,
        cfg=cfg,
    )
    kfold = pipeline.kfold_backtest(
        pv, pv, capacity, feats, n_folds=cfg.kfold_n_folds, seed=cfg.seed, cfg=cfg
    )
    metrics = pd.concat(
        [
            pipeline.aggregate(temporal, capacity, "TEMPORAL"),
            pipeline.aggregate(kfold, capacity, "KFOLD"),
        ],
        ignore_index=True,
    )
    if cfg.include_baselines:
        metrics = add_skill_columns(metrics, reference="SmartPersistence")
    metrics.insert(0, "kappa_clip", bound)
    return metrics


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    """POA-normalised GBM and the pre-declared stack, per bound and protocol.

    The ensemble column follows ``NNLSStack`` at every bound rather than
    whichever fusion method scored best at that bound, so the sweep compares
    one method across bounds instead of a changing set of methods.
    """
    rows = []
    daylight = metrics[metrics["subset"] == "daylight"]
    for (bound, mode), group in daylight.groupby(["kappa_clip", "mode"]):
        indexed = group.set_index("model")
        ensembles = group[group["model"].isin(pipeline.ENSEMBLE_NAMES)]
        best = ensembles.sort_values("RMSE").iloc[0]
        rows.append(
            {
                "kappa_clip": bound,
                "mode": mode,
                "clearness_gbm_R2": float(indexed.loc["GBM_kt", "R2"]),
                "clearness_gbm_nRMSE_pct": float(indexed.loc["GBM_kt", "nRMSE_pct"]),
                "ensemble": "NNLSStack",
                "ensemble_R2": float(indexed.loc["NNLSStack", "R2"]),
                "ensemble_nRMSE_pct": float(indexed.loc["NNLSStack", "nRMSE_pct"]),
                "lowest_error_ensemble": best["model"],
                "lowest_error_nRMSE_pct": float(best["nRMSE_pct"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["mode", "kappa_clip"])


def main() -> None:
    base = load_config()
    collected = []
    for bound in BOUNDS:
        _log(f"=== clearness-ratio bound = {bound} ===")
        collected.append(run_bound(bound, base))

    metrics = pd.concat(collected, ignore_index=True)
    summary = summarise(metrics)

    # Spread relative to the value the paper uses, so the text can state how
    # little the choice of 1.5 matters.
    for mode, group in summary.groupby("mode"):
        reference = group[group["kappa_clip"] == 1.5]
        if reference.empty:
            continue
        baseline_nrmse = float(reference["ensemble_nRMSE_pct"].iloc[0])
        summary.loc[summary["mode"] == mode, "ensemble_nRMSE_change_pp"] = (
            summary.loc[summary["mode"] == mode, "ensemble_nRMSE_pct"]
            - baseline_nrmse
        )

    summary.to_csv(OUT_DIR / "pv_v4_clip_sensitivity.csv", index=False)

    print("\n=== Sensitivity to the clearness-ratio upper bound (daylight) ===")
    for mode in ("KFOLD", "TEMPORAL"):
        view = summary[summary["mode"] == mode]
        if view.empty:
            continue
        print(f"\n--- {MODE_LABEL[mode]} ---")
        print(
            view[
                [
                    "kappa_clip",
                    "clearness_gbm_R2",
                    "clearness_gbm_nRMSE_pct",
                    "ensemble_R2",
                    "ensemble_nRMSE_pct",
                    "ensemble_nRMSE_change_pp",
                ]
            ].to_string(index=False, float_format=lambda v: f"{v:.3f}")
        )
        spread = float(
            np.ptp(view["ensemble_nRMSE_pct"].to_numpy())
        )
        print(
            f"  best-ensemble daylight nRMSE spans {spread:.3f} percentage "
            f"points across the whole sweep"
        )

    print("\nWrote pv_v4_clip_sensitivity.csv")


if __name__ == "__main__":
    main()
