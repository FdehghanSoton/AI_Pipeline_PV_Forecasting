"""Multi-seed stability run for the headline metrics.

The ensemble gains in this study are small, so a single seed is not enough to
claim an improvement. This driver repeats the full backtest for several seeds
and reports the mean and standard deviation of each metric across seeds. The
fold-construction seed and the CNN seed are both varied together so that the
reported spread reflects both data-split and optimisation randomness.

Usage::

    python run_multiseed.py                      # KFOLD, seeds 0..4
    python run_multiseed.py --mode TEMPORAL
    python run_multiseed.py --mode BOTH          # both protocols, one output
    python run_multiseed.py --seeds 0 1 2 3 4 5 6

Outputs (suffixed by the active PV_RUN_TAG, if any):

    pv_v4_multiseed_raw.csv         one metrics block per seed
    pv_v4_multiseed_summary.csv     mean and sd of each metric across seeds

Note: each seed retrains every base learner, including the CNN per fold, so a
multi-seed run is several times slower than a single ``analyze_pv_v4.py`` run.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from analyze_pv_v4 import (
    aggregate,
    build_dataset,
    kfold_backtest,
    temporal_backtest,
)
from baselines import add_skill_columns
from config import load_config


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_one_seed(pv, feats, capacity, mode: str, seed: int, cfg) -> pd.DataFrame:
    """Run one protocol for a single seed and return its pooled metrics."""
    from dataclasses import replace

    seed_cfg = replace(cfg, seed=seed, cnn_seeds=(seed,))
    if mode == "KFOLD":
        folds = kfold_backtest(
            pv, pv, capacity, feats, n_folds=cfg.kfold_n_folds, seed=seed, cfg=seed_cfg
        )
    else:
        folds = temporal_backtest(
            pv,
            pv,
            capacity,
            feats,
            n_folds=cfg.temporal_n_folds,
            first_test_days=cfg.first_test_days,
            cfg=seed_cfg,
        )
    metrics = aggregate(folds, capacity, mode)
    metrics["seed"] = seed
    return metrics


def summarise(raw: pd.DataFrame) -> pd.DataFrame:
    """Mean and sd of each metric across seeds, per (mode, model, subset)."""
    value_cols = ["R2", "MAE", "RMSE", "nMAE_pct", "nRMSE_pct"]
    grouped = raw.groupby(["mode", "model", "subset"])[value_cols]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).add_suffix("_sd")
    out = pd.concat([mean, std], axis=1).reset_index()
    out["n_seeds"] = raw.groupby(["mode", "model", "subset"])["seed"].nunique().to_numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["KFOLD", "TEMPORAL", "BOTH"], default="KFOLD"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4]
    )
    args = parser.parse_args()

    cfg = load_config()
    pv, feats, capacity = build_dataset(cfg)

    modes = ["KFOLD", "TEMPORAL"] if args.mode == "BOTH" else [args.mode]
    blocks = []
    for mode in modes:
        for seed in args.seeds:
            _log(f"=== seed {seed} ({mode}) ===")
            blocks.append(run_one_seed(pv, feats, capacity, mode, seed, cfg))
    raw = pd.concat(blocks, ignore_index=True)
    raw.to_csv(cfg.tagged("pv_v4_multiseed_raw.csv"), index=False)

    summary = summarise(raw)
    if cfg.include_baselines:
        # Skill is computed on the across-seed mean RMSE for readability.
        mean_metrics = summary.rename(columns={"RMSE_mean": "RMSE"}).copy()
        mean_metrics = add_skill_columns(mean_metrics, reference="SmartPersistence")
        summary["skill_vs_SmartPersistence"] = mean_metrics[
            "skill_vs_SmartPersistence"
        ].to_numpy()
    summary.to_csv(cfg.tagged("pv_v4_multiseed_summary.csv"), index=False)

    print("\n=== Multi-seed summary (daylight subset) ===")
    show = summary[summary["subset"] == "daylight"].copy()
    show = show.sort_values("nRMSE_pct_mean")
    cols = ["mode", "model", "R2_mean", "R2_sd", "nRMSE_pct_mean", "nRMSE_pct_sd"]
    with np.printoptions(precision=3):
        print(show[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
