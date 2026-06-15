"""Pipeline ablation study.

This driver substantiates the paper's central, pipeline-centred claim by
turning each ingredient off in turn and measuring the effect on forecast
skill. It runs the following configurations and writes a single comparison
table:

* ``full``              the complete pipeline (default policies)
* ``no_alignment``      no -1 h PV/weather timestamp alignment
* ``calendar_only``     solar-geometry and clearness features removed
* ``no_temporal``       lag/lead/rolling weather context removed

Each configuration is a full backtest including the CNN, so the study is
expensive. By default only the KFOLD protocol and a single CNN seed are used
to keep runtime manageable; pass ``--mode TEMPORAL`` or ``--both`` to extend.

Usage::

    python run_ablations.py
    python run_ablations.py --both

Output: ``pv_v4_ablation.csv`` (suffixed by PV_RUN_TAG if set).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import pandas as pd

from analyze_pv_v4 import (
    BASE_LEARNERS,
    ENSEMBLE_NAMES,
    aggregate,
    build_dataset,
    kfold_backtest,
    temporal_backtest,
)
from config import RunConfig, load_config


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ablation_configs(base: RunConfig) -> dict[str, RunConfig]:
    """Return the named ablation configurations derived from ``base``."""
    return {
        "full": base,
        "no_alignment": replace(base, time_shift_hours=0),
        "calendar_only": replace(base, use_physics_features=False),
        "no_temporal": replace(base, use_temporal_context=False),
    }


def _best_rows(metrics: pd.DataFrame, mode: str, label: str) -> list[dict]:
    """Extract best-ensemble and best-base daylight/all rows for one ablation."""
    rows = []
    for subset in ("ALL", "daylight"):
        met = metrics[(metrics["mode"] == mode) & (metrics["subset"] == subset)]
        ens = met[met["model"].isin(ENSEMBLE_NAMES)].sort_values("nRMSE_pct")
        base = met[met["model"].isin(BASE_LEARNERS)].sort_values("nRMSE_pct")
        if ens.empty or base.empty:
            continue
        best_ens = ens.iloc[0]
        best_base = base.iloc[0]
        rows.append(
            {
                "ablation": label,
                "mode": mode,
                "subset": subset,
                "best_ensemble": best_ens["model"],
                "ensemble_R2": float(best_ens["R2"]),
                "ensemble_nRMSE_pct": float(best_ens["nRMSE_pct"]),
                "best_base": best_base["model"],
                "base_R2": float(best_base["R2"]),
                "base_nRMSE_pct": float(best_base["nRMSE_pct"]),
            }
        )
    return rows


def run_config(label: str, cfg: RunConfig, modes: list[str]) -> list[dict]:
    pv, feats, capacity = build_dataset(cfg)
    rows: list[dict] = []
    if "KFOLD" in modes:
        folds = kfold_backtest(
            pv, pv, capacity, feats, n_folds=cfg.kfold_n_folds, seed=cfg.seed, cfg=cfg
        )
        rows += _best_rows(aggregate(folds, capacity, "KFOLD"), "KFOLD", label)
    if "TEMPORAL" in modes:
        folds = temporal_backtest(
            pv,
            pv,
            capacity,
            feats,
            n_folds=cfg.temporal_n_folds,
            first_test_days=cfg.first_test_days,
            cfg=cfg,
        )
        rows += _best_rows(aggregate(folds, capacity, "TEMPORAL"), "TEMPORAL", label)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["KFOLD", "TEMPORAL"], default="KFOLD")
    parser.add_argument(
        "--both", action="store_true", help="Run both protocols (slower)."
    )
    args = parser.parse_args()
    modes = ["KFOLD", "TEMPORAL"] if args.both else [args.mode]

    base = load_config()
    rows: list[dict] = []
    for label, cfg in ablation_configs(base).items():
        _log(f"=== ablation: {label}  modes={modes} ===")
        rows += run_config(label, cfg, modes)

    table = pd.DataFrame(rows)
    out = base.tagged("pv_v4_ablation.csv")
    table.to_csv(out, index=False)
    print("\n=== Ablation table ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved {out.name}")


if __name__ == "__main__":
    main()
