"""Downstream sensitivity of forecast skill to the PV timestamp alignment.

``scan_shift_by_fold.py`` shows that a sweep restricted to a single fold's
training days does not always separate the -1 h and -2 h candidates. This
driver runs the whole pipeline at each of those candidates and at no
correction, so the choice can be judged by its effect on forecast error rather
than by the alignment instrument alone.

Only the pre-declared NNLS stack and the best individual learner are reported,
so no method is selected on test metrics here.

Usage::

    python run_shift_sensitivity.py

Output: ``pv_v4_shift_sensitivity.csv``.
"""

from __future__ import annotations

import time
from dataclasses import replace

import pandas as pd

from analyze_pv_v4 import (
    BASE_LEARNERS,
    aggregate,
    build_dataset,
    kfold_backtest,
    temporal_backtest,
)
from config import RunConfig, load_config

SHIFTS = (-2, -1, 0)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _rows(metrics: pd.DataFrame, mode: str, shift: int) -> list[dict]:
    out = []
    met = metrics[(metrics["mode"] == mode) & (metrics["subset"] == "daylight")]
    if met.empty:
        return out
    stack = met[met["model"] == "NNLSStack"]
    base = met[met["model"].isin(BASE_LEARNERS)].sort_values("nRMSE_pct")
    out.append(
        {
            "shift_hours": shift,
            "mode": mode,
            "NNLS_nRMSE_pct": float(stack["nRMSE_pct"].iloc[0]),
            "NNLS_R2": float(stack["R2"].iloc[0]),
            "best_base": base["model"].iloc[0],
            "best_base_nRMSE_pct": float(base["nRMSE_pct"].iloc[0]),
        }
    )
    return out


def run_shift(shift: int, base: RunConfig) -> list[dict]:
    cfg = replace(base, time_shift_hours=shift)
    pv, feats, capacity = build_dataset(cfg)
    rows: list[dict] = []
    folds = kfold_backtest(
        pv, pv, capacity, feats, n_folds=cfg.kfold_n_folds, seed=cfg.seed, cfg=cfg
    )
    rows += _rows(aggregate(folds, capacity, "KFOLD"), "KFOLD", shift)
    folds = temporal_backtest(
        pv,
        pv,
        capacity,
        feats,
        n_folds=cfg.temporal_n_folds,
        first_test_days=cfg.first_test_days,
        cfg=cfg,
    )
    rows += _rows(aggregate(folds, capacity, "TEMPORAL"), "TEMPORAL", shift)
    return rows


def main() -> None:
    base = load_config()
    rows: list[dict] = []
    for shift in SHIFTS:
        _log(f"=== timestamp shift {shift:+d} h ===")
        rows += run_shift(shift, base)

    table = pd.DataFrame(rows).sort_values(["mode", "shift_hours"])
    out = base.tagged("pv_v4_shift_sensitivity.csv")
    table.to_csv(out, index=False)
    print("\n=== Timestamp-shift sensitivity (daylight) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved {out.name}")


if __name__ == "__main__":
    main()
