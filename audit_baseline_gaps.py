"""How often the reference forecasts fall back when their inputs are missing.

The two persistence references need the PV value 24 hours before the target
hour, and climatology needs the training mean for the target hour's month and
hour. Neither input is always available: 9.25% of the hourly grid is missing,
and an early rolling-origin fold can be asked about a calendar month it has
never trained on. This script counts how often each fallback is used, so the
paper can state the rule and its frequency instead of leaving it implicit.

It reports both the rate at which the 24-hour lag alone is unavailable and the
rate at which the whole ``LOOKBACK_DAYS`` window is exhausted, since only the
latter reaches the climatology fallback.

Run ``python audit_baseline_gaps.py``. Writes ``pv_v4_baseline_gaps.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import analyze_pv_v4 as m
import paths
from baselines import LOOKBACK_DAYS
from config import load_config


def _test_blocks(pv: pd.DataFrame, cfg) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """(protocol, train rows, test rows) for every fold of both protocols."""
    days = pv.index.normalize().unique()
    n_days = len(days)
    blocks = []

    fold_len = (n_days - cfg.first_test_days) // cfg.temporal_n_folds
    for fold in range(cfg.temporal_n_folds):
        start = days[cfg.first_test_days + fold * fold_len]
        end = (
            days[cfg.first_test_days + (fold + 1) * fold_len]
            if fold < cfg.temporal_n_folds - 1
            else days[-1] + pd.Timedelta(days=1)
        )
        train = pv[(pv.index < start) & (pv["is_missing"] == 0)]
        blocks.append(("TEMPORAL", train, pv[(pv.index >= start) & (pv.index < end)]))

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n_days)
    fold_size = n_days // cfg.kfold_n_folds
    for k in range(cfg.kfold_n_folds):
        hi = (k + 1) * fold_size if k < cfg.kfold_n_folds - 1 else n_days
        test_days = days[perm[k * fold_size : hi]]
        is_test = pv.index.normalize().isin(test_days)
        blocks.append(("KFOLD", pv[~is_test & (pv["is_missing"] == 0)], pv[is_test]))
    return blocks


def main() -> None:
    cfg = load_config()
    pv, _, _ = m.build_dataset(cfg)
    observed = pv["is_missing"].to_numpy() == 0

    rows = []
    for protocol, train, test in _test_blocks(pv, cfg):
        scored = test[(test["is_missing"] == 0) & (test["is_daylight"] == 1)]

        lag_index = scored.index - pd.Timedelta(hours=24)
        lag_missing = int((~lag_index.isin(pv.index[observed])).sum())

        # With the lookback rule, a hole only reaches the climatology fallback
        # when the same hour is unobserved on every one of the preceding days.
        deep_gap = np.ones(len(scored), dtype=bool)
        for day in range(1, LOOKBACK_DAYS + 1):
            back = scored.index - pd.Timedelta(days=day)
            deep_gap &= ~back.isin(pv.index[observed])
        lookback_exhausted = int(deep_gap.sum())

        seen = set(zip(train.index.month, train.index.hour, strict=False))
        keys = list(zip(scored.index.month, scored.index.hour, strict=False))
        unseen = sum(1 for key in keys if key not in seen)
        seen_hours = set(train.index.hour)
        unseen_hour = sum(1 for _, h in keys if h not in seen_hours)

        rows.append(
            {
                "protocol": protocol,
                "scored_daylight_hours": len(scored),
                "lag24_missing": lag_missing,
                "lag24_missing_pct": 100.0 * lag_missing / max(len(scored), 1),
                "lookback_exhausted": lookback_exhausted,
                "lookback_exhausted_pct": (
                    100.0 * lookback_exhausted / max(len(scored), 1)
                ),
                "unseen_month_hour": unseen,
                "unseen_month_hour_pct": 100.0 * unseen / max(len(scored), 1),
                "unseen_hour": unseen_hour,
                "unseen_hour_pct": 100.0 * unseen_hour / max(len(scored), 1),
            }
        )

    per_fold = pd.DataFrame(rows)
    counts = [
        "lag24_missing",
        "lookback_exhausted",
        "unseen_month_hour",
        "unseen_hour",
    ]
    totals = (
        per_fold.groupby("protocol")[["scored_daylight_hours", *counts]]
        .sum()
        .reset_index()
    )
    for column in counts:
        totals[column + "_pct"] = (
            100.0 * totals[column] / totals["scored_daylight_hours"]
        )
    totals.to_csv(paths.results_dir() / "pv_v4_baseline_gaps.csv", index=False)

    print("\n=== Per fold ===")
    print(per_fold.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n=== Pooled over folds ===")
    print(totals.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nWrote pv_v4_baseline_gaps.csv")


if __name__ == "__main__":
    main()
