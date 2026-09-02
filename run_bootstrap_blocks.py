"""Day-level and multi-day block bootstrap of the ensemble RMSE gain.

The saved out-of-fold forecasts are fixed. Each replicate resamples contiguous
blocks of target days, so the intervals speak to which test days are drawn,
not to retraining. Run after ``analyze_pv_v4.py``.

Writes ``pv_v4_bootstrap_blocks.csv``.
"""

from __future__ import annotations

import pandas as pd

import paths
from config import load_config
from stats_tests import paired_day_bootstrap


def main() -> None:
    cfg = load_config()
    predictions = pd.read_csv(
        cfg.tagged("pv_v4_predictions.csv"), parse_dates=["timestamp"]
    )
    rows = []
    for mode, group in predictions.groupby("mode"):
        scored = group[
            (group["is_missing"] == 0) & (group["is_daylight"] == 1)
        ]
        days = pd.DatetimeIndex(scored["timestamp"]).normalize().to_numpy()
        for width in (1, 3, 7):
            boot = paired_day_bootstrap(
                scored["y_actual"].to_numpy(),
                scored["NNLSStack"].to_numpy(),
                scored["BestSingleByVal"].to_numpy(),
                days,
                block_days=width,
            )
            rows.append(
                {
                    "mode": mode,
                    "block_days": width,
                    **boot.as_dict(),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(paths.results_dir() / "pv_v4_bootstrap_blocks.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
