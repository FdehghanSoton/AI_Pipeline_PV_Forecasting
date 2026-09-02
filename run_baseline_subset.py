"""Does the skill over smart persistence depend on hours with a missing lag?

The persistence references need the measured PV value one day before the target
hour. Where that hour was never recorded the reference has to fall back, and a
reference that is handicapped on those hours would flatter every model scored
against it. Squared error is dominated by its largest terms, so a small number
of such hours could in principle carry the reported skill.

This script rescores the saved forecasts on the subset of held-out hours whose
24-hour lag was actually observed, so no reference forecast on that subset used
a fallback at all. If the skill over smart persistence survives, the headline
claim does not rest on the fallback rule.

Run ``python run_baseline_subset.py`` after ``analyze_pv_v4.py``. Writes
``pv_v4_baseline_subset.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import analyze_pv_v4 as m
import paths
from baselines import skill_score
from config import load_config

MODELS = ("NNLSStack", "BestSingleByVal", "SmartPersistence", "Persistence")


def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def main() -> None:
    cfg = load_config()
    predictions = pd.read_csv(
        cfg.tagged("pv_v4_predictions.csv"), parse_dates=["timestamp"]
    )
    pv, _, capacity = m.build_dataset(cfg)
    observed = pv.index[pv["is_missing"].to_numpy() == 0]

    scored = predictions[
        (predictions["is_missing"] == 0) & (predictions["is_daylight"] == 1)
    ].copy()
    lag = pd.DatetimeIndex(scored["timestamp"]) - pd.Timedelta(hours=24)
    scored["lag_observed"] = lag.isin(observed)

    rows = []
    for mode, group in scored.groupby("mode"):
        for label, block in (
            ("all scored hours", group),
            ("lag-24 observed only", group[group["lag_observed"]]),
        ):
            y = block["y_actual"].to_numpy()
            rmse = {name: _rmse(y, block[name].to_numpy()) for name in MODELS}
            reference = rmse["SmartPersistence"]
            rows.append(
                {
                    "mode": mode,
                    "subset": label,
                    "n": len(block),
                    "pct_of_scored": 100.0 * len(block) / max(len(group), 1),
                    **{f"rmse_{name}": rmse[name] for name in MODELS},
                    "nRMSE_NNLS_pct": 100.0 * rmse["NNLSStack"] / capacity,
                    "skill_NNLS": skill_score(rmse["NNLSStack"], reference),
                    "skill_best_single": skill_score(
                        rmse["BestSingleByVal"], reference
                    ),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(paths.results_dir() / "pv_v4_baseline_subset.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nWrote pv_v4_baseline_subset.csv")


if __name__ == "__main__":
    main()
