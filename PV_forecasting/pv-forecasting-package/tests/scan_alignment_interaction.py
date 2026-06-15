"""Test the alignment x temporal-context interaction.

Hypothesis: the learned ensemble is insensitive to the -1 h timestamp alignment
only because the lag/lead/rolling temporal-context features already expose the
neighbouring-hour weather, letting a flexible model compensate for a one-hour
offset. If so, removing the alignment should hurt substantially *only when the
temporal-context features are also removed*.

Runs four configurations under both protocols and prints best-ensemble daylight
skill, plus the marginal effect of alignment with and without temporal context.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from run_ablations import run_config
from config import load_config


def main() -> None:
    base = load_config()
    configs = {
        "full (align + temporal)": base,
        "no_temporal (align only)": replace(base, use_temporal_context=False),
        "no_align (temporal only)": replace(base, time_shift_hours=0),
        "no_align + no_temporal": replace(
            base, time_shift_hours=0, use_temporal_context=False
        ),
    }
    rows = []
    for label, cfg in configs.items():
        rows += run_config(label, cfg, ["KFOLD", "TEMPORAL"])
    df = pd.DataFrame(rows)
    day = df[df["subset"] == "daylight"].copy()

    print("\n=== Best-ensemble daylight skill ===")
    show = day[["ablation", "mode", "ensemble_R2", "ensemble_nRMSE_pct"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n=== Marginal effect of REMOVING the -1h alignment (daylight nRMSE) ===")
    for mode in ("KFOLD", "TEMPORAL"):
        d = day[day["mode"] == mode].set_index("ablation")["ensemble_nRMSE_pct"]
        with_tc = d["no_align (temporal only)"] - d["full (align + temporal)"]
        without_tc = d["no_align + no_temporal"] - d["no_temporal (align only)"]
        print(
            f"{mode:9s}  with temporal context: {with_tc:+.2f} pp   "
            f"without temporal context: {without_tc:+.2f} pp"
        )


if __name__ == "__main__":
    main()
