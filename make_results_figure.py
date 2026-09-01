"""Daylight scatter and a representative held-out week.

Run ``python analyze_pv_v4.py`` first, then ``python make_results_figure.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import model_labels
import paths

OUT_DIR = paths.results_dir()
METRICS = OUT_DIR / "pv_v4_metrics.csv"
PREDICTIONS = OUT_DIR / "pv_v4_predictions.csv"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Best daylight model per protocol (matches the representative-week figure).
MODE = "KFOLD"
MODEL = "NNLSStack"
MODEL_LABEL = model_labels.label(MODEL)


def _require(paths: list[Path]) -> None:
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing experiment outputs: "
            + ", ".join(missing)
            + ". Run `python analyze_pv_v4.py` first."
        )


def _metric(metrics: pd.DataFrame, subset: str, field: str) -> float:
    row = metrics[
        (metrics["mode"] == MODE)
        & (metrics["model"] == MODEL)
        & (metrics["subset"] == subset)
    ]
    return float(row[field].iloc[0])


def _representative_week(df: pd.DataFrame) -> pd.DataFrame:
    """Select the week whose daylight RMSE is closest to the median week."""
    valid = df[df["is_missing"] == 0].copy()
    valid["week"] = valid.index.to_period("W-SUN").astype(str)
    candidates = []
    for week, group in valid.groupby("week"):
        daylight = group[group["is_daylight"] == 1]
        if len(group) < 7 * 20 or len(daylight) < 25:
            continue
        rmse = float(np.sqrt(np.mean((daylight["y_actual"] - daylight[MODEL]) ** 2)))
        candidates.append((week, rmse))
    if not candidates:
        raise RuntimeError("No complete representative week could be selected.")
    cand = pd.DataFrame(candidates, columns=["week", "rmse"])
    chosen = cand.iloc[(cand["rmse"] - cand["rmse"].median()).abs().argmin()]["week"]
    return chosen


def main() -> None:
    _require([METRICS, PREDICTIONS])
    metrics = pd.read_csv(METRICS)
    predictions = pd.read_csv(PREDICTIONS)
    predictions = predictions[predictions["mode"] == MODE].copy()
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], utc=True)
    predictions = predictions.set_index("timestamp").sort_index()

    r2_day = _metric(metrics, "daylight", "R2")
    nrmse_day = _metric(metrics, "daylight", "nRMSE_pct")

    fig, (ax_scatter, ax_week) = plt.subplots(1, 2, figsize=(6.5, 3.2))

    # Panel A: daylight scatter.
    day = predictions[(predictions["is_missing"] == 0) & (predictions["is_daylight"] == 1)]
    lim = float(max(day["y_actual"].max(), day[MODEL].max()))
    ax_scatter.hexbin(day[MODEL], day["y_actual"], gridsize=40, cmap="viridis", mincnt=1)
    ax_scatter.plot([0, lim], [0, lim], "r--", linewidth=1)
    ax_scatter.set_xlabel("Forecast PV power (W)")
    ax_scatter.set_ylabel("Observed PV power (W)")
    ax_scatter.set_title(f"Daylight forecast ({MODEL_LABEL})")
    ax_scatter.text(
        0.05,
        0.95,
        f"$R^2$ = {r2_day:.3f}\nnRMSE = {nrmse_day:.2f}%",
        transform=ax_scatter.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.7"),
    )

    # Panel B: representative week with missing shown as gaps.
    chosen_week = _representative_week(predictions)
    week = predictions[predictions.index.to_period("W-SUN").astype(str) == chosen_week]
    full_index = pd.date_range(week.index.min(), week.index.max(), freq="h", tz="UTC")
    actual = week["y_actual"].where(week["is_missing"] == 0).reindex(full_index)
    forecast = week[MODEL].reindex(full_index)
    ax_week.plot(full_index, actual, linewidth=1.4, label="Observed")
    ax_week.plot(full_index, forecast, linewidth=1.1, label="Forecast")
    ax_week.set_ylabel("PV power (W)")
    ax_week.set_xlabel("UTC timestamp")
    ax_week.set_title("Representative held-out week")
    ax_week.legend(ncol=2, loc="upper right")
    ax_week.grid(alpha=0.2)
    for label in ax_week.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")

    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"pv_v4_fig_results.{extension}", bbox_inches="tight")
    plt.close(fig)
    print("Wrote pv_v4_fig_results.pdf and .png")


if __name__ == "__main__":
    main()
