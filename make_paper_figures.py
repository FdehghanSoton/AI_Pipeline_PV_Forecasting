"""Create figures from V4 output files.

Run ``python analyze_pv_v4.py`` first.
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
CORR_KFOLD = OUT_DIR / "pv_v4_residual_corr_kfold.csv"
CORR_TEMPORAL = OUT_DIR / "pv_v4_residual_corr_temporal.csv"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

MODEL_INFO = {
    key: (model_labels.short_label(key), model_labels.is_ensemble(key))
    for key in (*model_labels.BASE_LEARNERS, *model_labels.ENSEMBLES)
}


def _require(paths: list[Path]) -> None:
    missing = [str(path.name) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing experiment outputs: "
            + ", ".join(missing)
            + ". Run `python analyze_pv_v4.py` first."
        )


def figure_model_comparison() -> None:
    _require([METRICS])
    df = pd.read_csv(METRICS)
    df = df[df["model"].isin(MODEL_INFO)].copy()

    order_source = (
        df[(df["mode"] == "KFOLD") & (df["subset"] == "daylight")]
        .set_index("model")["R2"]
        .sort_values()
    )
    models = list(order_source.index)
    labels = [MODEL_INFO[model][0] for model in models]
    ensemble_rows = np.array([MODEL_INFO[model][1] for model in models])

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.2), sharey=True)
    panels = [("KFOLD", "Random day-fold"), ("TEMPORAL", "Rolling-origin")]
    y_pos = np.arange(len(models))
    bar_height = 0.38

    for ax, (mode, title) in zip(axes, panels, strict=False):
        values = df[df["mode"] == mode].set_index(["model", "subset"])["R2"]
        r2_all = [values.loc[(model, "ALL")] for model in models]
        r2_day = [values.loc[(model, "daylight")] for model in models]

        for row, is_ensemble in zip(y_pos, ensemble_rows, strict=False):
            if is_ensemble:
                ax.axhspan(row - 0.5, row + 0.5, color="0.94", zorder=0)

        ax.barh(y_pos + bar_height / 2, r2_all, height=bar_height, label="All hours")
        ax.barh(
            y_pos - bar_height / 2,
            r2_day,
            height=bar_height,
            label="Daylight",
        )
        ax.set_xlim(0, 1.0)
        ax.set_xlabel(r"$R^2$")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels)
    for tick, is_ensemble in zip(
        axes[0].get_yticklabels(), ensemble_rows, strict=False
    ):
        if is_ensemble:
            tick.set_fontweight("bold")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(
            OUT_DIR / f"pv_v4_fig_model_comparison.{extension}", bbox_inches="tight"
        )
    plt.close(fig)


def figure_residual_corr() -> None:
    _require([CORR_KFOLD, CORR_TEMPORAL])
    matrices = [
        ("Random day-fold", pd.read_csv(CORR_KFOLD, index_col=0)),
        ("Rolling-origin", pd.read_csv(CORR_TEMPORAL, index_col=0)),
    ]
    labels = ["Ridge", "GBM", "POA-norm.\nGBM", "Per-hour\nGBM", "CNN"]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.05), layout="constrained")
    image = None
    for panel_index, (ax, (title, matrix)) in enumerate(
        zip(axes, matrices, strict=False)
    ):
        values = matrix.to_numpy(dtype=float)
        image = ax.imshow(values, vmin=0.6, vmax=1.0, cmap="viridis")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_yticklabels(labels if panel_index == 0 else [], fontsize=7.5)
        ax.tick_params(length=0)
        ax.set_title(title)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if value < 0.75 or value > 0.9 else "black",
                )

    assert image is not None
    colorbar = fig.colorbar(image, ax=list(axes), shrink=0.82, pad=0.025)
    colorbar.set_label("Daylight residual correlation")
    for extension in ("pdf", "png"):
        fig.savefig(
            OUT_DIR / f"pv_v4_fig_residual_corr.{extension}", bbox_inches="tight"
        )
    plt.close(fig)


def _representative_week(df: pd.DataFrame, model: str) -> pd.DataFrame:
    valid = df[df["is_missing"] == 0].copy()
    valid["timestamp"] = pd.to_datetime(valid["timestamp"], utc=True)
    valid = valid.set_index("timestamp").sort_index()
    valid["week"] = valid.index.to_period("W-SUN").astype(str)

    candidates: list[tuple[str, float, int]] = []
    for week, group in valid.groupby("week"):
        daylight = group[group["is_daylight"] == 1]
        if len(group) < 7 * 20 or len(daylight) < 25:
            continue
        rmse = float(np.sqrt(np.mean((daylight["y_actual"] - daylight[model]) ** 2)))
        candidates.append((week, rmse, len(group)))
    if not candidates:
        raise RuntimeError("No complete representative week could be selected.")

    candidate_df = pd.DataFrame(candidates, columns=["week", "rmse", "n"])
    median_rmse = candidate_df["rmse"].median()
    selected = candidate_df.iloc[(candidate_df["rmse"] - median_rmse).abs().argmin()][
        "week"
    ]
    return valid[valid["week"] == selected]


def figure_representative_week() -> None:
    _require([PREDICTIONS])
    predictions = pd.read_csv(PREDICTIONS)
    configurations = [("KFOLD", "NNLSStack"), ("TEMPORAL", "RidgeStack")]

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.2), sharex=False)
    for ax, (mode, model) in zip(axes, configurations, strict=False):
        subset = predictions[predictions["mode"] == mode]
        week = _representative_week(subset, model)
        ax.plot(week.index, week["y_actual"], linewidth=1.4, label="Observed")
        ax.plot(week.index, week[model], linewidth=1.2, label="Forecast")
        ax.set_ylabel("PV power (W)")
        ax.set_title(
            f"{model_labels.mode_label(mode)}: {model_labels.label(model)}"
        )
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=2, loc="upper right")
    axes[-1].set_xlabel("UTC timestamp")
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(
            OUT_DIR / f"pv_v4_fig_representative_week.{extension}", bbox_inches="tight"
        )
    plt.close(fig)


def main() -> None:
    figure_model_comparison()
    figure_residual_corr()
    figure_representative_week()
    print(
        "Wrote model-comparison, residual-correlation and representative-week "
        "figures in PDF and PNG formats."
    )


if __name__ == "__main__":
    main()
