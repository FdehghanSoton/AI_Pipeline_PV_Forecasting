"""Appendix figures from saved pipeline outputs.

Skips a panel when its input CSV is absent. Run the pipeline first, then
``python make_appendix_figures.py``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import model_labels
import paths

OUT_DIR = paths.results_dir()

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

BASE_LEARNERS = list(model_labels.BASE_LEARNERS)
ENSEMBLES = list(model_labels.ENSEMBLES)
BASELINES = ["Persistence", "Climatology", "SmartPersistence"]
MODE_LABEL = model_labels.MODE_LABEL
ABLATION_LABEL = {
    "full": "Full pipeline",
    "no_alignment": "No $-1$h alignment",
    "calendar_only": "Calendar-only features",
    "no_temporal": "No temporal context",
}


def _save(fig: plt.Figure, stem: str) -> None:
    for extension in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{extension}", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {stem}.pdf and .png")


def figure_monthly_skill() -> None:
    path = OUT_DIR / "pv_v4_per_month.csv"
    if not path.exists():
        print("skip monthly skill: pv_v4_per_month.csv not found")
        return
    df = pd.read_csv(path)
    daylight = df[df["subset"] == "daylight"]
    fig, axes = plt.subplots(
        1, df["mode"].nunique(), figsize=(6.5, 3.2), squeeze=False
    )
    for ax, (mode, group) in zip(axes[0], daylight.groupby("mode"), strict=False):
        pivot = group.pivot_table(index="month", values="R2")
        ax.imshow(
            pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0
        )
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks([])
        ax.set_title(f"{MODE_LABEL.get(mode, mode)}\ndaylight $R^2$")
        for row, value in enumerate(pivot["R2"].to_numpy()):
            ax.text(0, row, f"{value:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if value < 0.6 else "black")
    fig.tight_layout()
    _save(fig, "pv_v4_fig_app_monthly_skill")


def figure_missingness() -> None:
    path = OUT_DIR / "data_audit_missing_runs.csv"
    if not path.exists():
        print("skip missingness: data_audit_missing_runs.csv not found (run audit_data.py)")
        return
    runs = pd.read_csv(path, parse_dates=["start", "end"])
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    for _, run in runs.iterrows():
        ax.axvspan(run["start"], run["end"], color="firebrick", alpha=0.6)
    ax.set_yticks([])
    ax.set_xlabel("UTC date")
    ax.set_title(
        f"Missing-PV intervals ({len(runs)} runs, "
        f"longest {int(runs['hours'].max())} h)"
    )
    fig.tight_layout()
    _save(fig, "pv_v4_fig_app_missingness")


def _load_predictions() -> pd.DataFrame | None:
    path = OUT_DIR / "pv_v4_predictions.csv"
    if not path.exists():
        print("skip: pv_v4_predictions.csv not found")
        return None
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[df["is_missing"] == 0].copy()


def figure_error_by_hour(model: str = "NNLSStack") -> None:
    df = _load_predictions()
    if df is None:
        return
    fig, axes = plt.subplots(1, df["mode"].nunique(), figsize=(6.5, 3.0), squeeze=False)
    for ax, (mode, group) in zip(axes[0], df.groupby("mode"), strict=False):
        use = model if model in group.columns else "GBM"
        group = group.assign(hour=group["timestamp"].dt.hour)
        group["abs_err"] = (group["y_actual"] - group[use]).abs()
        by_hour = group.groupby("hour")["abs_err"].mean()
        ax.bar(by_hour.index, by_hour.to_numpy(), color="steelblue")
        ax.set_xlabel("Hour of day (UTC)")
        ax.set_ylabel("Mean absolute error (kW)")
        ax.set_title(
            f"{model_labels.mode_label(mode)}\n{model_labels.label(use)}"
        )
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, "pv_v4_fig_app_error_by_hour")


def figure_error_by_clearness(model: str = "NNLSStack") -> None:
    df = _load_predictions()
    if df is None:
        return
    if "clearness_kt" not in df.columns:
        print("skip error-by-clearness: clearness_kt not in predictions (rerun V4)")
        return
    df = df[df["is_daylight"] == 1].copy()
    bins = np.linspace(0, 1.2, 13)
    df["bin"] = pd.cut(df["clearness_kt"], bins)
    fig, axes = plt.subplots(1, df["mode"].nunique(), figsize=(6.5, 3.0), squeeze=False)
    for ax, (mode, group) in zip(axes[0], df.groupby("mode"), strict=False):
        use = model if model in group.columns else "GBM"
        group["abs_err"] = (group["y_actual"] - group[use]).abs()
        by_bin = group.groupby("bin", observed=True)["abs_err"].mean()
        centres = [interval.mid for interval in by_bin.index]
        ax.plot(centres, by_bin.to_numpy(), "o-", color="darkorange")
        ax.set_xlabel("Clearness index $k_t$")
        ax.set_ylabel("Mean absolute error (kW)")
        ax.set_title(
            f"{model_labels.mode_label(mode)}\n{model_labels.label(use)}"
        )
        ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, "pv_v4_fig_app_error_by_clearness")


def figure_skill_baselines() -> None:
    path = OUT_DIR / "pv_v4_metrics.csv"
    if not path.exists():
        print("skip skill-baselines: pv_v4_metrics.csv not found")
        return
    df = pd.read_csv(path)
    day = df[df["subset"] == "daylight"]
    order = BASELINES + BASE_LEARNERS + ENSEMBLES
    present = [m for m in order if m in set(day["model"])]
    fig, axes = plt.subplots(1, day["mode"].nunique(), figsize=(6.5, 3.6), squeeze=False)
    for ax, (mode, group) in zip(axes[0], day.groupby("mode"), strict=False):
        values = group.set_index("model")["nRMSE_pct"].reindex(present)
        colours = [
            "0.6" if m in BASELINES else ("steelblue" if m in BASE_LEARNERS else "seagreen")
            for m in present
        ]
        model_labels.check_covered(present)
        ax.barh(range(len(present)), values.to_numpy(), color=colours)
        ax.set_yticks(range(len(present)))
        ax.set_yticklabels([model_labels.short_label(m) for m in present])
        ax.invert_yaxis()
        ax.set_xlabel("Daylight nRMSE (% of capacity)")
        ax.set_title(MODE_LABEL.get(mode, mode))
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    _save(fig, "pv_v4_fig_app_skill_baselines")


def figure_ablation() -> None:
    path = OUT_DIR / "pv_v4_ablation.csv"
    if not path.exists():
        print("skip ablation: pv_v4_ablation.csv not found (run run_ablations.py --both)")
        return
    df = pd.read_csv(path)
    day = df[df["subset"] == "daylight"].copy()
    order = ["full", "no_alignment", "calendar_only", "no_temporal"]
    present_modes = [m for m in ("KFOLD", "TEMPORAL") if m in set(day["mode"])]
    fig, axes = plt.subplots(
        1, len(present_modes), figsize=(6.5, 2.9), squeeze=False, sharey=False
    )
    for panel_index, (ax, mode) in enumerate(zip(axes[0], present_modes, strict=False)):
        group = day[day["mode"] == mode].set_index("ablation")
        rows = [a for a in order if a in group.index]
        values = [group.loc[a, "ensemble_nRMSE_pct"] for a in rows]
        labels = [ABLATION_LABEL.get(a, a) for a in rows]
        colours = ["seagreen" if a == "full" else "firebrick" for a in rows]
        ypos = range(len(rows))
        ax.barh(list(ypos), values, color=colours)
        ax.set_yticks(list(ypos))
        ax.set_yticklabels(labels if panel_index == 0 else [])
        ax.invert_yaxis()
        ax.set_xlabel("Daylight nRMSE (% of capacity)")
        ax.set_title(MODE_LABEL.get(mode, mode))
        ax.grid(axis="x", alpha=0.25)
        full_val = group.loc["full", "ensemble_nRMSE_pct"] if "full" in group.index else None
        for y, (a, v) in enumerate(zip(rows, values, strict=False)):
            tag = f"{v:.2f}"
            if full_val is not None and a != "full":
                delta = v - full_val
                sign = "+" if delta >= 0 else "\u2212"
                tag += f" ({sign}{abs(delta):.2f})"
            ax.text(v, y, " " + tag, va="center", ha="left", fontsize=7.5)
        ax.set_xlim(0, max(values) * 1.32)
    fig.tight_layout()
    _save(fig, "pv_v4_fig_app_ablation")


def figure_alignment() -> None:
    path = OUT_DIR / "pv_time_shift_scan.csv"
    if not path.exists():
        print("skip alignment: pv_time_shift_scan.csv not found (optional)")
        return
    df = pd.read_csv(path)
    shift_col = next((c for c in df.columns if "shift" in c.lower()), df.columns[0])
    metric_col = next(
        (c for c in df.columns if c.lower() in {"r2", "rmse", "nrmse", "nrmse_pct"}),
        df.columns[-1],
    )
    fig, ax = plt.subplots(figsize=(4.55, 3.0))
    ax.plot(df[shift_col], df[metric_col], "o-", color="purple")
    ax.set_xlabel("Candidate PV timestamp shift (hours)")
    ax.set_ylabel(model_labels.metric_label(metric_col))
    ax.set_title("Clear-sky agreement against timestamp shift")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, "pv_v4_fig_app_alignment")


def main() -> None:
    figure_monthly_skill()
    figure_missingness()
    figure_error_by_hour()
    figure_error_by_clearness()
    figure_skill_baselines()
    figure_ablation()
    figure_alignment()


if __name__ == "__main__":
    main()
