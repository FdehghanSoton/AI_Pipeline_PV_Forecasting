"""Draw the pipeline architecture diagram.

Run ``python make_pipeline_diagram.py``. Writes ``pipeline_diagram.pdf`` and
``.png``. Model names come from ``model_labels``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import model_labels
import paths

OUT_DIR = paths.results_dir()

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    }
)

STAGE_FILL = "#eef2f7"
STAGE_EDGE = "#4a6382"
INPUT_FILL = "#e8f0e6"
INPUT_EDGE = "#4a7c46"
MODEL_FILL = "#fdf2e2"
MODEL_EDGE = "#b5822e"
OUTPUT_FILL = "#f2e8f0"
OUTPUT_EDGE = "#8a4a78"

FIGURE_WIDTH_IN = 6.5

COLUMN_WIDTH = 2.38
COLUMN_GAP = 0.24
TITLE_SIZE = 8.5
BODY_SIZE = 7.2
TITLE_LINE_HEIGHT = 0.24
BODY_LINE_HEIGHT = 0.215


def _box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    lines: list[str],
    fill: str,
    edge: str,
) -> None:
    """Draw one stage. The body starts below however many lines the title uses."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=fill,
            edgecolor=edge,
            linewidth=0.9,
        )
    )
    title_lines = title.count("\n") + 1
    ax.text(
        x + width / 2,
        y + height - 0.15,
        title,
        ha="center",
        va="top",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        color=edge,
        linespacing=1.15,
    )
    body_top = y + height - 0.19 - TITLE_LINE_HEIGHT * title_lines
    for index, line in enumerate(lines):
        ax.text(
            x + 0.09,
            body_top - BODY_LINE_HEIGHT * index,
            line,
            ha="left",
            va="top",
            fontsize=BODY_SIZE,
        )


def _arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.9,
            color="#55606c",
            shrinkA=0,
            shrinkB=0,
        )
    )


def draw() -> None:
    n_columns = 5
    width = 0.1 * 2 + n_columns * COLUMN_WIDTH + (n_columns - 1) * COLUMN_GAP

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 3.05))
    ax.set_xlim(0, width)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    def left_of(column: int) -> float:
        return 0.1 + column * (COLUMN_WIDTH + COLUMN_GAP)

    def right_of(column: int) -> float:
        return left_of(column) + COLUMN_WIDTH

    def centre_of(column: int) -> float:
        return left_of(column) + COLUMN_WIDTH / 2

    _box(
        ax,
        left_of(0),
        2.65,
        COLUMN_WIDTH,
        2.0,
        "PV measurements",
        [
            "Inverter AC power,",
            "hourly, one year",
            "9.25% of hours",
            "unobserved",
        ],
        INPUT_FILL,
        INPUT_EDGE,
    )
    _box(
        ax,
        left_of(0),
        0.45,
        COLUMN_WIDTH,
        2.0,
        "Weather inputs",
        [
            "Open-Meteo archive,",
            "ECMWF IFS analysis",
            "15 hourly variables",
            "(12 for the day-ahead",
            "forecast experiment)",
        ],
        INPUT_FILL,
        INPUT_EDGE,
    )

    _box(
        ax,
        left_of(1),
        1.30,
        COLUMN_WIDTH,
        2.5,
        "Alignment and\nquality control",
        [
            "Correct timestamp",
            "offset ($-1$ h)",
            "Flag missing hours",
            "Keep complete days",
        ],
        STAGE_FILL,
        STAGE_EDGE,
    )

    _box(
        ax,
        left_of(2),
        1.30,
        COLUMN_WIDTH,
        2.5,
        "Solar and weather\nfeatures",
        [
            "Solar geometry (SPA)",
            "Clearness index $k_t$",
            "Plane-of-array",
            "  irradiance",
            "Lag, lead and rolling",
            "  weather context",
            "Calendar harmonics",
        ],
        STAGE_FILL,
        STAGE_EDGE,
    )

    learners = [model_labels.label(key) for key in model_labels.BASE_LEARNERS]
    _box(
        ax,
        left_of(3),
        1.30,
        COLUMN_WIDTH,
        2.5,
        "Base learners",
        list(learners),
        MODEL_FILL,
        MODEL_EDGE,
    )

    _box(
        ax,
        left_of(4),
        2.65,
        COLUMN_WIDTH,
        2.0,
        "Ensemble fusion",
        [
            "Weights fitted on",
            "validation rows only:",
            "mean, inverse RMSE,",
            "ridge and NNLS",
            "  stacking",
        ],
        MODEL_FILL,
        MODEL_EDGE,
    )
    _box(
        ax,
        left_of(4),
        0.45,
        COLUMN_WIDTH,
        2.0,
        "Evaluation",
        [
            "Random day-fold and",
            "rolling-origin splits",
            "Daylight metrics, skill",
            "over three baselines",
        ],
        OUTPUT_FILL,
        OUTPUT_EDGE,
    )

    _arrow(ax, (right_of(0), 3.65), (left_of(1), 2.95))
    _arrow(ax, (right_of(0), 1.45), (left_of(1), 2.15))
    _arrow(ax, (right_of(1), 2.55), (left_of(2), 2.55))
    _arrow(ax, (right_of(2), 2.55), (left_of(3), 2.55))
    _arrow(ax, (right_of(3), 2.95), (left_of(4), 3.65))
    _arrow(ax, (centre_of(4), 2.65), (centre_of(4), 2.45))

    ax.text(
        width / 2,
        0.06,
        "Every fitted quantity, including the capacity estimate, ensemble weights\n"
        "and feature scaling, is estimated inside the training fold only.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        style="italic",
        color="#55606c",
    )

    ax.set_position((0.0, 0.0, 1.0, 1.0))
    for extension in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"pipeline_diagram.{extension}")
    plt.close(fig)
    print("Wrote pipeline_diagram.pdf and .png")


if __name__ == "__main__":
    draw()
