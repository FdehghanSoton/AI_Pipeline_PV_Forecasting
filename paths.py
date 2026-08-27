"""Where the pipeline reads its inputs and writes its outputs.

Every script in this repository resolves paths through this module rather than
relative to its own file, so that the measurements, the weather caches and the
generated results each live in one obvious place and a script can be run from
any working directory.

``PV_data.csv`` is deliberately absent from the repository: it is site
measurement data that is not ours to publish. ``data/README.md`` records what
it must contain and how to obtain it. Everything else needed to reproduce the
paper, including the weather caches, is committed.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DATA_DIR = REPO_ROOT / "data"
WEATHER_DIR = DATA_DIR / "weather"
RESULTS_DIR = REPO_ROOT / "results"

PV_CSV = DATA_DIR / "PV_data.csv"


def results_dir() -> Path:
    """Return the results directory, creating it on first use."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def weather_dir() -> Path:
    """Return the weather-cache directory, creating it on first use."""
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    return WEATHER_DIR


def require_pv_csv() -> Path:
    """Return the path to the PV measurements, with a useful error if absent."""
    if not PV_CSV.exists():
        raise FileNotFoundError(
            f"{PV_CSV} not found. The site measurements are not distributed "
            "with this repository; see data/README.md for the expected format "
            "and how to obtain them."
        )
    return PV_CSV
