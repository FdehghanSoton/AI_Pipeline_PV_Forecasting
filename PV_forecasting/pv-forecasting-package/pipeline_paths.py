"""Shared filesystem locations for private inputs and generated artefacts.

The defaults deliberately keep confidential measurements and camera-ready
outputs in Git-ignored directories. The orchestration command sets these
environment variables explicitly for every subprocess, so individual scripts
agree on one input directory and one run-specific output directory.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def _environment_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


DATA_DIR = _environment_path("PV_DATA_DIR", PACKAGE_ROOT / "private_data")
OUTPUT_DIR = _environment_path(
    "PV_OUTPUT_DIR", PACKAGE_ROOT / "camera_ready_outputs"
)
PV_DATA_PATH = _environment_path("PV_DATA_PATH", DATA_DIR / "PV_data.csv")
WEATHER_CACHE_PATH = _environment_path(
    "PV_WEATHER_CACHE_PATH", DATA_DIR / "weather_cache.csv"
)
WEATHER_META_PATH = _environment_path(
    "PV_WEATHER_META_PATH", DATA_DIR / "weather_cache.meta.json"
)


def ensure_output_dir() -> Path:
    """Create and return the active generated-output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
