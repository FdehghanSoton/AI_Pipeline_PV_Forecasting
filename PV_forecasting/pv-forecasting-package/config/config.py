"""Central configuration for the PV-forecasting pipeline.

This module collects the site geometry, capacity and daylight policies,
evaluation settings, seeds, and feature/ablation switches that were
previously scattered as magic numbers across the version-numbered scripts.

All switches can be overridden through environment variables so that a single
command can reproduce a specific ablation or sensitivity run, for example::

    PV_TIME_SHIFT=0 PV_RUN_TAG=ablation_noshift python analyze_pv_v4.py
    PV_USE_PHYSICS=0 PV_RUN_TAG=ablation_calendar python analyze_pv_v4.py
    PV_DROP_MISSING_DAYS=1 PV_RUN_TAG=sens_nomissing python analyze_pv_v4.py

The defaults select the leakage-safe policies (fold-internal capacity for the
learned signal and a geometry-based daylight mask).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

OUT_DIR = Path(__file__).parent

# Acceptable absolute tolerances for full-pipeline regression checks. A rerun
# with identical configuration and seeds should reproduce the committed
# headline metrics to within these bounds; small drift is expected from
# library version changes and non-deterministic GPU kernels.
REGRESSION_ABS_TOL_R2 = 0.01
REGRESSION_ABS_TOL_NRMSE_PCT = 0.5


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None else value.strip()


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _env_int_tuple(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    parts = [p for p in value.replace(";", ",").split(",") if p.strip() != ""]
    return tuple(int(p) for p in parts)


@dataclass(frozen=True)
class SiteConfig:
    """Site geometry. Tilt and azimuth are modelling assumptions, not verified
    plant metadata; replace once the installation specification is confirmed."""

    lat: float = 50.91
    lon: float = -1.40
    alt: float = 30.0
    tilt: float = 30.0
    azimuth: float = 180.0


@dataclass(frozen=True)
class RunConfig:
    """Pipeline behaviour and ablation switches.

    Attributes
    ----------
    time_shift_hours
        Backward PV-index shift applied before joining with weather. The
        default ``-1`` is the empirically selected alignment correction; set
        to ``0`` for the no-alignment ablation.
    capacity_policy
        ``"fold_train"`` estimates the capacity used by the clearness target
        and CNN normalisation from each fold's training rows (leakage-safe);
        ``"global"`` uses the full-series 99.9th percentile everywhere.
    metric_capacity_policy
        Denominator used for normalised metrics. Kept at ``"global"`` so that
        normalised errors are comparable across folds and models.
    daylight_policy
        ``"geometric"`` flags daylight from solar elevation (no PV labels);
        ``"pv_median"`` reproduces the legacy per-(month, hour) PV-median rule.
    use_physics_features / use_temporal_context
        Feature-group ablation switches.
    drop_missing_days
        Missing-data sensitivity: drop every day containing any missing PV
        measurement before backtesting.
    cnn_seeds
        One or more seeds for the CNN. With several seeds the per-fold CNN
        prediction is the mean across seeds (seed averaging).
    """

    time_shift_hours: int = -1
    capacity_policy: str = "fold_train"
    metric_capacity_policy: str = "global"
    daylight_policy: str = "geometric"
    daylight_elevation_deg: float = 5.0
    daylight_pct: float = 0.05
    use_physics_features: bool = True
    use_temporal_context: bool = True
    drop_missing_days: bool = False
    include_baselines: bool = True
    cnn_seeds: tuple[int, ...] = (0,)
    kfold_n_folds: int = 5
    temporal_n_folds: int = 4
    first_test_days: int = 120
    seed: int = 0
    run_tag: str = ""
    site: SiteConfig = field(default_factory=SiteConfig)

    def tagged(self, name: str) -> Path:
        """Return an output path, inserting ``run_tag`` before the suffix."""
        path = Path(name)
        if not self.run_tag:
            return OUT_DIR / path.name
        stem = path.stem
        suffix = path.suffix
        return OUT_DIR / f"{stem}__{self.run_tag}{suffix}"

    def describe(self) -> dict[str, object]:
        """Machine-readable record of the active configuration for provenance."""
        return {
            "time_shift_hours": self.time_shift_hours,
            "capacity_policy": self.capacity_policy,
            "metric_capacity_policy": self.metric_capacity_policy,
            "daylight_policy": self.daylight_policy,
            "daylight_elevation_deg": self.daylight_elevation_deg,
            "daylight_pct": self.daylight_pct,
            "use_physics_features": self.use_physics_features,
            "use_temporal_context": self.use_temporal_context,
            "drop_missing_days": self.drop_missing_days,
            "include_baselines": self.include_baselines,
            "cnn_seeds": list(self.cnn_seeds),
            "kfold_n_folds": self.kfold_n_folds,
            "temporal_n_folds": self.temporal_n_folds,
            "first_test_days": self.first_test_days,
            "seed": self.seed,
            "run_tag": self.run_tag,
            "site": {
                "lat": self.site.lat,
                "lon": self.site.lon,
                "alt": self.site.alt,
                "tilt": self.site.tilt,
                "azimuth": self.site.azimuth,
            },
        }


def load_config() -> RunConfig:
    """Build a :class:`RunConfig` from environment variables and defaults.

    Environment variables are read at call time, so a process can set them
    before invoking the pipeline and tests can override them in isolation.
    """
    defaults = RunConfig()
    return RunConfig(
        time_shift_hours=_env_int("PV_TIME_SHIFT", defaults.time_shift_hours),
        capacity_policy=_env_str("PV_CAPACITY_POLICY", defaults.capacity_policy),
        metric_capacity_policy=_env_str(
            "PV_METRIC_CAPACITY_POLICY", defaults.metric_capacity_policy
        ),
        daylight_policy=_env_str("PV_DAYLIGHT_POLICY", defaults.daylight_policy),
        daylight_elevation_deg=_env_float(
            "PV_DAYLIGHT_ELEV_DEG", defaults.daylight_elevation_deg
        ),
        daylight_pct=_env_float("PV_DAYLIGHT_PCT", defaults.daylight_pct),
        use_physics_features=_env_flag("PV_USE_PHYSICS", defaults.use_physics_features),
        use_temporal_context=_env_flag(
            "PV_USE_TEMPORAL", defaults.use_temporal_context
        ),
        drop_missing_days=_env_flag(
            "PV_DROP_MISSING_DAYS", defaults.drop_missing_days
        ),
        include_baselines=_env_flag(
            "PV_INCLUDE_BASELINES", defaults.include_baselines
        ),
        cnn_seeds=_env_int_tuple("PV_CNN_SEEDS", defaults.cnn_seeds),
        kfold_n_folds=_env_int("PV_KFOLD_FOLDS", defaults.kfold_n_folds),
        temporal_n_folds=_env_int("PV_TEMPORAL_FOLDS", defaults.temporal_n_folds),
        first_test_days=_env_int("PV_FIRST_TEST_DAYS", defaults.first_test_days),
        seed=_env_int("PV_SEED", defaults.seed),
        run_tag=_env_str("PV_RUN_TAG", defaults.run_tag),
    )
