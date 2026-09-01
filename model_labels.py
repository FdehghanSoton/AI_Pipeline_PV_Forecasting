"""Single source of truth for the names.

``LABEL`` is the name used in running text and in figure legends.
``SHORT_LABEL`` is the same name shortened for a crowded tick axis, and
``WRAPPED_LABEL`` breaks it across two lines for the same purpose. All three
describe the same model, so a reader moving between a figure and the text sees
consistent terminology.
"""

from __future__ import annotations

BASE_LEARNERS: tuple[str, ...] = ("Ridge", "GBM", "GBM_kt", "GBM_per_hour", "CNN")
ENSEMBLES: tuple[str, ...] = ("Mean", "InvRMSE", "RidgeStack", "NNLSStack")
BASELINES: tuple[str, ...] = (
    "SmartPersistence",
    "Persistence",
    "ClearSky",
    "Climatology",
)

LABEL: dict[str, str] = {
    "Ridge": "Ridge regression",
    "GBM": "Gradient boosting",
    "GBM_kt": "POA-normalised GBM",
    "GBM_per_hour": "Per-hour GBM",
    "CNN": "CNN",
    "Mean": "Arithmetic-mean ensemble",
    "InvRMSE": "Inverse-RMSE ensemble",
    "RidgeStack": "Ridge stacking",
    "NNLSStack": "NNLS stacking",
    "BestSingleByVal": "Best single model on validation",
    "SmartPersistence": "Smart persistence",
    "Persistence": "Persistence",
    "ClearSky": "Clear-sky scaling",
    "Climatology": "Hourly climatology",
}

SHORT_LABEL: dict[str, str] = {
    "Ridge": "Ridge",
    "GBM": "GBM",
    "GBM_kt": "POA-norm. GBM",
    "GBM_per_hour": "Per-hour GBM",
    "CNN": "CNN",
    "Mean": "Mean ensemble",
    "InvRMSE": "Inverse RMSE",
    "RidgeStack": "Ridge stacking",
    "NNLSStack": "NNLS stacking",
    "BestSingleByVal": "Best single",
    "SmartPersistence": "Smart persistence",
    "Persistence": "Persistence",
    "ClearSky": "Clear sky",
    "Climatology": "Climatology",
}

WRAPPED_LABEL: dict[str, str] = {
    "Ridge": "Ridge",
    "GBM": "GBM",
    "GBM_kt": "POA-norm.\nGBM",
    "GBM_per_hour": "Per-hour\nGBM",
    "CNN": "CNN",
    "Mean": "Mean\nensemble",
    "InvRMSE": "Inverse\nRMSE",
    "RidgeStack": "Ridge\nstacking",
    "NNLSStack": "NNLS\nstacking",
    "BestSingleByVal": "Best\nsingle",
    "SmartPersistence": "Smart\npersistence",
    "Persistence": "Persistence",
    "ClearSky": "Clear\nsky",
    "Climatology": "Climatology",
}

MODE_LABEL: dict[str, str] = {
    "KFOLD": "Random day-fold",
    "TEMPORAL": "Rolling-origin",
}

# Metric column names carry units, so an axis label taken straight from a CSV
# header is both unreadable and unitless. Keys are lower-cased for lookup.
METRIC_LABEL: dict[str, str] = {
    "r2": "Daylight $R^2$",
    "mae": "Mean absolute error (W)",
    "rmse": "Root mean squared error (W)",
    "nmae": "Normalised MAE (% of capacity)",
    "nmae_pct": "Normalised MAE (% of capacity)",
    "nrmse": "Normalised RMSE (% of capacity)",
    "nrmse_pct": "Normalised RMSE (% of capacity)",
    "skill": "Skill score against smart persistence",
    "n": "Number of hours",
}


def label(key: str) -> str:
    """Reader-facing name for a model key, for text and legends."""
    return LABEL.get(key, key)


def short_label(key: str) -> str:
    """Shortened reader-facing name, for crowded tick axes."""
    return SHORT_LABEL.get(key, LABEL.get(key, key))


def wrapped_label(key: str) -> str:
    """Two-line reader-facing name, for narrow tick axes."""
    return WRAPPED_LABEL.get(key, SHORT_LABEL.get(key, LABEL.get(key, key)))


def mode_label(key: str) -> str:
    """Reader-facing name for an evaluation protocol."""
    return MODE_LABEL.get(key, key)


def metric_label(column: str) -> str:
    """Axis label with units for a metric column name."""
    return METRIC_LABEL.get(column.lower(), column)


def is_ensemble(key: str) -> bool:
    return key in ENSEMBLES


def check_covered(keys: object) -> None:
    """Fail loudly if a model key would reach a figure without a proper name.

    Called by the figure scripts before plotting, so that adding a model
    without adding its name is caught at build time rather than discovered in
    a submitted PDF.
    """
    unknown = sorted({str(k) for k in keys} - set(LABEL))
    if unknown:
        raise KeyError(
            "No reader-facing label for model key(s): "
            + ", ".join(unknown)
            + ". Add them to model_labels.LABEL before plotting."
        )
