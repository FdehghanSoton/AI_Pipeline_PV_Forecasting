"""Weather-source experiment: retrospective analysis versus a constant-lead forecast product.

The headline results take target-day weather from a historical analysis
archive, which no operational forecaster can access before the target day.
This script quantifies what that hindsight is worth by running the same
pipeline on four weather inputs:

``analysis_full``
    ECMWF IFS operational analysis, all fifteen variables. This reproduces the
    paper's headline configuration.

``analysis_matched``
    ECMWF IFS operational analysis, restricted to the twelve variables the
    previous-runs forecast product also serves. This is the correct comparison
    point for the forecast-product run, because it differs from it only in
    whether the target day is known.

``forecast_day1``
    A constant 24-hour-lead profile from the Open-Meteo previous-runs product,
    twelve variables. The profile is assembled from successive model updates,
    not from one forecast issuance, and the request is not pinned to the IFS.

``era5_full``
    ERA5 reanalysis, all fifteen variables. Weather-product sensitivity only.

The ``analysis_matched`` to ``forecast_day1`` step is the forecast-input
penalty: it mixes forecast lead with a change of weather product.

Each configuration is a full backtest under both protocols, so this script
runs the pipeline four times.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace

import pandas as pd

import analyze_pv_v4 as pipeline
import paths
from baselines import BASELINE_NAMES, add_skill_columns
from config import RunConfig, load_config
from stats_tests import diebold_mariano

OUT_DIR = paths.results_dir()


@dataclass(frozen=True)
class Configuration:
    key: str
    label: str
    weather_source: str
    weather_variable_set: str


CONFIGURATIONS: tuple[Configuration, ...] = (
    Configuration(
        "analysis_full",
        "IFS analysis, 15 variables",
        "ifs",
        "full",
    ),
    Configuration(
        "analysis_matched",
        "IFS analysis, 12 variables",
        "ifs",
        "forecast_matched",
    ),
    Configuration(
        "forecast_day1",
        "Day-ahead forecast, 12 variables",
        "forecast_day1",
        "forecast_matched",
    ),
    Configuration(
        "era5_full",
        "ERA5 reanalysis, 15 variables",
        "era5",
        "full",
    ),
)

MODE_LABEL = {"KFOLD": "Random day-fold", "TEMPORAL": "Rolling-origin"}


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def run_configuration(
    configuration: Configuration, base: RunConfig
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Run both backtests for one weather configuration.

    Returns the pooled metric table, the per-timestamp predictions, and the
    number of input features the configuration produced.
    """
    cfg = replace(
        base,
        weather_source=configuration.weather_source,
        weather_variable_set=configuration.weather_variable_set,
        run_tag=f"weather_{configuration.key}",
    )
    pipeline.set_global_seed(cfg.seed)
    pv, feats, capacity = pipeline.build_dataset(cfg)

    temporal = pipeline.temporal_backtest(
        pv,
        pv,
        capacity,
        feats,
        n_folds=cfg.temporal_n_folds,
        first_test_days=cfg.first_test_days,
        cfg=cfg,
    )
    kfold = pipeline.kfold_backtest(
        pv, pv, capacity, feats, n_folds=cfg.kfold_n_folds, seed=cfg.seed, cfg=cfg
    )

    metrics = pd.concat(
        [
            pipeline.aggregate(temporal, capacity, "TEMPORAL"),
            pipeline.aggregate(kfold, capacity, "KFOLD"),
        ],
        ignore_index=True,
    )
    if cfg.include_baselines:
        metrics = add_skill_columns(metrics, reference="SmartPersistence")
    metrics.insert(0, "configuration", configuration.key)

    predictions = pd.concat(
        [
            pipeline.folds_to_frame(temporal, "TEMPORAL"),
            pipeline.folds_to_frame(kfold, "KFOLD"),
        ],
        ignore_index=True,
    )
    predictions.insert(0, "configuration", configuration.key)

    cfg.tagged("pv_v4_run_config.json").write_text(
        json.dumps(cfg.describe(), indent=2) + "\n", encoding="utf-8"
    )
    return metrics, predictions, len(feats)


def best_per_mode(metrics: pd.DataFrame) -> pd.DataFrame:
    """Pre-declared stack and best base learner on daylight rows, per protocol.

    Every configuration is summarised by ``NNLSStack``. Following one method
    across weather products keeps the comparison, and the paired test built on
    it below, from mixing a change of weather input with a change of fusion
    method.
    """
    rows = []
    daylight = metrics[metrics["subset"] == "daylight"]
    for (configuration, mode), group in daylight.groupby(["configuration", "mode"]):
        stacks = group[group["model"] == "NNLSStack"]
        bases = group[group["model"].isin(pipeline.BASE_LEARNERS)]
        reference = group[group["model"] == "SmartPersistence"]
        if stacks.empty or bases.empty:
            continue
        stack = stacks.iloc[0]
        best_base = bases.sort_values("RMSE").iloc[0]
        rows.append(
            {
                "configuration": configuration,
                "mode": mode,
                "n": int(stack["n"]),
                "best_ensemble": "NNLSStack",
                "ensemble_R2": float(stack["R2"]),
                "ensemble_RMSE": float(stack["RMSE"]),
                "ensemble_nRMSE_pct": float(stack["nRMSE_pct"]),
                "best_base": best_base["model"],
                "base_R2": float(best_base["R2"]),
                "base_nRMSE_pct": float(best_base["nRMSE_pct"]),
                "smart_persistence_nRMSE_pct": (
                    float(reference["nRMSE_pct"].iloc[0])
                    if not reference.empty
                    else float("nan")
                ),
                "skill_vs_smart_persistence": (
                    float(stack["skill_vs_SmartPersistence"])
                    if "skill_vs_SmartPersistence" in stack
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


# Pairwise comparisons worth a significance test. Each is (reference, variant,
# name), where the variant is the configuration whose extra realism or
# different product is being priced.
COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("analysis_matched", "forecast_day1", "operational_penalty"),
    ("analysis_full", "era5_full", "weather_product_sensitivity"),
    ("analysis_full", "analysis_matched", "dropped_cloud_levels"),
)


def compare_configurations(
    summary: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    """Price each configuration change against its reference.

    Every comparison is made on the identical held-out timestamps, and a
    Diebold-Mariano test reports whether the change in squared error is
    distinguishable from zero. Because the daylight-hour sets are identical
    across configurations by construction, this is a paired comparison.
    """
    rows = []
    for reference_key, variant_key, name in COMPARISONS:
        for mode in ("KFOLD", "TEMPORAL"):
            reference = _summary_row(summary, reference_key, mode)
            variant = _summary_row(summary, variant_key, mode)
            if reference is None or variant is None:
                continue

            reference_predictions = _daylight_series(
                predictions, reference_key, mode, reference["best_ensemble"], "reference"
            )
            variant_predictions = _daylight_series(
                predictions, variant_key, mode, variant["best_ensemble"], "variant"
            )
            joined = reference_predictions.join(
                variant_predictions[["variant"]], how="inner"
            )
            test = diebold_mariano(
                joined["y_actual"].to_numpy(),
                joined["variant"].to_numpy(),
                joined["reference"].to_numpy(),
                loss="squared",
                name_a=variant_key,
                name_b=reference_key,
            )
            rows.append(
                {
                    "comparison": name,
                    "reference": reference_key,
                    "variant": variant_key,
                    "mode": mode,
                    "n_common": int(len(joined)),
                    "reference_nRMSE_pct": reference["ensemble_nRMSE_pct"],
                    "variant_nRMSE_pct": variant["ensemble_nRMSE_pct"],
                    "nRMSE_change_pp": (
                        variant["ensemble_nRMSE_pct"] - reference["ensemble_nRMSE_pct"]
                    ),
                    "relative_rmse_change_pct": (
                        100.0
                        * (variant["ensemble_RMSE"] / reference["ensemble_RMSE"] - 1.0)
                    ),
                    "reference_R2": reference["ensemble_R2"],
                    "variant_R2": variant["ensemble_R2"],
                    "reference_skill": reference["skill_vs_smart_persistence"],
                    "variant_skill": variant["skill_vs_smart_persistence"],
                    **test.as_dict(),
                }
            )
    return pd.DataFrame(rows)


def _summary_row(summary: pd.DataFrame, configuration: str, mode: str):
    match = summary[
        (summary["configuration"] == configuration) & (summary["mode"] == mode)
    ]
    return None if match.empty else match.iloc[0]


def _daylight_series(
    predictions: pd.DataFrame,
    configuration: str,
    mode: str,
    model: str,
    column: str,
) -> pd.DataFrame:
    subset = predictions[
        (predictions["configuration"] == configuration)
        & (predictions["mode"] == mode)
        & (predictions["is_missing"] == 0)
        & (predictions["is_daylight"] == 1)
    ]
    frame = subset[["timestamp", "y_actual", model]].rename(columns={model: column})
    frame = frame.drop_duplicates(subset="timestamp").set_index("timestamp")
    return frame.sort_index()


def baseline_degradation(metrics: pd.DataFrame) -> pd.DataFrame:
    """How much each reference baseline degrades under forecast weather.

    Smart persistence multiplies yesterday's measured clearness by the
    target-day plane-of-array irradiance, which is transposed from the weather
    product. It therefore inherits the weather product's error, and under the
    analysis configurations it also inherits the analysis's hindsight. This
    table makes that dependence explicit, because it is what allows the
    ensemble's skill score to hold up even as its absolute error grows.
    """
    daylight = metrics[metrics["subset"] == "daylight"]
    baselines = daylight[daylight["model"].isin(BASELINE_NAMES)]
    return (
        baselines.pivot_table(
            index=["mode", "model"], columns="configuration", values="nRMSE_pct"
        )
        .round(3)
        .reset_index()
    )


def main() -> None:
    base = load_config()
    all_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    feature_counts: dict[str, int] = {}

    for configuration in CONFIGURATIONS:
        _log(f"=== {configuration.key}: {configuration.label} ===")
        metrics, predictions, n_features = run_configuration(configuration, base)
        all_metrics.append(metrics)
        all_predictions.append(predictions)
        feature_counts[configuration.key] = n_features

    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics.to_csv(OUT_DIR / "pv_v4_weather_metrics.csv", index=False)
    predictions.to_csv(OUT_DIR / "pv_v4_weather_predictions.csv", index=False)

    summary = best_per_mode(metrics)
    summary["n_features"] = summary["configuration"].map(feature_counts)
    summary.to_csv(OUT_DIR / "pv_v4_weather_summary.csv", index=False)

    comparisons = compare_configurations(summary, predictions)
    if not comparisons.empty:
        comparisons.to_csv(OUT_DIR / "pv_v4_weather_comparisons.csv", index=False)

    degradation = baseline_degradation(metrics)
    degradation.to_csv(OUT_DIR / "pv_v4_weather_baselines.csv", index=False)

    label = {c.key: c.label for c in CONFIGURATIONS}
    print("\n=== Daylight results by weather source (best ensemble per protocol) ===")
    for mode in ("KFOLD", "TEMPORAL"):
        print(f"\n--- {MODE_LABEL[mode]} ---")
        view = summary[summary["mode"] == mode].copy()
        view["weather input"] = view["configuration"].map(label)
        columns = [
            "weather input",
            "n_features",
            "best_ensemble",
            "ensemble_R2",
            "ensemble_nRMSE_pct",
            "smart_persistence_nRMSE_pct",
            "skill_vs_smart_persistence",
        ]
        print(view[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(
        "\n=== Reference baselines by weather source "
        "(daylight nRMSE, % of capacity) ==="
    )
    print(degradation.to_string(index=False))

    if not comparisons.empty:
        print("\n=== Paired comparisons with Diebold-Mariano tests ===")
        columns = [
            "comparison",
            "mode",
            "n_common",
            "reference_nRMSE_pct",
            "variant_nRMSE_pct",
            "relative_rmse_change_pct",
            "reference_skill",
            "variant_skill",
            "p_value",
            "better",
        ]
        print(
            comparisons[columns].to_string(
                index=False, float_format=lambda v: f"{v:.4g}"
            )
        )

    print(
        "\nWrote pv_v4_weather_metrics.csv, pv_v4_weather_predictions.csv, "
        "pv_v4_weather_summary.csv, pv_v4_weather_comparisons.csv and "
        "pv_v4_weather_baselines.csv"
    )


if __name__ == "__main__":
    main()
