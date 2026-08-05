from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from run_camera_ready_pipeline import (
    APPENDIX_FIGURES,
    MAIN_OUTPUTS,
    PACKAGE_ROOT,
    build_steps,
    create_manifest,
    enforce_private_path,
)
from summarise_camera_ready_results import REQUIRED_INPUTS, generate_summary


def test_execution_plan_covers_all_camera_ready_evidence() -> None:
    steps = build_steps(sys.executable, [0, 1, 2])
    names = [step.name for step in steps]
    assert names == [
        "data_audit",
        "alignment_scan",
        "main_experiment",
        "multiseed_kfold",
        "multiseed_temporal",
        "ablations",
        "result_summary",
        "paper_figures",
        "results_figure",
        "appendix_figures",
    ]
    assert "pv_v4_per_fold_metrics.csv" in MAIN_OUTPUTS
    assert "pv_v4_significance.csv" in MAIN_OUTPUTS
    assert "pv_v4_fig_app_ablation.pdf" in APPENDIX_FIGURES
    assert "pv_v4_fig_app_alignment.pdf" in APPENDIX_FIGURES
    kfold, temporal = steps[3], steps[4]
    assert set(kfold.expected_outputs).isdisjoint(temporal.expected_outputs)
    assert dict(kfold.environment)["PV_RUN_TAG"] == "multiseed_kfold"
    assert dict(temporal.environment)["PV_RUN_TAG"] == "multiseed_temporal"
    assert tuple(kfold.command[-3:]) == ("0", "1", "2")


def _write_synthetic_outputs(directory: Path) -> None:
    metric_rows = []
    significance_rows = []
    for mode in ("KFOLD", "TEMPORAL"):
        for subset in ("ALL", "daylight"):
            offset = 1.0 if subset == "daylight" else 0.0
            metric_rows.extend(
                [
                    {
                        "mode": mode,
                        "model": "SmartPersistence",
                        "subset": subset,
                        "R2": 0.70,
                        "MAE": 7.0,
                        "nMAE_pct": 7.0,
                        "RMSE": 10.0 + offset,
                        "nRMSE_pct": 10.0 + offset,
                    },
                    {
                        "mode": mode,
                        "model": "Ridge",
                        "subset": subset,
                        "R2": 0.80,
                        "MAE": 6.0,
                        "nMAE_pct": 6.0,
                        "RMSE": 9.0 + offset,
                        "nRMSE_pct": 9.0 + offset,
                    },
                    {
                        "mode": mode,
                        "model": "NNLSStack",
                        "subset": subset,
                        "R2": 0.85,
                        "MAE": 5.0,
                        "nMAE_pct": 5.0,
                        "RMSE": 8.0 + offset,
                        "nRMSE_pct": 8.0 + offset,
                    },
                ]
            )
            significance_rows.append(
                {
                    "mode": mode,
                    "subset": subset,
                    "best_ensemble": "NNLSStack",
                    "best_base": "Ridge",
                    "abs_rmse_gain": 1.0,
                    "rel_rmse_gain_pct": 100.0 / (9.0 + offset),
                    "dm_statistic": -2.5,
                    "p_value": 0.02,
                    "n": 100,
                    "better": "NNLSStack",
                    "loss": "squared",
                }
            )
    pd.DataFrame(metric_rows).to_csv(directory / "pv_v4_metrics.csv", index=False)
    pd.DataFrame(significance_rows).to_csv(
        directory / "pv_v4_significance.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "mode": mode,
                "fold": "fold_1",
                "model": "NNLSStack",
                "subset": "daylight",
                "RMSE": 8.0,
            }
            for mode in ("KFOLD", "TEMPORAL")
        ]
    ).to_csv(directory / "pv_v4_per_fold_metrics.csv", index=False)

    multiseed = pd.DataFrame(
        [{"mode": "KFOLD", "model": "NNLSStack", "subset": "daylight", "seed": 0}]
    )
    multiseed_summary = pd.DataFrame(
        [
            {
                "mode": "KFOLD",
                "model": "NNLSStack",
                "subset": "daylight",
                "RMSE_mean": 8.0,
                "RMSE_sd": 0.1,
                "n_seeds": 1,
            }
        ]
    )
    for mode in ("kfold", "temporal"):
        raw = multiseed.assign(mode=mode.upper())
        summary = multiseed_summary.assign(mode=mode.upper())
        raw.to_csv(
            directory / f"pv_v4_multiseed_raw__multiseed_{mode}.csv", index=False
        )
        summary.to_csv(
            directory / f"pv_v4_multiseed_summary__multiseed_{mode}.csv",
            index=False,
        )
    pd.DataFrame(
        [
            {
                "ablation": "full",
                "mode": "KFOLD",
                "subset": "daylight",
                "best_ensemble": "NNLSStack",
                "ensemble_nRMSE_pct": 8.0,
            }
        ]
    ).to_csv(directory / "pv_v4_ablation.csv", index=False)
    pd.DataFrame(
        [
            {"shift_hours": shift, "R2": 0.8 - abs(shift + 1) * 0.1}
            for shift in range(-3, 4)
        ]
    ).to_csv(directory / "pv_time_shift_scan.csv", index=False)


def test_summary_derives_claims_without_hard_coded_results(tmp_path: Path) -> None:
    _write_synthetic_outputs(tmp_path)
    claims = generate_summary(tmp_path)

    headline = pd.read_csv(tmp_path / "camera_ready_headline_metrics.csv")
    row = headline[
        (headline["mode"] == "KFOLD") & (headline["subset"] == "ALL")
    ].iloc[0]
    assert row["improvement_vs_smart_persistence_pct"] == pytest.approx(20.0)
    assert row["improvement_vs_best_base_pct"] == pytest.approx(100.0 / 9.0)
    assert row["dm_p_value"] == pytest.approx(0.02)
    assert len(claims["headline_results"]) == 4
    assert (tmp_path / "camera_ready_claims.json").is_file()
    assert (tmp_path / "camera_ready_results_table.tex").is_file()
    assert all((tmp_path / name).is_file() for name in REQUIRED_INPUTS)


def test_manifest_contains_required_provenance_without_private_paths(
    tmp_path: Path,
) -> None:
    pv = tmp_path / "PV_data.csv"
    weather = tmp_path / "weather_cache.csv"
    metadata = tmp_path / "weather_cache.meta.json"
    pv.write_text("private pv", encoding="utf-8")
    weather.write_text("private weather", encoding="utf-8")
    metadata.write_text(
        json.dumps(
            {
                "weather_model": "era5",
                "source_url": "https://archive-api.open-meteo.com/v1/archive",
            }
        ),
        encoding="utf-8",
    )
    config = {
        "seed": 0,
        "cnn_seeds": [0],
        "daylight_elevation_deg": 5.0,
        "metric_capacity_policy": "global",
        "site": {
            "lat": 50.91,
            "lon": -1.40,
            "tilt": 30.0,
            "azimuth": 180.0,
        },
    }
    manifest = create_manifest(
        ["python", "run_camera_ready_pipeline.py", "--data-dir", str(tmp_path)],
        {"commit": "abc123", "branch": "Masood-dev", "dirty": False},
        config,
        [0, 1, 2],
        pv,
        weather,
        metadata,
    )
    serialised = json.dumps(manifest)
    assert str(tmp_path) not in serialised
    assert manifest["command"][-1] == "<private-data-dir>"
    assert manifest["weather"]["source"] == "era5"
    assert manifest["data"]["pv_input"]["sha256"]
    assert manifest["random_seeds"]["multiseed"] == [0, 1, 2]
    assert manifest["configuration"]["site"]["tilt"] == 30.0


def test_repository_local_private_and_output_defaults_are_git_ignored() -> None:
    enforce_private_path(PACKAGE_ROOT / "private_data", "Private data")
    enforce_private_path(PACKAGE_ROOT / "camera_ready_outputs", "Outputs")
    with pytest.raises(RuntimeError, match="not Git-ignored"):
        enforce_private_path(PACKAGE_ROOT / "unignored_camera_ready", "Outputs")
