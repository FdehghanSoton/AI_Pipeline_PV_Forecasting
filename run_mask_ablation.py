"""Ablation of how the CNN is told about missing PV observations.

A reviewer noted that filling missing values with zero injects information
that was never observed, and asked how the missing-value mask is actually
used. It is used in two independent places, and this script measures each.

``full``
    Both uses active, which is the configuration the paper reports. A binary
    availability channel tells the network which historical PV values were
    really observed, and hours with no observation are given zero weight in
    the training loss so they never produce a gradient.

``no_channel``
    The availability channel is removed. Filled zeros remain excluded from the
    loss, but on the input side the network can no longer tell a filled zero in
    the history from a real one.

``no_loss_weighting``
    The availability channel is kept, but imputed zeros are trained on as if
    they were observed.

``neither``
    Both switched off. This is the naive fill-with-zero approach the reviewer
    warns about, and is the arm the other three should be compared against.

Only the CNN's inputs change, so the honest place to read the effect is the
CNN row. The best ensemble is reported alongside it because the CNN is only
one of five base learners and stacking can absorb a weaker member.

Run ``python run_mask_ablation.py``. Output goes to
``pv_v4_mask_ablation.csv``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import pandas as pd

import analyze_pv_v4 as pipeline
import paths
from baselines import add_skill_columns
from config import RunConfig, load_config
from stats_tests import diebold_mariano

OUT_DIR = paths.results_dir()

MODE_LABEL = {"KFOLD": "Random day-fold", "TEMPORAL": "Rolling-origin"}


@dataclass(frozen=True)
class Arm:
    key: str
    label: str
    availability_channel: bool
    weight_missing_targets: bool


ARMS: tuple[Arm, ...] = (
    Arm("full", "Mask as channel and loss weight", True, False),
    Arm("no_channel", "Mask as loss weight only", False, False),
    Arm("no_loss_weighting", "Mask as channel only", True, True),
    Arm("neither", "No mask (naive fill with zero)", False, True),
)


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def run_arm(arm: Arm, base: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = replace(
        base,
        cnn_availability_channel=arm.availability_channel,
        cnn_weight_missing_targets=arm.weight_missing_targets,
        run_tag=f"mask_{arm.key}",
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
    metrics.insert(0, "arm", arm.key)

    predictions = pd.concat(
        [
            pipeline.folds_to_frame(temporal, "TEMPORAL"),
            pipeline.folds_to_frame(kfold, "KFOLD"),
        ],
        ignore_index=True,
    )
    predictions.insert(0, "arm", arm.key)
    return metrics, predictions


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    daylight = metrics[metrics["subset"] == "daylight"]
    for (arm, mode), group in daylight.groupby(["arm", "mode"]):
        indexed = group.set_index("model")
        ensembles = group[group["model"].isin(pipeline.ENSEMBLE_NAMES)]
        best = ensembles.sort_values("RMSE").iloc[0]
        rows.append(
            {
                "arm": arm,
                "mode": mode,
                "cnn_R2": float(indexed.loc["CNN", "R2"]),
                "cnn_RMSE": float(indexed.loc["CNN", "RMSE"]),
                "cnn_nRMSE_pct": float(indexed.loc["CNN", "nRMSE_pct"]),
                "best_ensemble": best["model"],
                "ensemble_R2": float(best["R2"]),
                "ensemble_nRMSE_pct": float(best["nRMSE_pct"]),
            }
        )
    return pd.DataFrame(rows)


def significance(predictions: pd.DataFrame) -> pd.DataFrame:
    """Test each reduced arm's CNN against the naive no-mask arm."""
    rows = []
    for mode in ("KFOLD", "TEMPORAL"):
        naive = _cnn_series(predictions, "neither", mode)
        if naive.empty:
            continue
        for arm in ARMS:
            if arm.key == "neither":
                continue
            variant = _cnn_series(predictions, arm.key, mode)
            joined = naive.join(
                variant[["cnn"]].rename(columns={"cnn": "variant"}), how="inner"
            )
            if joined.empty:
                continue
            test = diebold_mariano(
                joined["y_actual"].to_numpy(),
                joined["variant"].to_numpy(),
                joined["cnn"].to_numpy(),
                loss="squared",
                name_a=arm.key,
                name_b="neither",
            )
            rows.append({"arm": arm.key, "mode": mode, **test.as_dict()})
    return pd.DataFrame(rows)


def _cnn_series(predictions: pd.DataFrame, arm: str, mode: str) -> pd.DataFrame:
    subset = predictions[
        (predictions["arm"] == arm)
        & (predictions["mode"] == mode)
        & (predictions["is_missing"] == 0)
        & (predictions["is_daylight"] == 1)
    ]
    if subset.empty:
        return subset
    frame = subset[["timestamp", "y_actual", "CNN"]].rename(columns={"CNN": "cnn"})
    return frame.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()


def main() -> None:
    base = load_config()
    all_metrics = []
    all_predictions = []
    for arm in ARMS:
        _log(f"=== {arm.key}: {arm.label} ===")
        metrics, predictions = run_arm(arm, base)
        all_metrics.append(metrics)
        all_predictions.append(predictions)

    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    summary = summarise(metrics)

    label = {a.key: a.label for a in ARMS}
    order = {a.key: i for i, a in enumerate(ARMS)}
    summary["order"] = summary["arm"].map(order)
    summary = summary.sort_values(["mode", "order"]).drop(columns="order")

    # Cost of each reduction relative to the full configuration.
    for mode, group in summary.groupby("mode"):
        reference = group[group["arm"] == "full"]
        if reference.empty:
            continue
        cnn_reference = float(reference["cnn_nRMSE_pct"].iloc[0])
        summary.loc[summary["mode"] == mode, "cnn_nRMSE_change_pp"] = (
            summary.loc[summary["mode"] == mode, "cnn_nRMSE_pct"] - cnn_reference
        )

    summary.to_csv(OUT_DIR / "pv_v4_mask_ablation.csv", index=False)
    tests = significance(predictions)
    if not tests.empty:
        tests.to_csv(OUT_DIR / "pv_v4_mask_ablation_tests.csv", index=False)

    print("\n=== How the CNN is told about missing observations (daylight) ===")
    for mode in ("KFOLD", "TEMPORAL"):
        view = summary[summary["mode"] == mode].copy()
        if view.empty:
            continue
        view["configuration"] = view["arm"].map(label)
        print(f"\n--- {MODE_LABEL[mode]} ---")
        print(
            view[
                [
                    "configuration",
                    "cnn_R2",
                    "cnn_nRMSE_pct",
                    "cnn_nRMSE_change_pp",
                    "best_ensemble",
                    "ensemble_nRMSE_pct",
                ]
            ].to_string(index=False, float_format=lambda v: f"{v:.3f}")
        )

    if not tests.empty:
        print("\n=== CNN against the naive no-mask arm (Diebold-Mariano) ===")
        print(tests.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print("\nWrote pv_v4_mask_ablation.csv and pv_v4_mask_ablation_tests.csv")


if __name__ == "__main__":
    main()
