"""Generate traceable camera-ready claims and tables from saved outputs.

No numerical result is hard-coded here.  The script reads the outputs of the
main, multi-seed, significance, fold-level, and ablation runs and writes one
machine-readable claims file plus CSV/LaTeX headline tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_paths import OUTPUT_DIR

BASE_LEARNERS = {"Ridge", "GBM", "GBM_kt", "GBM_per_hour", "CNN"}
ENSEMBLES = {"Mean", "InvRMSE", "RidgeStack", "NNLSStack", "BestSingleByVal"}
REFERENCE = "SmartPersistence"

REQUIRED_INPUTS = (
    "pv_v4_metrics.csv",
    "pv_v4_per_fold_metrics.csv",
    "pv_v4_significance.csv",
    "pv_v4_multiseed_raw__multiseed_kfold.csv",
    "pv_v4_multiseed_summary__multiseed_kfold.csv",
    "pv_v4_multiseed_raw__multiseed_temporal.csv",
    "pv_v4_multiseed_summary__multiseed_temporal.csv",
    "pv_v4_ablation.csv",
    "pv_time_shift_scan.csv",
)


def _read_required(output_dir: Path, filename: str) -> pd.DataFrame:
    path = output_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required camera-ready output is missing: {path}")
    return pd.read_csv(path)


def _finite_or_none(value: object) -> object:
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {str(key): _finite_or_none(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def derive_headline_metrics(
    metrics: pd.DataFrame, significance: pd.DataFrame
) -> pd.DataFrame:
    """Derive every protocol/subset headline comparison from saved metrics."""
    required = {
        "mode",
        "model",
        "subset",
        "R2",
        "MAE",
        "RMSE",
        "nMAE_pct",
        "nRMSE_pct",
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"pv_v4_metrics.csv lacks columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for (mode, subset), group in metrics.groupby(["mode", "subset"], sort=True):
        ensemble = group[group["model"].isin(ENSEMBLES)].sort_values("RMSE")
        base = group[group["model"].isin(BASE_LEARNERS)].sort_values("RMSE")
        reference = group[group["model"] == REFERENCE]
        if ensemble.empty or base.empty or reference.empty:
            raise ValueError(
                f"Missing ensemble, base learner, or {REFERENCE} for {mode}/{subset}"
            )
        best_ensemble = ensemble.iloc[0]
        best_base = base.iloc[0]
        smart = reference.iloc[0]
        ensemble_rmse = float(best_ensemble["RMSE"])
        base_rmse = float(best_base["RMSE"])
        smart_rmse = float(smart["RMSE"])

        dm_rows = significance[
            (significance["mode"] == mode)
            & (significance["subset"] == subset)
        ]
        if len(dm_rows) != 1:
            raise ValueError(
                f"Expected one Diebold-Mariano row for {mode}/{subset}, "
                f"found {len(dm_rows)}"
            )
        dm = dm_rows.iloc[0]
        if (
            dm["best_ensemble"] != best_ensemble["model"]
            or dm["best_base"] != best_base["model"]
        ):
            raise ValueError(
                f"Significance/model selection mismatch for {mode}/{subset}"
            )

        rows.append(
            {
                "mode": mode,
                "subset": subset,
                "best_ensemble": best_ensemble["model"],
                "ensemble_R2": float(best_ensemble["R2"]),
                "ensemble_MAE": float(best_ensemble["MAE"]),
                "ensemble_RMSE": ensemble_rmse,
                "ensemble_nMAE_pct": float(best_ensemble["nMAE_pct"]),
                "ensemble_nRMSE_pct": float(best_ensemble["nRMSE_pct"]),
                "smart_persistence_R2": float(smart["R2"]),
                "smart_persistence_MAE": float(smart["MAE"]),
                "smart_persistence_RMSE": smart_rmse,
                "smart_persistence_nMAE_pct": float(smart["nMAE_pct"]),
                "smart_persistence_nRMSE_pct": float(smart["nRMSE_pct"]),
                "improvement_vs_smart_persistence_pct": (
                    (smart_rmse - ensemble_rmse) / smart_rmse * 100.0
                ),
                "best_base": best_base["model"],
                "best_base_R2": float(best_base["R2"]),
                "best_base_MAE": float(best_base["MAE"]),
                "best_base_RMSE": base_rmse,
                "best_base_nMAE_pct": float(best_base["nMAE_pct"]),
                "best_base_nRMSE_pct": float(best_base["nRMSE_pct"]),
                "improvement_vs_best_base_pct": (
                    (base_rmse - ensemble_rmse) / base_rmse * 100.0
                ),
                "dm_statistic": float(dm["dm_statistic"]),
                "dm_p_value": float(dm["p_value"]),
                "dm_n": int(dm["n"]),
                "dm_better": dm["better"],
                "dm_loss": dm["loss"],
            }
        )
    return pd.DataFrame(rows).sort_values(["mode", "subset"]).reset_index(drop=True)


def _latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_")


def write_latex_table(headline: pd.DataFrame, path: Path) -> None:
    """Write a compact table whose values all come from ``headline``."""
    lines = [
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        (
            r"Protocol & Subset & Ensemble & $R^2$ & nRMSE (\%) & "
            r"Gain vs SP (\%) & Gain vs base (\%) \\"
        ),
        r"\midrule",
    ]
    for row in headline.itertuples(index=False):
        lines.append(
            f"{_latex_escape(row.mode)} & {_latex_escape(row.subset)} & "
            f"{_latex_escape(row.best_ensemble)} & {row.ensemble_R2:.3f} & "
            f"{row.ensemble_nRMSE_pct:.2f} & "
            f"{row.improvement_vs_smart_persistence_pct:.2f} & "
            f"{row.improvement_vs_best_base_pct:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_summary(output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    """Validate saved outputs and generate claims, CSV, and LaTeX artefacts."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {name: _read_required(output_dir, name) for name in REQUIRED_INPUTS}
    headline = derive_headline_metrics(
        frames["pv_v4_metrics.csv"], frames["pv_v4_significance.csv"]
    )
    headline.to_csv(output_dir / "camera_ready_headline_metrics.csv", index=False)
    write_latex_table(headline, output_dir / "camera_ready_results_table.tex")

    per_fold = frames["pv_v4_per_fold_metrics.csv"]
    claims: dict[str, object] = {
        "schema_version": 1,
        "derivation": (
            "Generated only from saved pipeline outputs; percentages are relative "
            "RMSE reductions and are not embedded constants."
        ),
        "headline_results": _records(headline),
        "fold_level_results": {
            "row_count": int(len(per_fold)),
            "folds_by_mode": {
                str(mode): sorted(group["fold"].astype(str).unique().tolist())
                for mode, group in per_fold.groupby("mode")
            },
            "records": _records(per_fold),
        },
        "multiseed": {
            "kfold_raw": _records(
                frames["pv_v4_multiseed_raw__multiseed_kfold.csv"]
            ),
            "kfold_summary": _records(
                frames["pv_v4_multiseed_summary__multiseed_kfold.csv"]
            ),
            "temporal_raw": _records(
                frames["pv_v4_multiseed_raw__multiseed_temporal.csv"]
            ),
            "temporal_summary": _records(
                frames["pv_v4_multiseed_summary__multiseed_temporal.csv"]
            ),
        },
        "ablation": _records(frames["pv_v4_ablation.csv"]),
        "alignment_scan": _records(frames["pv_time_shift_scan.csv"]),
    }
    (output_dir / "camera_ready_claims.json").write_text(
        json.dumps(claims, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote traceable camera-ready claims and tables to {output_dir}")
    return claims


if __name__ == "__main__":
    generate_summary()
