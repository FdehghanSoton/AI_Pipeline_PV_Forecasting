"""Translate forecast error into a day-ahead commitment cost.

The proxy is day-ahead commitment under imbalance settlement, which is how a
site of this kind would actually monetise a forecast. The operator commits the
forecast for each hour of the target day. Where generation falls short of the
commitment the shortfall must be covered at a penalty; where it exceeds the
commitment the surplus is spilled or sold at a discount. Both are linear in
the deviation, so the cost is fully determined by the forecast, the outturn,
and one number: the ratio of shortfall to surplus penalty.

That ratio matters more than its absolute level. Under a symmetric penalty the
cost is proportional to MAE and the exercise says nothing new. Under an
asymmetric one, which is the realistic case, the cost-minimising commitment is
a quantile of the predictive distribution rather than its mean, and a model
trained to minimise squared error is systematically committing too much. We
therefore also measure how much of the penalty a single out-of-sample scaling
factor recovers, which bounds what the deterministic forecasts leave on the
table and quantifies the case for the probabilistic extension.

Costs are reported per unit of delivered energy with the surplus penalty set
to one, so no electricity price has to be assumed; a price only rescales every
row identically. Scaling factors are fitted leave-one-fold-out so that every
scored hour stays out of sample.

Run ``python run_operational_value.py`` with ``pv_v4_predictions.csv`` present.
Writes ``pv_v4_operational_value.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import model_labels
import paths

OUT_DIR = paths.results_dir()
PRED_PATH = OUT_DIR / "pv_v4_predictions.csv"

# Shortfall penalty relative to a surplus penalty of one. 1.0 is the
# degenerate symmetric case, kept as a reference point; 5.0 is roughly the
# spread between the two imbalance prices seen in tight UK settlement periods.
PENALTY_RATIOS = [1.0, 2.0, 3.0, 5.0]

MODELS = [
    "Persistence",
    "Climatology",
    "SmartPersistence",
    "Ridge",
    "GBM",
    "GBM_kt",
    "GBM_per_hour",
    "CNN",
    "RidgeStack",
    "NNLSStack",
]

PROTOCOLS = {"KFOLD": "Random day-fold", "TEMPORAL": "Rolling-origin"}


def imbalance_cost(
    actual: np.ndarray, commitment: np.ndarray, ratio: float
) -> float:
    """Cost per unit of delivered energy, with the surplus penalty set to one.

    A commitment is clipped at zero because an operator cannot promise negative
    generation, which also stops models with negative night-time predictions
    from being rewarded for them.
    """
    commitment = np.clip(commitment, 0.0, None)
    deviation = actual - commitment
    shortfall = np.clip(-deviation, 0.0, None)
    surplus = np.clip(deviation, 0.0, None)
    delivered = actual.sum()
    if delivered <= 0:
        return float("nan")
    return float((ratio * shortfall.sum() + surplus.sum()) / delivered)


def best_scale_out_of_sample(
    df: pd.DataFrame, model: str, ratio: float
) -> tuple[float, float]:
    """Cost after a leave-one-fold-out multiplicative commitment adjustment.

    Returns the cost and the mean scaling factor. The factor is chosen on a
    coarse grid rather than by an optimiser because the objective is piecewise
    linear in the scale and a grid is both adequate and transparent.
    """
    grid = np.arange(0.20, 1.51, 0.01)
    commitments: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    scales: list[float] = []
    for fold, held_out in df.groupby("fold"):
        others = df[df["fold"] != fold]
        if others.empty:
            continue
        a = others["y_actual"].to_numpy()
        p = others[model].to_numpy()
        costs = [imbalance_cost(a, s * p, ratio) for s in grid]
        scale = float(grid[int(np.nanargmin(costs))])
        scales.append(scale)
        commitments.append(scale * held_out[model].to_numpy())
        actuals.append(held_out["y_actual"].to_numpy())
    if not actuals:
        return float("nan"), float("nan")
    return (
        imbalance_cost(
            np.concatenate(actuals), np.concatenate(commitments), ratio
        ),
        float(np.mean(scales)),
    )


def main() -> None:
    preds = pd.read_csv(PRED_PATH)
    rows: list[dict[str, object]] = []

    for protocol, protocol_label in PROTOCOLS.items():
        # Commitment is made for every hour of the day, including the night,
        # because a scheduler consumes a complete 24-hour profile. Hours with
        # no measured outturn cannot be settled and are dropped.
        sub = preds[
            (preds["mode"] == protocol) & (preds["is_missing"] == 0)
        ].copy()
        if sub.empty:
            continue
        actual = sub["y_actual"].to_numpy()

        for model in MODELS:
            if model not in sub.columns:
                continue
            raw = sub[model].to_numpy()
            record: dict[str, object] = {
                "protocol": protocol_label,
                "model": model,
                "label": model_labels.label(model),
                "rmse": float(np.sqrt(np.mean((actual - raw) ** 2))),
                "mae": float(np.mean(np.abs(actual - raw))),
            }
            for ratio in PENALTY_RATIOS:
                cost = imbalance_cost(actual, raw, ratio)
                record[f"cost_r{ratio:g}"] = cost
                if ratio > 1.0:
                    tuned, scale = best_scale_out_of_sample(sub, model, ratio)
                    record[f"cost_r{ratio:g}_tuned"] = tuned
                    record[f"scale_r{ratio:g}"] = scale
                    record[f"saving_r{ratio:g}_pct"] = (
                        100.0 * (cost - tuned) / cost if cost > 0 else np.nan
                    )
            rows.append(record)

    table = pd.DataFrame(rows)

    # Express each cost as a saving against smart persistence, the reference an
    # operator would otherwise commit, so the columns read on the same scale as
    # the skill scores used elsewhere in the paper.
    for protocol_label in table["protocol"].unique():
        mask = table["protocol"] == protocol_label
        reference = table.loc[mask & (table["model"] == "SmartPersistence")]
        for ratio in PENALTY_RATIOS:
            column = f"cost_r{ratio:g}"
            base = float(reference[column].iloc[0])
            table.loc[mask, f"skill_r{ratio:g}"] = (
                1.0 - table.loc[mask, column] / base
            )

    table.to_csv(OUT_DIR / "pv_v4_operational_value.csv", index=False)

    print("\n=== Day-ahead commitment cost per unit of delivered energy ===")
    print("(surplus penalty = 1; shortfall penalty = r; lower is better)\n")
    for protocol_label in table["protocol"].unique():
        block = table[table["protocol"] == protocol_label]
        cols = (
            ["label", "rmse"]
            + [f"cost_r{r:g}" for r in PENALTY_RATIOS]
            + ["skill_r3"]
        )
        print(f"--- {protocol_label} ---")
        print(block[cols].to_string(index=False, float_format="%.3f"))
        print()

    print("=== Value left on the table by committing the mean (r = 3) ===\n")
    for protocol_label in table["protocol"].unique():
        block = table[table["protocol"] == protocol_label]
        cols = ["label", "cost_r3", "scale_r3", "cost_r3_tuned", "saving_r3_pct"]
        print(f"--- {protocol_label} ---")
        print(block[cols].to_string(index=False, float_format="%.3f"))
        print()

    print("Wrote pv_v4_operational_value.csv")


if __name__ == "__main__":
    main()
