"""Which naive forecast is the right standard of reference here?

Yang recommends that the standard of reference for day-ahead solar forecasting
be the most accurate naive method, so that skill scores are not exaggerated, and
argues specifically for the optimal convex combination of climatology and
persistence rather than for clear-sky (smart) persistence, which he notes is
best suited to short horizons.

This script builds that combination from the saved fold predictions and compares
it with the three naive references the paper already reports, so the choice of
denominator in the skill score rests on measurement rather than on convention.
The combination weight is fitted leave-one-fold-out: for each fold the weight is
chosen on the other folds and applied to the held-out one, so every scored hour
stays out of sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import model_labels
import paths

CAPACITY = 1664.0


def _rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err**2)))


def best_weight(df: pd.DataFrame) -> float:
    """Weight w minimising the RMSE of w*climatology + (1-w)*persistence."""
    grid = np.linspace(0.0, 1.0, 201)
    y = df["y_actual"].to_numpy()
    clim = df["Climatology"].to_numpy()
    pers = df["Persistence"].to_numpy()
    losses = [_rmse(y - (w * clim + (1 - w) * pers)) for w in grid]
    return float(grid[int(np.argmin(losses))])


def combination_predictions(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """Leave-one-fold-out convex combination of climatology and persistence."""
    preds, actuals, weights = [], [], []
    for fold, held in df.groupby("fold"):
        others = df[df["fold"] != fold]
        w = best_weight(others) if not others.empty else 0.5
        weights.append(w)
        pred = w * held["Climatology"].to_numpy() + (1 - w) * held["Persistence"].to_numpy()
        preds.append(np.clip(pred, 0.0, None))
        actuals.append(held["y_actual"].to_numpy())
    return np.concatenate(preds), np.concatenate(actuals), float(np.mean(weights))


rows = []
pred = pd.read_csv(paths.results_dir() / "pv_v4_predictions.csv")
daylight = pred[(pred["is_daylight"] == 1) & (pred["is_missing"] == 0)].copy()

for mode in ["KFOLD", "TEMPORAL"]:
    df = daylight[daylight["mode"] == mode]
    combo, y, mean_w = combination_predictions(df)
    entries = {
        "Persistence": df["Persistence"].to_numpy(),
        "Climatology": df["Climatology"].to_numpy(),
        "SmartPersistence": df["SmartPersistence"].to_numpy(),
        "ClimPersCombination": combo,
    }
    for name, p in entries.items():
        actual = y if name == "ClimPersCombination" else df["y_actual"].to_numpy()
        rmse = _rmse(actual - p)
        rows.append(
            {
                "mode": mode,
                "reference": name,
                "label": model_labels.label(name)
                if name != "ClimPersCombination"
                else "Climatology-persistence combination",
                "n": len(p),
                "RMSE": rmse,
                "nRMSE_pct": 100.0 * rmse / CAPACITY,
                "combination_weight_on_climatology": (
                    mean_w if name == "ClimPersCombination" else np.nan
                ),
            }
        )

table = pd.DataFrame(rows)
out = paths.results_dir() / "pv_v4_standard_of_reference.csv"
table.to_csv(out, index=False)

print("\n=== Candidate standards of reference, daylight subset ===")
for mode in ["KFOLD", "TEMPORAL"]:
    block = table[table["mode"] == mode].sort_values("nRMSE_pct")
    print(f"\n--- {'Random day-fold' if mode == 'KFOLD' else 'Rolling-origin'} ---")
    print(block[["label", "n", "RMSE", "nRMSE_pct"]].to_string(index=False, float_format="%.3f"))
    w = block["combination_weight_on_climatology"].dropna()
    if not w.empty:
        print(f"  mean weight on climatology: {w.iloc[0]:.3f}")
    best = block.iloc[0]
    print(f"  most accurate reference: {best['label']} at {best['nRMSE_pct']:.2f}% of capacity")

print(f"\nWrote {out.name}")
