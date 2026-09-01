"""Re-select the PV timestamp shift inside each fold, using training days only.

``scan_time_shift.py`` sweeps the candidate shift over the whole record, which
means the selected value is informed by hours that later become test data. This
script repeats exactly the same parameter-free instrument (daylight R2 of the
physics proxy ``capacity * GHI / 1000``) but restricts every sweep to the
training days of a single fold, and takes the capacity from those days too. If
each fold selects the same shift as the full-record sweep, the alignment choice
can be reproduced without ever consulting held-out hours.

The first block reported is the initial 120-day training pool, which is never
scored under the rolling-origin protocol.

Run ``python scan_shift_by_fold.py``. Writes ``pv_shift_by_fold.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import analyze_pv_v4 as m
import paths
from analyze_pv_cnn2d import flag_daylight, load_pv, load_weather
from config import load_config

OUT_PATH = paths.results_dir() / "pv_shift_by_fold.csv"
SHIFTS = range(-3, 4)


def _r2_on_days(
    pv: pd.DataFrame, days: pd.DatetimeIndex, shift: int
) -> tuple[float, int]:
    """Daylight R2 of the physics proxy on ``days`` with the PV moved by ``shift``."""
    df = pv.copy()
    df["y"] = df["y"].shift(shift)
    df["is_missing"] = df["is_missing"].shift(shift).fillna(1).astype(int)
    df = df.dropna(subset=["y"])
    df = df[df.index.normalize().isin(days)]

    ok = df["is_daylight"].to_numpy().astype(bool) & (
        df["is_missing"].to_numpy() == 0
    )
    if ok.sum() < 100:
        return float("nan"), int(ok.sum())

    y = df["y"].to_numpy()
    capacity = float(np.quantile(y[df["is_missing"].to_numpy() == 0], 0.999))
    y_hat = capacity * df["shortwave_radiation"].to_numpy() / 1000.0
    return float(r2_score(y[ok], y_hat[ok])), int(ok.sum())


def _fold_training_days(cfg) -> list[tuple[str, pd.DatetimeIndex]]:
    """Training-day sets for every fold of both protocols, plus the initial pool."""
    pv, _, _ = m.build_dataset(cfg)
    days = pv.index.normalize().unique()
    blocks: list[tuple[str, pd.DatetimeIndex]] = [
        ("initial training pool", pd.DatetimeIndex(days[: cfg.first_test_days]))
    ]

    n_days = len(days)
    fold_len = (n_days - cfg.first_test_days) // cfg.temporal_n_folds
    for fold in range(cfg.temporal_n_folds):
        start = days[cfg.first_test_days + fold * fold_len]
        blocks.append((f"T{fold + 1}", pd.DatetimeIndex(days[days < start])))

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n_days)
    fold_size = n_days // cfg.kfold_n_folds
    for k in range(cfg.kfold_n_folds):
        hi = (k + 1) * fold_size if k < cfg.kfold_n_folds - 1 else n_days
        test_days = days[perm[k * fold_size : hi]]
        keep = pd.DatetimeIndex([d for d in days if d not in set(test_days)])
        blocks.append((f"K{k + 1}", keep))
    return blocks


def main() -> None:
    cfg = load_config()
    blocks = _fold_training_days(cfg)

    pv = flag_daylight(load_pv(paths.PV_CSV))
    pv = pv.join(load_weather()[["shortwave_radiation"]], how="left")
    pv["shortwave_radiation"] = pv["shortwave_radiation"].ffill().bfill()

    rows = []
    for label, days in blocks:
        scores = {s: _r2_on_days(pv, days, s) for s in SHIFTS}
        best = max(SHIFTS, key=lambda s: (-np.inf if np.isnan(scores[s][0])
                                          else scores[s][0]))
        row = {"block": label, "n_train_days": len(days), "selected_shift": best}
        row.update({f"R2_shift_{s:+d}": scores[s][0] for s in SHIFTS})
        rows.append(row)
        detail = "  ".join(f"{s:+d}h:{scores[s][0]:.3f}" for s in SHIFTS)
        print(f"{label:22s} n_days={len(days):4d}  best={best:+d}h   {detail}")

    table = pd.DataFrame(rows)
    table.to_csv(OUT_PATH, index=False)
    chosen = set(table["selected_shift"])
    print(f"\nDistinct shifts selected across blocks: {sorted(chosen)}")
    print(f"Wrote {OUT_PATH.name}")


if __name__ == "__main__":
    main()
