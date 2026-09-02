"""Audit of the pipeline's 1.5 clipping bounds.

The bound is applied in three places that are not equivalent:

``clearness_kt``
    Feature: GHI / top-of-atmosphere irradiance. Stays near or below 1.

``kappa``
    POA-normalised GBM target: measured power / plane-of-array clear-sky
    power. Can become large at low sun elevation.

``smart_persistence_ratio``
    Same formula as ``kappa``, used by the smart-persistence baseline.

Run ``python audit_clipping.py``. Writes ``pv_v4_clipping_audit.csv`` and
``.json``.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd

import analyze_pv_v4 as pipeline
import paths
from config import load_config

OUT_DIR = paths.results_dir()

UPPER_BOUND = 1.5
DENOMINATOR_FLOOR_KW = 1.0
TOA_THRESHOLD = 50.0


def _describe(
    name: str,
    formula: str,
    raw: np.ndarray,
    bound: float,
    role: str,
) -> dict[str, object]:
    finite = raw[np.isfinite(raw)]
    n = int(finite.size)
    above = finite > bound
    return {
        "quantity": name,
        "formula": formula,
        "role": role,
        "upper_bound": bound,
        "n_daylight_observed": n,
        "n_above_bound": int(above.sum()),
        "pct_above_bound": round(100.0 * float(above.mean()) if n else float("nan"), 4),
        "max_raw": round(float(finite.max()), 4) if n else float("nan"),
        "p99_raw": round(float(np.percentile(finite, 99)), 4) if n else float("nan"),
        "median_raw": round(float(np.median(finite)), 4) if n else float("nan"),
        "bound_is_active": bool(above.any()),
    }


def audit() -> pd.DataFrame:
    cfg = replace(load_config(), weather_source="ifs", weather_variable_set="full")
    pv, _, capacity = pipeline.build_dataset(cfg)

    observed_daylight = pv["is_daylight"].astype(bool) & (pv["is_missing"] == 0)
    day = pv[observed_daylight]

    # Clearness index, before the clip in add_solar_features.
    toa = day["ghi_toa_horiz"].to_numpy()
    kt_raw = np.where(
        toa > TOA_THRESHOLD,
        day["shortwave_radiation"].to_numpy() / np.maximum(toa, 1.0),
        0.0,
    )

    # Plane-of-array clear-sky power envelope, floored as in the pipeline.
    envelope = np.maximum(
        capacity * day["poa_global"].to_numpy() / 1000.0, DENOMINATOR_FLOOR_KW
    )
    ratio_raw = day["y"].to_numpy() / envelope

    rows = [
        _describe(
            "clearness_kt",
            "GHI / top-of-atmosphere horizontal irradiance",
            kt_raw,
            UPPER_BOUND,
            "model feature",
        ),
        _describe(
            "kappa",
            "measured power / plane-of-array clear-sky power envelope",
            ratio_raw,
            UPPER_BOUND,
            "POA-normalised GBM target",
        ),
        _describe(
            "smart_persistence_ratio",
            "measured power / plane-of-array clear-sky power envelope",
            ratio_raw,
            UPPER_BOUND,
            "smart-persistence reference forecast",
        ),
    ]
    frame = pd.DataFrame(rows)

    # The reason the plane-of-array ratio explodes: a near-zero denominator.
    low_envelope = envelope <= DENOMINATOR_FLOOR_KW * 1.001
    context = {
        "capacity_w": round(float(capacity), 1),
        "denominator_floor_kw": DENOMINATOR_FLOOR_KW,
        "n_daylight_observed": int(observed_daylight.sum()),
        "pct_daylight_at_denominator_floor": round(
            100.0 * float(low_envelope.mean()), 4
        ),
        "pct_above_bound_explained_by_floor": round(
            100.0
            * float((low_envelope & (ratio_raw > UPPER_BOUND)).sum())
            / max(int((ratio_raw > UPPER_BOUND).sum()), 1),
            2,
        ),
        "max_clearness_kt_observed": round(float(kt_raw.max()), 4),
    }

    frame.to_csv(OUT_DIR / "pv_v4_clipping_audit.csv", index=False)
    (OUT_DIR / "pv_v4_clipping_audit.json").write_text(
        json.dumps({"context": context, "bounds": rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== Clipping-bound audit (daylight, observed rows only) ===")
    print(
        frame[
            [
                "quantity",
                "role",
                "n_above_bound",
                "pct_above_bound",
                "median_raw",
                "p99_raw",
                "max_raw",
                "bound_is_active",
            ]
        ].to_string(index=False)
    )
    print("\n=== Context ===")
    for key, value in context.items():
        print(f"  {key}: {value}")
    print(
        "\nWrote pv_v4_clipping_audit.csv and pv_v4_clipping_audit.json"
    )
    return frame


if __name__ == "__main__":
    audit()
