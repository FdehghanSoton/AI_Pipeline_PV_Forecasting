"""Quantify how far the constant-lead forecast product is from the analysis.

Differences are reported on the daylight subset defined exactly as everywhere
else in the paper, by solar elevation above five degrees, because that is the
subset the headline metrics use and irradiance errors at night are not
informative. Cloud cover is additionally reported over all hours, since cloud
is defined at night too.

Run ``python audit_weather_difference.py``. Writes
``pv_v4_weather_difference.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

import paths
import weather_sources
from config import SiteConfig

VARIABLES = [
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "cloud_cover",
    "temperature_2m",
]


def load(source: str) -> pd.DataFrame:
    path = weather_sources.SOURCES[source].cache_path
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python weather_sources.py --source {source}`."
        )
    return pd.read_csv(path, parse_dates=["time"]).set_index("time")


def main() -> None:
    site = SiteConfig()
    analysis = load("ifs")
    forecast = load("forecast_day1")
    index = analysis.index.intersection(forecast.index)
    analysis, forecast = analysis.loc[index], forecast.loc[index]

    position = pvlib.solarposition.get_solarposition(
        index, site.lat, site.lon, altitude=site.alt
    )
    daylight = (position["elevation"] > 5.0).to_numpy()

    rows: list[dict[str, object]] = []
    for variable in VARIABLES:
        if variable not in analysis.columns or variable not in forecast.columns:
            continue
        for label, mask in (("daylight", daylight), ("all hours", np.ones_like(daylight))):
            a = analysis[variable].to_numpy()[mask.astype(bool)]
            f = forecast[variable].to_numpy()[mask.astype(bool)]
            rows.append(
                {
                    "variable": variable,
                    "subset": label,
                    "n": int(len(a)),
                    "analysis_mean": float(a.mean()),
                    "forecast_mean": float(f.mean()),
                    "mae": float(np.abs(a - f).mean()),
                    "rmse": float(np.sqrt(((a - f) ** 2).mean())),
                    "correlation": float(np.corrcoef(a, f)[0, 1]),
                }
            )

    table = pd.DataFrame(rows)
    table.to_csv(paths.results_dir() / "pv_v4_weather_difference.csv", index=False)

    print("\n=== Constant-lead forecast product against the analysis it replaces ===\n")
    print(table.to_string(index=False, float_format="%.2f"))
    print("\nWrote pv_v4_weather_difference.csv")


if __name__ == "__main__":
    main()
