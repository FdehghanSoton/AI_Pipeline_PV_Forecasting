"""Weather retrieval with pinned, auditable provenance.

Three weather sources are supported, each written to its own cache file with a
sidecar metadata file recording exactly which Open-Meteo product produced it:

``ifs``
    ECMWF Integrated Forecasting System operational analysis archive
    (Open-Meteo ``models=ecmwf_ifs``, roughly 9 km). This is the product the
    paper's headline results use. Open-Meteo's historical default,
    ``best_match``, resolves to this same product at the study site, which is
    why the original unpinned cache is byte-compatible with it.

``era5``
    ERA5 reanalysis (Open-Meteo ``models=era5``, roughly 25 km). Retained for
    the weather-product sensitivity check only.

``forecast_day1``
    A constant 24-hour-lead profile from the Open-Meteo previous-model-runs
    archive via the ``_previous_day1`` suffix. The profile is assembled from
    successive model updates, not from one forecast issuance, and the request
    is not pinned to the IFS.

The model is always sent explicitly, never left to ``best_match``, so a rerun
returns the same product regardless of any future change to Open-Meteo's
default. Every fetch writes ``<cache>.meta.json``; the loader refuses to serve
a cache whose recorded source does not match the one requested.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

import paths

OUT_DIR = paths.weather_dir()

LAT = 50.9097
LON = -1.4044
SITE_NAME = "Southampton, UK"

# Full 15-variable panel used by the headline experiment.
WEATHER_VARS: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "shortwave_radiation",
    "direct_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "precipitation",
    "pressure_msl",
)

# The previous-runs archive does not serve the three sub-level cloud fields, so
# an operational run has twelve of the fifteen variables available. The
# matched-feature analysis run uses this same subset, which is what makes the
# analysis-versus-forecast comparison a fair one.
FORECAST_UNAVAILABLE_VARS: tuple[str, ...] = (
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
)
FORECAST_VARS: tuple[str, ...] = tuple(
    v for v in WEATHER_VARS if v not in FORECAST_UNAVAILABLE_VARS
)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"


@dataclass(frozen=True)
class WeatherSource:
    """Definition of one retrievable weather product."""

    key: str
    cache_name: str
    url: str
    model: str | None
    variables: tuple[str, ...]
    variable_suffix: str
    description: str

    @property
    def cache_path(self) -> Path:
        return OUT_DIR / self.cache_name

    @property
    def meta_path(self) -> Path:
        return OUT_DIR / (self.cache_name + ".meta.json")


SOURCES: dict[str, WeatherSource] = {
    "ifs": WeatherSource(
        key="ifs",
        cache_name="weather_cache_ifs.csv",
        url=ARCHIVE_URL,
        model="ecmwf_ifs",
        variables=WEATHER_VARS,
        variable_suffix="",
        description=(
            "ECMWF IFS operational analysis archive (~9 km) via the Open-Meteo "
            "historical-weather API"
        ),
    ),
    "era5": WeatherSource(
        key="era5",
        cache_name="weather_cache_era5.csv",
        url=ARCHIVE_URL,
        model="era5",
        variables=WEATHER_VARS,
        variable_suffix="",
        description=(
            "ERA5 reanalysis (~25 km) via the Open-Meteo historical-weather API"
        ),
    ),
    "forecast_day1": WeatherSource(
        key="forecast_day1",
        cache_name="weather_cache_forecast_day1.csv",
        url=PREVIOUS_RUNS_URL,
        model=None,
        variables=FORECAST_VARS,
        variable_suffix="_previous_day1",
        description=(
            "Constant 24-hour-lead profile from the Open-Meteo "
            "previous-model-runs archive (_previous_day1); not one issuance"
        ),
    ),
}


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def fetch(
    source_key: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch one weather source, caching it with provenance metadata.

    Columns are returned under their plain names (the ``_previous_day1``
    suffix is stripped) so that downstream feature construction is identical
    for every source.
    """
    source = SOURCES[source_key]
    requested = ",".join(v + source.variable_suffix for v in source.variables)

    if source.cache_path.exists() and not force_refresh:
        recorded = _recorded_source(source)
        if recorded == source_key:
            cached = _read_cache(source.cache_path)
            if _covers(cached, start, end):
                _log(f"{source.cache_name}: using cache ({len(cached)} rows)")
                return cached
        else:
            _log(
                f"{source.cache_name}: cached source {recorded!r} does not "
                f"match requested {source_key!r}; refetching"
            )

    params: dict[str, object] = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": requested,
        "timezone": "UTC",
    }
    if source.model is not None:
        params["models"] = source.model

    _log(
        f"querying {source.key} for {SITE_NAME}: "
        f"{params['start_date']} to {params['end_date']}"
    )
    response = requests.get(source.url, params=params, timeout=300)
    response.raise_for_status()
    payload = response.json()

    frame = pd.DataFrame(payload["hourly"])
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.set_index("time").sort_index()
    if source.variable_suffix:
        frame = frame.rename(
            columns={
                v + source.variable_suffix: v
                for v in source.variables
            }
        )
    frame = frame[list(source.variables)]

    frame.to_csv(source.cache_path, index_label="time")
    source.meta_path.write_text(
        json.dumps(
            {
                "source": source.key,
                "description": source.description,
                "open_meteo_url": source.url,
                "open_meteo_model": source.model or "previous_runs_day1",
                "variable_suffix": source.variable_suffix or None,
                "requested_latitude": LAT,
                "requested_longitude": LON,
                "grid_latitude": payload.get("latitude"),
                "grid_longitude": payload.get("longitude"),
                "grid_elevation_m": payload.get("elevation"),
                "start_date": params["start_date"],
                "end_date": params["end_date"],
                "variables": list(source.variables),
                "unavailable_variables": (
                    list(FORECAST_UNAVAILABLE_VARS)
                    if source.key == "forecast_day1"
                    else []
                ),
                "retrieved_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "n_rows": int(len(frame)),
                "n_missing_values": int(frame.isna().to_numpy().sum()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _log(
        f"{source.cache_name}: wrote {len(frame)} rows, "
        f"{int(frame.isna().to_numpy().sum())} missing values"
    )
    return frame


def _recorded_source(source: WeatherSource) -> str | None:
    if not source.meta_path.exists():
        return None
    try:
        return json.loads(source.meta_path.read_text()).get("source")
    except (OSError, ValueError):
        return None


def _read_cache(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["time"])
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return frame.set_index("time").sort_index()


def _covers(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    if frame.empty:
        return False
    return frame.index.min() <= start and frame.index.max() >= end


def main() -> None:
    """Fetch every source over the span of the PV record."""
    from analyze_pv_v2 import load_pv_v2

    pv = load_pv_v2(OUT_DIR / "PV_data.csv", time_shift_hours=-1)
    start = pv.index.min().normalize()
    end = pv.index.max().normalize()
    for key in SOURCES:
        fetch(key, start, end)


if __name__ == "__main__":
    main()
