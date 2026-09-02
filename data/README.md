# Data

This directory holds the inputs to the pipeline. The weather is committed; the
PV measurements are not.

## `weather/` — committed

Hourly weather for the site, retrieved from the [Open-Meteo](https://open-meteo.com)
API. Each cache has a `.meta.json` recording the model, the requested and
returned grid point, the variable list, the date range and the retrieval time,
so any number in the paper can be traced to a specific product. Each
``.meta.json`` also stores a SHA-256 checksum of the committed CSV.

| File | Product | Used for |
| --- | --- | --- |
| `weather_cache_ifs.csv` | ECMWF IFS operational analysis (~9 km) | headline results |
| `weather_cache_era5.csv` | ERA5 reanalysis (31 km, 0.25° grid) | sensitivity check |
| `weather_cache_forecast_day1.csv` | forecast values at a constant 24-hour lead, not a single issuance | forecast-input sensitivity |
| `weather_cache.csv` | the original unpinned cache | kept for continuity |

Regenerate any of them with:

```bash
python weather_sources.py --source ifs        # or era5, forecast_day1
```

The forecast cache is the exception: it comes from Open-Meteo's previous-runs
API, which serves a rolling window, so the archived forecasts for the study
period cannot be re-fetched indefinitely. It is committed for that reason.

## `PV_data.csv` — not committed

Hourly AC power from the charging-station array. These are site
measurements that we are not able to redistribute, so the file is excluded by
`.gitignore` and the pipeline raises a clear error if it is missing. Enquiries
about access should go to the corresponding author.

The loader expects the long-format export produced by the site's InfluxDB
instance: one row per reading, with at least

| Column | Meaning |
| --- | --- |
| `_time` | ISO-8601 UTC timestamp |
| `_field` | measurement name; rows with `PPV` are used |
| `_value` | AC power in watts (inverter channel `PPV`) |

Lines beginning with `#` are ignored, so the annotated CSV that InfluxDB emits
can be used as-is. Any source giving one hourly power value per timestamp will
work if it is reshaped to these three columns.

The pipeline applies a one-hour backward shift by default (`PV_TIME_SHIFT=-1`).
That correction is specific to this site; check a new record with
`python scan_time_shift.py` before keeping the default.
