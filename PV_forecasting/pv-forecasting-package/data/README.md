# Data files

The forecasting scripts expect the following files in the repository root when run locally:

- `PV_data.csv`: site-specific hourly inverter active-power data with `_time`, `_field`, and `_value` columns.
- `weather_cache.csv`: hourly historical weather retrieved by `analyze_pv_with_weather.py`.
- `weather_cache.meta.json`: provenance for the weather cache, including the requested Open-Meteo model.

These files are excluded from the public GitHub package because the PV measurements are site-specific and the weather cache can be regenerated.

For an ERA5-only cache, run:

```bash
PV_WEATHER_MODEL=era5 python analyze_pv_with_weather.py
```

Delete an older cache first, or call `fetch_weather(..., force_refresh=True)` from Python. Do not describe a legacy cache as ERA5 unless its metadata confirms `weather_model: era5`.

## Derived quantities and policies

- Capacity: a full-series 99.9th-percentile value is used only as a fixed normalised-metric denominator. The capacity entering the clearness target and CNN normalisation is estimated per fold from training rows (`config.RunConfig.capacity_policy`). Replace with verified nameplate capacity when available.
- Daylight: defined from solar elevation (`config.RunConfig.daylight_policy = "geometric"`), not from observed PV, so the evaluation subset is leakage-free.
- Site geometry (latitude, longitude, altitude, tilt, azimuth) lives in `config.SiteConfig`. Tilt and azimuth are modelling assumptions until confirmed from the installation specification.
