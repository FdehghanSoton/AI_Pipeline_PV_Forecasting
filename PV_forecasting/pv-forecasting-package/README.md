# A Physics-Aware AI Pipeline for Decision-Support in Day-Ahead Photovoltaic Forecasting


The supplied weather cache should be treated as an **unknown legacy Open-Meteo archive cache**, not automatically as ERA5. Open-Meteo's default historical product is `best_match`, which may combine several models. To run an ERA5-only experiment, regenerate the cache explicitly:

```bash
PV_WEATHER_MODEL=era5 python analyze_pv_with_weather.py
```

## Main workflow

```text
         input data
              |
              v
    timestamp alignment and data audit
              |
              v
 weather + solar-geometry + temporal features
              |
              v
 Ridge | GBM | clearness GBM | per-hour GBM | 2D CNN
              |
              v
 Mean | inverse RMSE | ridge stack | NNLS stack
              |
              v
 random day-fold and rolling-origin evaluation
```

## Quick start

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-forecasting.txt
```

Place these confidential inputs in the Git-ignored `private_data/` directory:

```text
private_data/PV_data.csv
private_data/weather_cache.csv
private_data/weather_cache.meta.json
```

Never commit or upload these source-data files. The repository ignore rules and
the camera-ready runner both enforce that boundary. Then regenerate every
result, credibility analysis, table, and figure with one command:

```bash
python run_camera_ready_pipeline.py
```

The runner creates a unique Git-ignored `camera_ready_outputs/run-.../`
directory. It refuses a dirty Git worktree, hashes the three private inputs,
logs every subprocess, validates every expected output, and writes an atomic
`run_manifest.json`. The manifest records the Git commit, seeds, data date
range, weather source, coordinates, assumed tilt/azimuth, daylight threshold,
reporting capacity, software versions, step timings, and output checksums.
Private absolute paths are not recorded.

Use `python run_camera_ready_pipeline.py --dry-run` to inspect the plan without
reading inputs or creating files. An interrupted run can be resumed with the
same commit, configuration, seeds, inputs, and explicit output directory:

```bash
python run_camera_ready_pipeline.py --output-dir camera_ready_outputs/run-YYYYMMDDTHHMMSSZ-COMMIT --resume
```

Behaviour is controlled by `config/config.py` and can be overridden through
environment variables; the defaults are the leakage-safe policies. The main
run's active configuration is also saved as `pv_v4_run_config.json`.

The main run writes:

- `pv_v4_metrics.csv` (includes a clear-sky-persistence skill column)
- `pv_v4_per_fold_metrics.csv` (per-fold spread)
- `pv_v4_predictions.csv` (includes `clearness_kt`)
- `pv_v4_per_month.csv`
- `pv_v4_residual_corr_kfold.csv`
- `pv_v4_residual_corr_temporal.csv`
- `pv_v4_significance.csv` (Diebold-Mariano ensemble vs best base learner)
- `pv_v4_run_config.json`
- `pv_v4_summary.png`
- paper-ready figures in PDF and PNG format
- `camera_ready_headline_metrics.csv`, `camera_ready_claims.json`, and
  `camera_ready_results_table.tex` (all headline values derived from saved outputs)


Additional credibility runs (each is a full backtest, so slower):

```bash
python run_ablations.py                              # pipeline ablation table
python run_multiseed.py --seeds 0 1 2 3 4            # multi-seed mean +/- sd
PV_DROP_MISSING_DAYS=1 PV_RUN_TAG=sens_nomissing python analyze_pv_v4.py  # missing-day sensitivity
PV_CAPACITY_POLICY=global PV_DAYLIGHT_POLICY=pv_median PV_RUN_TAG=legacy python analyze_pv_v4.py  # legacy policies
```

## File guide

### Main pipeline

- `analyze_pv_with_weather.py`: retrieves and records provenance for historical weather.
- `analyze_pv_v2.py`: PV loading, candidate timestamp shift, solar features, and ensemble utilities.
- `analyze_pv_v3.py`: full weather panel, temporal-context features, and shared plotting utilities.
- `analyze_pv_cnn2d.py`: leakage-aware CNN tensor construction and training.
- `analyze_pv_v4.py`: main random day-fold and rolling-origin experiment.
- `make_paper_figures.py`: generates figures directly from saved predictions and correlations.

### Supporting tools

- `audit_data.py`: records missingness, data range, empirical capacity, and weather-cache status.
- `test_time_shift.py`: compares candidate timestamp shifts empirically.
- `analyze_pv_diagnostic.py`: exploratory data-quality diagnostics.
- `analyze_pv_forecastability.py`: non-ML benchmark and forecastability analysis.

### Historical scripts

- `analyze_pv_ensemble.py`: superseded pre-V4 experiment.
- `analyze_pv_trends.py`: optional exploratory trend analysis with an additional dependency.

These historical scripts are not required to reproduce the paper's V4 results.


## Development checks

```bash
pip install -r requirements-dev.txt
ruff check .
python -m compileall -q .
pytest
```

## Citation

Please cite the final paper, the selected weather product, and pvlib. The Open-Meteo historical-weather documentation should also be recorded in the data-provenance statement.

