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

Place your `PV_data.csv` and `weather_cache.csv` in the repository root. Then run the one-command reproduce sequence:

```bash
python audit_data.py            # data-quality audit + missingness runs
python analyze_pv_v4.py         # main experiment (both protocols, both subsets)
python make_paper_figures.py    # model-comparison, residual-corr, representative-week
python make_results_figure.py   # compact main-text results figure
python make_appendix_figures.py # appendix figure set (skips absent inputs)
```

Behaviour is controlled by `config.py` and can be overridden through environment variables; the defaults are the leakage-safe policies. A provenance record of the active configuration is written to `pv_v4_run_config.json` next to the outputs.

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

## Missing PV measurements

Missing PV readings are marked before their numeric values are replaced. A
zero used for tensor construction is only a finite placeholder and is not
treated as observed zero generation. The CNN receives nine ordered input
channels: normalised historical PV, a past/target-day indicator, a historical
PV-availability mask, and six weather channels. Missing target hours have zero
loss weight and are excluded from both all-hours and daylight-only metrics.
Validation and test days are also masked wherever they would otherwise appear
inside a CNN history window. Temporal train/validation splits are made on whole
calendar days.

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

