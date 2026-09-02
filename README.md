# An AI-Based Decision-Support Pipeline for Day-Ahead Photovoltaic Forecasting

Code and results for the UK AI Conference 2026 paper of the same name.

The pipeline forecasts hourly photovoltaic (PV) output a day ahead for a
charging-station site in Southampton, United Kingdom, using about a
year of inverter measurements and public weather data. It applies a one-hour
timestamp correction to an apparent offset between the two records, builds
leakage-safe solar-geometry and clearness-index features, adds short-term
atmospheric context, and fuses five structurally different predictors with
weights learned on validation rows only.

```text
        PV measurements                  Open-Meteo weather
              |                                  |
              +----------- timestamp ------------+
                           alignment
                              |
        solar geometry, clearness index, lag / lead / rolling context
                              |
    Ridge | GBM | POA-normalised GBM | per-hour GBM | 2D CNN
                              |
        mean | inverse RMSE | ridge stack | NNLS stack
                              |
        random day-fold and rolling-origin evaluation
```

## Layout

| Path | Contents |
| --- | --- |
| `*.py` | the pipeline: shared modules and the entry points listed below |
| `data/` | weather caches (committed) and the PV measurements (not committed) |
| `results/` | every table and figure in the paper, as generated |
| `tests/` | unit and regression tests |
| `legacy/` | superseded scripts, kept for provenance and not needed to reproduce |

Paths are resolved through `paths.py`, so scripts can be run from any working
directory and always read from `data/` and write to `results/`.

## Setup

Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The PV measurements are not distributed with this repository. Place your own
`PV_data.csv` in `data/`; `data/README.md` gives the expected format and what
to check before trusting the default timestamp correction. The weather caches
are already present, so nothing needs to be downloaded.

## Reproducing the paper

Everything below runs on a laptop CPU. The main experiment takes a few minutes;
the four credibility runs are each a full backtest and take longer.

```bash
python audit_data.py             # data-quality audit, missing runs, DST check
python analyze_pv_v4.py          # main experiment: both protocols, both subsets
python make_paper_figures.py     # model comparison, residual correlation, week
python make_results_figure.py    # main-text results figure
python make_appendix_figures.py  # appendix figures (skips absent inputs)
python make_weights_figure.py    # stacking weights
python make_pipeline_diagram.py  # pipeline diagram
```

The experiments that answer specific questions in the paper:

```bash
python run_ablations.py --both   # pipeline ablation table (Appendix B), both protocols
python run_multiseed.py --mode BOTH --seeds 0 1 2 3 4   # five-run robustness of the headline numbers
python run_weather_experiment.py # analysis vs a constant-24-hour-lead forecast product, and ERA5
python audit_clipping.py         # how often the clearness bounds actually bind
python run_clip_sensitivity.py   # sweep of the clearness-ratio bound
python run_mask_ablation.py      # what the CNN's missingness handling is worth
python run_cost_benefit.py       # cost of each base learner and its marginal value
python run_operational_value.py  # forecast error priced as a commitment cost
python run_standard_of_reference.py  # which naive forecast the skill score should use
```

The checks that answer "could this choice have been made without held-out
data?", which are cheap apart from the last:

```bash
python audit_baseline_gaps.py    # how often the reference forecasts fall back
python run_baseline_subset.py    # skill restricted to hours whose 24-hour lag exists
python run_bootstrap_blocks.py   # day, 3-day, and 7-day block bootstrap of the ensemble gain
python scan_shift_by_fold.py     # timestamp shift re-selected inside each fold
python check_stack_constraint.py # what forcing the stack weights to sum to one costs
python run_shift_sensitivity.py  # the pipeline run at each candidate timestamp shift
```

Each writes CSV files into `results/`, and `analyze_pv_v4.py` also writes
`pv_v4_run_config.json` recording the exact configuration that produced them.

Once the runs above have completed, `verify_paper_numbers.py` re-derives every
number quoted in the paper from the files in `results/` and reports any that no
longer agree, so a change in the pipeline cannot silently invalidate the text:

```bash
python verify_paper_numbers.py
```

### Which script produced which table

| Paper element | Script | Output |
| --- | --- | --- |
| Main results table | `analyze_pv_v4.py` | `pv_v4_metrics.csv` |
| Baseline fallback rates quoted in Section 5.2 | `audit_baseline_gaps.py` | `pv_v4_baseline_gaps.csv` |
| Skill on hours with a recorded 24-hour lag, Section 5.2 | `run_baseline_subset.py` | `pv_v4_baseline_subset.csv` |
| Choice of reference forecast, quoted in Section 5.2 | `run_standard_of_reference.py` | `pv_v4_standard_of_reference.csv` |
| Stacking-weight constraint check in Appendix A | `check_stack_constraint.py` | `pv_v4_stack_constraint.csv` |
| Appendix B, "Pipeline Ablation" | `run_ablations.py --both` and `run_shift_sensitivity.py` | `pv_v4_ablation.csv`, `pv_v4_shift_sensitivity.csv` |
| Appendix B, per-fold timestamp selection | `scan_shift_by_fold.py` | `pv_shift_by_fold.csv` |
| Appendix C, constant-24-hour-lead forecast-product sensitivity | `run_weather_experiment.py` | `pv_v4_weather_metrics.csv` |
| Appendix D, "Sensitivity to the POA-Normalised Target Bound" | `run_clip_sensitivity.py` | `pv_v4_clip_sensitivity.csv` |
| Appendix E, "Pricing Forecast Error as a Commitment Cost" | `run_operational_value.py` | `pv_v4_operational_value.csv` |
| Appendix F, "Computational Cost and Value of Each Base Learner" | `run_cost_benefit.py` | `pv_v4_cost_benefit.csv` |
| Appendix G, "Missing Data Handling in the Convolutional Network" | `run_mask_ablation.py` | `pv_v4_mask_ablation.csv` |
| Appendix H, model settings | (listed in the paper; source in `analyze_pv_v4.py` and `analyze_pv_cnn2d.py`) | --- |
| Timestamp-shift scan figure | `scan_time_shift.py` | `pv_time_shift_scan.csv` |
| Five-run robustness quoted in the results | `run_multiseed.py` | `pv_v4_multiseed_summary.csv` |
| Day-level and multi-day block bootstrap of the ensemble gain | `analyze_pv_v4.py` and `run_bootstrap_blocks.py` | `pv_v4_significance.csv`, `pv_v4_bootstrap_blocks.csv` |
| Every number in the paper, checked | `verify_paper_numbers.py` | console report |

## Configuration

`config.py` holds every switch, and each can be overridden by an environment
variable so a single command reproduces a specific run. The defaults are the
leakage-safe policies used for the headline numbers.

```bash
PV_TIME_SHIFT=0        PV_RUN_TAG=noshift    python analyze_pv_v4.py
PV_USE_PHYSICS=0       PV_RUN_TAG=calendar   python analyze_pv_v4.py
PV_WEATHER_SOURCE=forecast_day1 PV_RUN_TAG=fcst python analyze_pv_v4.py
PV_KAPPA_CLIP=inf      PV_RUN_TAG=noclip     python analyze_pv_v4.py
```

Outputs are suffixed with `PV_RUN_TAG`, so tagged runs do not overwrite the
headline results.

## Weather provenance

The headline results use the ECMWF IFS operational analysis archive. This is
retrospective: it describes the target day rather than predicting it, so it is
not available to a forecaster in advance. `run_weather_experiment.py` prices
that retrospective view by rerunning against the Open-Meteo previous-runs
product, which gives each hour as it was predicted 24 hours earlier. That is a
constant forecast lead assembled from successive model updates, not a single
forecast issuance, and the request is not pinned to the IFS, so the difference
mixes forecast lead with a change of weather product. Each cache in
`data/weather/` carries a `.meta.json` naming the model, grid point and
retrieval time.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

`tests/test_regression.py` checks the headline metrics against committed values
within the tolerances declared in `config.py`. It needs `data/PV_data.csv` and
is skipped when the measurements are absent.

## Data availability

The code, weather caches, and generated tables are in this repository. Each
weather cache records the request, the returned grid point, the retrieval
time, and a SHA-256 checksum of the committed file; see `data/README.md`.
The inverter measurements are site data that we are not able to redistribute;
enquiries about access should go to the corresponding author.

## Citation

Please cite the paper, the weather product used (ECMWF IFS via Open-Meteo for
the headline results), pvlib for the solar-position and clear-sky models, and
Open-Meteo itself.

## Acknowledgements

The authors acknowledge support from EPSRC through the Turing AI Fellowship
"Citizen-Centric AI Systems" (EP/V022067/1) and the "Future Electric Vehicle
Energy networks supporting Renewables (FEVER)" project (EP/W005883/1), the
IRIDIS High Performance Computing Facility, and the Low Carbon Comfort Centre
at the University of Southampton.
