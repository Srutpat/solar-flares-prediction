# Precursor-Based Solar Flare Forecasting — SoLEXS / HEL1OS (Aditya-L1)

A configuration-driven research framework for short-horizon solar flare-onset forecasting.
**Status: Phases 2–5 implemented in code. Real-data ingestion, quality checks, preprocessing, synchronization, reference-event retrieval and onset-label generation are implemented. Model training and forecasting evaluation are not yet implemented.**

## Research question
Do pre-flare signatures in synchronized soft-X-ray (SoLEXS) and hard-X-ray (HEL1OS) observations give
*measurable incremental predictive value* for short-horizon flare-onset forecasting?

Core experiment — three arms under identical scientific conditions: **A** SoLEXS-only, **B** HEL1OS-only,
**C** combined. The framework makes no claim that combining the instruments is novel, that fusion beats
single-instrument input, that LSTM beats XGBoost, or that any warning time is guaranteed.

## Pipeline
```text
DATA -> Phase 2 Data Adapter -> Validation -> Preprocessing -> Temporal Sync -> Event Catalogue/Labels
     -> Causal Features -> Forecast Dataset -> Chronological Split
     -> XGBoost (primary) / LSTM (experimental) -> Threshold (validation only)
     -> Evaluation -> Event-level Lead Time -> Three-arm comparison
     -> Ablation + Interpretability -> Report
```

Every stage has an interface in `src/core/interfaces.py`. Phases 2–5 implement real FITS ingestion, quality checks,
GTI-aware preprocessing, causal synchronization, reference-event retrieval, and gap-aware onset labels. Phases 6–13
remain staged for subsequent implementation.

## Phase 2: actual data support
The adapter is designed for the real supplied Aditya-L1 FITS products.

### SoLEXS
The adapter can inspect:
- Level-1 `.lc.gz` light curves
- Level-1 `.gti.gz` good-time files
- Level-1 `.pi.gz` spectral files (inventory/schema inspection; not automatically used as forecasting streams)
- extracted directories or the original `.zip` package

For SoLEXS light curves, the adapter recognizes the actual Level-1 time-series structure and standardizes the verified
`TIME`/measurement stream without selecting a final energy channel or feature. GTIs are reported but **not applied**
in Phase 2.

### HEL1OS
The adapter can inspect the supplied FITS directory/ZIP and discovers all light-curve extensions across the available
CZT/CdTe files. It does not choose one detector or energy band. The discovered measurement columns, extension names,
detector identifiers, units, cadence, coverage, missing values, and ranges are reported.

### What Phase 2 deliberately does not do
- no detector/band selection
- no feature engineering
- no resampling
- no synchronization between instruments
- no GTI filtering
- no NaN imputation or row dropping
- no flare detection or labels
- no model training

Detector/band selection and feature engineering remain deferred until the scientific analysis plan is fixed.
Phase 3 applies supplied GTIs; Phase 4 synchronizes the validated streams; Phase 5 creates reference labels.

## Installation
Python 3.10+ is required.

```powershell
pip install -r requirements.txt
```

or:

```powershell
pip install -e .
```

Phase 2 requires `astropy` for FITS/FITS.GZ handling.

## Running Phase 2
The data paths are operational inputs and are intentionally not part of the three-arm comparability hash.
The adapter accepts a FITS file, a directory, or a ZIP archive.

### SoLEXS-only
```powershell
python run_experiment.py --config configs/solexs.yaml --solexs-source "D:\path\AL1_SLX_L1_20260925_v1.0.zip"
```

### HEL1OS-only
```powershell
python run_experiment.py --config configs/helios.yaml --helios-source "D:\path\HLS_20260925_000013_43174sec_lev1_V111.zip"
```

### Combined
```powershell
python run_experiment.py --config configs/combined.yaml `
  --solexs-source "D:\path\AL1_SLX_L1_20260925_v1.0.zip" `
  --helios-source "D:\path\HLS_20260925_000013_43174sec_lev1_V111.zip"
```

The run stops after the Phase-2 adapter because Phase 3 validation is still a placeholder.

Each Phase-2 run creates:
```text
experiments/runs/<experiment-id>/
├── data_adapter_report.json
├── data_samples/             # first 1000 rows per discovered light-curve stream
├── config.resolved.yaml
├── manifest.json
├── run.log
└── transformations.jsonl
```

`data_adapter_report.json` is the main Phase-2 output. It records the discovered files, FITS extensions,
measurement candidates, timestamp representations, cadence, coverage, missing values, duplicate timestamps,
and numeric ranges. It also records items that remain intentionally unresolved.

## Standardized Phase-2 output
Each discovered light-curve stream is represented internally as a separate standardized time series with:
- `timestamp` — UTC timestamp
- original measurement column(s), unchanged in name/value
- instrument
- detector
- FITS extension
- source file
- original time representation metadata

Multiple streams remain separate. This prevents the adapter from silently deciding which detector, energy band,
or measurement is scientifically preferred.

## Configuration
`configs/base.yaml` contains shared settings. The three arm files differ in experiment name and input configuration.
Operational data locations may be supplied at runtime with `--solexs-source`, `--helios-source`, or `--data-source`.
These locations do not alter the scientific comparability hash.

## Reproducibility
- Experiment ID: `<name>-<UTC timestamp>-<8-char config hash>`
- Run directories are never overwritten.
- Each run records resolved configuration, hashes, seed, package versions, Git commit when available, UTC logs,
and transformation records.
- The adapter does not modify raw data.

## Tests
```powershell
python -m pytest -q
```

The automated suite covers adapter, validation, GTI preprocessing, synchronization, catalogue parsing, and label logic.
The actual FITS integration run requires Astropy to be installed in the environment executing the project.

## Roadmap
1. Framework foundation ✓  
2. FITS data adapter ✓  
3. Validation & preprocessing  
4. Temporal synchronization  
5. Event catalogue & labels  
6. Causal feature engineering  
7. Forecast dataset & chronological split  
8. XGBoost baseline  
9. Three-arm experiments  
10. Evaluation & event-level lead time  
11. Ablation & interpretability  
12. Experimental LSTM  
13. Reporting and automation

Each phase follows: **implement → test → run on real data → inspect output → confirm → continue.**

## Phase 5 — reference flare catalogue and forecast labels

Phase 5 imports an **externally validated** flare catalogue; it does not call arbitrary
peaks in SoLEXS/HEL1OS data flares. Supply a CSV with these required columns:

- `onset_utc`: reference flare onset timestamp, parseable as UTC (ISO-8601 recommended)
- `source`: catalogue name / reference identifier for traceability

Optional columns: `event_id`, `peak_utc`, `end_utc`, `class`, `notes`. A starter header is in
`data/reference_flare_catalogue_template.csv`. Populate it only with events from a reliable
catalogue and cite that catalogue in your research paper. Do not invent or extrapolate events.

Run with the added argument:

```powershell
python run_experiment.py --config configs/combined.yaml --solexs-source "D:\Downloads\AL1_SLX_L1_20260925_v1.0.zip" --helios-source "D:\Downloads\HLS_20260925_000013_43174sec_lev1_V111.zip" --reference-catalogue "D:\path\to\validated_flare_catalogue.csv"
```

Or set `labels.reference_catalogue_path` in the selected YAML. Label semantics for horizon H:
`y_H(t)=1` if a reference onset is in `(t, t+H]`. Rows lacking the full future horizon are
marked invalid (`y_H` is blank and `valid_H` is false), not incorrectly treated as negatives.
The run writes `reference_event_catalogue_validated.csv`, `event_catalogue_report.json`,
`forecast_labels.csv`, and `label_report.json`. Events outside the synchronized observation
coverage are retained in the imported catalogue but excluded from positive labels.

The 25 September 2026 data window is only about 12 hours. A successful Phase 5 run does not
imply that the window contains enough independent events to train or evaluate a reliable model.
Check event counts and coverage before proceeding to feature engineering/model training.


## Phase 3 update: GTI-aware preprocessing

The supplied SoLEXS Level-1 GTI tables use Unix-second START/STOP values; the supplied HEL1OS GTI tables use MJD `tstart`/`tstop` values. The adapter attaches the intervals to detector streams, merging the intervals from split HEL1OS archives. Phase 3 filters out rows outside declared GTIs before synchronization and records the removed-row count in `transformations.jsonl`. A missing GTI is reported rather than silently treated as an applied GTI.

## Phase 5: reference catalogue and forecast labels

When `labels.reference_catalogue_path` is not set, the framework retrieves the matching UTC daily edited solar event report from the NOAA/NCEI archive, with the NASA CCMC archive as a fallback. Raw report files are retained under `experiments/runs/<run-id>/noaa_swpc/` and their URLs/counts are recorded in the transformation log. Only XRA rows with an exact, unprefixed Begin time are used as onset events; rows marked before/after/uncertain are not converted into exact onset timestamps. No flare is inferred from arbitrary SoLEXS/HEL1OS count spikes.

The framework writes `reference_event_catalogue_validated.csv`, `event_catalogue_report.json`, `forecast_labels.csv`, and `label_report.json`. Labels are generated for 15, 30, 45 and 60 minutes using the rule `y_H(t)=1` iff a reference onset occurs in `(t, t+H]`. Rows without a complete future horizon are marked invalid. If the reference report contains no exact-time XRA events, the run records zero events rather than inventing positives.

### Important scope and scientific caution

The NOAA/SWPC edited daily event report is an independent reference source but is described by NOAA as preliminary event reports reviewed and edited by forecasters. It is not the same as the science-quality GOES XRS flare report. Preserve the exact source version and report URL for each experiment. Before making publication claims, compare the resulting onset records with an appropriate science-quality GOES catalogue when available and document any timing/class differences. The current dataset overlap is short and must not be treated as sufficient for model-performance claims without event-count and split-feasibility checks.

### Multiple HEL1OS archives

For split daily coverage, `--helios-source` accepts multiple archive paths in one command. Example for 17 October 2026:

```powershell
python run_experiment.py --config configs/combined.yaml `
  --solexs-source "D:\Downloads\AL1_SLX_L1_20261017_v1.0.zip" `
  --helios-source "D:\Downloads\HLS_20261017_000024_43165sec_lev1_V111.zip" "D:\Downloads\HLS_20261017_120003_43191sec_lev1_V111.zip"
```

## Running Phase 5 on the latest supplied observations (17 October 2026)

From the extracted project directory in PowerShell, run:

```powershell
python run_experiment.py --config configs/combined.yaml `
  --solexs-source "D:\Downloads\AL1_SLX_L1_20261017_v1.0.zip" `
  --helios-source "D:\Downloads\HLS_20261017_000024_43165sec_lev1_V111.zip" "D:\Downloads\HLS_20261017_120003_43191sec_lev1_V111.zip"
```

The Phase-5 implementation automatically retrieves the date-matched NOAA/SWPC edited solar event report when internet access is available; a hand-created CSV is not required. The run should write `reference_event_catalogue_validated.csv`, `event_catalogue_report.json`, `forecast_labels.csv`, and `label_report.json` before stopping at the Phase-6 placeholder. Inspect the run log and those reports. A zero-positive result is a valid possible outcome and must not be replaced with synthetic events.

## Phase 6 — causal feature engineering

Phase 6 builds a timestamp-aligned feature table from the standardized streams and writes
`causal_features.csv` plus `feature_provenance.json` in the run directory. It includes
per-channel levels, first differences per second, trailing rolling means/standard
deviations, and trailing SoLEXS–HEL1OS rolling Pearson correlations when both instruments
are present. Error/uncertainty/quality columns are excluded from signal predictors and
listed in the provenance report. Stream values are joined on exact timestamps; missing
values are preserved, never interpolated or forward-filled. Rolling statistics restart
at detected observation gaps.

The three arm configs currently share a 60-minute maximum history setting and 1-, 5-,
and 15-minute rolling windows. These are initial experiment settings and should be
reviewed before the final research runs; keep them identical across all three arms for
comparability. Phase 6 does not train a model. Phase 7 (dataset assembly and chronological
splitting) and later modelling/evaluation stages remain placeholders.
