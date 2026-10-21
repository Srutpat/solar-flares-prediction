# Preliminary Results — Real-Data Ingestion and Synchronization

## Scope and status

This preliminary result reports the successful Phase 2–4 integration run completed on 25 September 2026 SoLEXS and HEL1OS Level-1 observations. It evaluates whether the real instrument products can be ingested, checked, preprocessed, and temporally aligned. It does **not** report flare-forecast skill, model accuracy, or evidence that multi-instrument fusion improves prediction.

## Observed pipeline results

| Measure | Observed result |
|---|---:|
| FITS files discovered | 18 |
| Standardized light-curve streams | 21 |
| Rows processed across streams | 949,251 |
| Duplicate timestamp rows reported by validation | 0 |
| Large cadence gaps reported by generic validation | 0 |
| Streams reaching synchronization | 21 |
| Common synchronized interval start (UTC) | 2026-09-25 00:01:30.498423 |
| Common synchronized interval end (UTC) | 2026-09-25 11:57:08.431471 |
| Common interval duration | 11 h 55 min 37.9 s |

The above values are copied from the successful experiment log for run `combined-20261019T174051Z-6ded5acf`. The zero-gap value refers to the generic cadence-gap validator's configured criterion; it should not be interpreted as proof that every timestamp is continuously observed or that all instrument good-time intervals are gap-free. GTI-aware preprocessing and label-time gap checks are treated separately.

## Paper-ready draft

The initial real-data integration experiment demonstrated that the ingestion and temporal-alignment pipeline could process 18 FITS files into 21 standardized SoLEXS/HEL1OS light-curve streams, comprising 949,251 rows across the discovered streams. The validation stage reported no duplicate timestamp rows and no large cadence gaps under its configured criterion. The synchronization stage completed for all 21 streams over a common interval from 00:01:30.498423 to 11:57:08.431471 UTC on 25 September 2026, corresponding to approximately 11 h 55 min 38 s. These results establish operational feasibility for the data-integration stage, rather than predictive performance. The next evaluation step is to associate the observed intervals with independently reported solar-flare onsets and quantify the resulting valid positive and negative labels at each forecast horizon.

## Important limitation

Phase 5 now implements reference-catalogue retrieval/validation and horizon-label generation, including exclusion of forecast horizons that cross observation gaps. However, the current report above predates the completed Phase 5 implementation and therefore contains no observed event counts or label counts. Those must be obtained by running the updated framework against the actual data archives and recording the downloaded NOAA/SWPC report provenance. NOAA describes these edited event reports as preliminary reports manually reviewed by forecasters; any paper should disclose that reference-label limitation and, where possible, cross-check against a science-quality GOES X-ray flare catalogue.
