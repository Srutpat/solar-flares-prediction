"""Pipeline stage registry and runner.

Phase 1 scope: declare the stages in order and report honestly which ones are
implemented. Orchestration of implemented stages is added phase by phase
(three-arm glue in Phase 9, full automation in Phase 13).
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Optional

from src.core.logging_setup import get_logger
from src.data_adapter.adapter import DataAdapterError, run_data_adapter
from src.validation.quality_checks import DataQualityValidator
from src.preprocessing.cleaner import TimeSeriesPreprocessor
from src.synchronization.synchronizer import CausalSynchronizer
from src.event_detection.catalogue import ReferenceEventCatalogueBuilder
from src.event_detection.labels import OnsetLabelBuilder
from src.feature_engineering.causal_features import CausalFeatureBuilder
from src.core.errors import DataUndetermined
from src.core.logging_setup import TransformationLog
import json


@dataclass(frozen=True)
class Stage:
    name: str
    phase: int
    module: str
    cls: str
    description: str


STAGES = (
    Stage("data_adapter", 2, "src.data_adapter.adapter", "ConfiguredDataAdapter", "Schema inspection + standardized time series"),
    Stage("validation", 3, "src.validation.quality_checks", "DataQualityValidator", "Quality checks and reports"),
    Stage("preprocessing", 3, "src.preprocessing.cleaner", "TimeSeriesPreprocessor", "Logged cleaning"),
    Stage("synchronization", 4, "src.synchronization.synchronizer", "CausalSynchronizer", "Causal alignment of instruments"),
    Stage("event_catalogue", 5, "src.event_detection.catalogue", "ReferenceEventCatalogueBuilder", "Event catalogue"),
    Stage("labels", 5, "src.event_detection.labels", "OnsetLabelBuilder", "Onset labels per horizon"),
    Stage("causal_features", 6, "src.feature_engineering.causal_features", "CausalFeatureBuilder", "Causal features + provenance"),
    Stage("dataset", 7, "src.dataset_builder.builder", "ForecastDatasetBuilder", "Forecast dataset"),
    Stage("split", 7, "src.dataset_builder.splits", "EventAwareChronologicalSplitter", "Chronological split"),
    Stage("xgboost", 8, "src.models.xgboost_model", "XGBoostForecaster", "Primary baseline model"),
    Stage("three_arm", 9, "src.experiments.three_arm", "ThreeArmComparison", "SoLEXS vs HEL1OS vs combined"),
    Stage("threshold", 10, "src.evaluation.threshold", "ValidationThresholdSelector", "Validation-only threshold"),
    Stage("evaluation", 10, "src.evaluation.metrics", "ForecastEvaluator", "Skill metrics"),
    Stage("lead_time", 10, "src.evaluation.lead_time", "EventLeadTimeAnalyzer", "Event-level lead time"),
    Stage("ablation", 11, "src.experiments.ablation", "FeatureAblationRunner", "Feature ablation"),
    Stage("interpretability", 11, "src.interpretability.explain", "XGBoostExplainer", "Importance / SHAP"),
    Stage("lstm", 12, "src.models.lstm_model", "LSTMForecaster", "Experimental sequence model"),
    Stage("reporting", 13, "src.reporting.report", "ResearchReportWriter", "Findings report"),
)


def stage_status() -> list:
    """[(Stage, implemented: bool)] - reads each component's IMPLEMENTED flag."""
    out = []
    for st in STAGES:
        cls = getattr(importlib.import_module(st.module), st.cls)
        out.append((st, bool(getattr(cls, "IMPLEMENTED", False))))
    return out


def format_status_table() -> str:
    rows = [f"{'stage':<18}{'phase':<7}status"]
    for st, ok in stage_status():
        rows.append(f"{st.name:<18}{st.phase:<7}{'implemented' if ok else 'placeholder (not implemented)'}")
    return "\n".join(rows)


def run_pipeline(cfg, run_dir) -> Optional[Stage]:
    """Run the stages implemented so far and stop at the first placeholder.

    Phase 2 implements only the data adapter. It runs when a source is configured
    and then the pipeline stops at the Phase-3 validation placeholder. If no real
    source is configured, the adapter reports that cleanly and the run stops at
    Phase 2 without inventing a dataset.
    """
    log = get_logger("pipeline")
    result = None
    streams = None
    quality = None
    catalogue = None
    for st, ok in stage_status():
        if not ok:
            log.warning("Stopping at stage '%s': placeholder, scheduled for Phase %d.", st.name, st.phase)
            return st
        if st.name == "data_adapter":
            try:
                result = run_data_adapter(cfg, run_dir)
            except DataAdapterError as exc:
                log.warning("Data adapter did not run: %s", exc)
                return st
            streams = result.streams
            log.info("Stage '%s' completed: %d files, %d standardized streams.",
                     st.name, len(result.files), len(result.streams))
        elif st.name == "validation":
            quality = DataQualityValidator().check(streams, TransformationLog(run_dir))
            (run_dir / "data_quality_report.json").write_text(
                json.dumps(quality.to_dict(), indent=2, default=str), encoding="utf-8")
            log.info("Stage '%s' completed: %d streams, %d rows, %d duplicate timestamp rows, %d large gaps.",
                     st.name, quality.summary["streams"], quality.summary["rows"],
                     quality.summary["duplicate_timestamp_rows"], quality.summary["large_gap_count"])
        elif st.name == "preprocessing":
            streams = TimeSeriesPreprocessor(cfg, run_dir).apply(streams, TransformationLog(run_dir))
            sample_dir = run_dir / "preprocessed_samples"
            sample_dir.mkdir(exist_ok=True)
            for stream_id, stream in streams.items():
                safe = __import__("re").sub(r"[^A-Za-z0-9_.-]+", "_", stream_id)
                stream.dataframe.head(1000).to_csv(sample_dir / f"{safe}.csv", index=False)
            log.info("Stage '%s' completed: %d standardized streams.", st.name, len(streams))
        elif st.name == "synchronization":
            synchronizer = CausalSynchronizer(cfg, run_dir)
            try:
                streams = synchronizer.synchronize(streams, TransformationLog(run_dir))
            except (ValueError, KeyError) as exc:
                log.warning("Synchronization did not run: %s", exc)
                return st
            report = synchronizer.report()
            (run_dir / "synchronization_report.json").write_text(
                json.dumps(report, indent=2, default=str), encoding="utf-8")
            sample_dir = run_dir / "synchronized_samples"
            sample_dir.mkdir(exist_ok=True)
            for stream_id, stream in streams.items():
                safe = __import__("re").sub(r"[^A-Za-z0-9_.-]+", "_", stream_id)
                stream.dataframe.head(1000).to_csv(sample_dir / f"{safe}.csv", index=False)
            log.info(
                "Stage '%s' completed: %d synchronized streams; overlap=%s to %s.",
                st.name, len(streams), report.get("overlap_start_utc"), report.get("overlap_end_utc")
            )
        elif st.name == "event_catalogue":
            try:
                catalogue = ReferenceEventCatalogueBuilder(cfg, cache_dir=run_dir).build(streams, TransformationLog(run_dir))
            except (DataUndetermined, ValueError, OSError) as exc:
                log.warning("Event catalogue did not run: %s", exc)
                return st
            catalogue.to_csv(run_dir / "reference_event_catalogue_validated.csv", index=False)
            summary = {
                "catalogue_mode": str(catalogue["catalogue_source_kind"].iloc[0]) if len(catalogue) else ("user_supplied_validated_csv" if cfg.labels.reference_catalogue_path else "NOAA_SWPC_daily_event_reports"),
                "reference_catalogue_path": cfg.labels.reference_catalogue_path,
                "events_total": int(len(catalogue)),
                "events_in_observation_overlap": int(catalogue["within_observation_overlap"].sum()),
                "events_outside_observation_overlap": int((~catalogue["within_observation_overlap"]).sum()),
                "observation_segments": int(catalogue["observation_segment_count"].iloc[0]) if len(catalogue) else 0,
                "class_counts": catalogue["class"].fillna("").replace("", "unspecified").value_counts().to_dict()
                    if "class" in catalogue.columns else {},
                "required_columns": ["onset_utc", "source"],
                "onset_rule": "NOAA/SWPC XRA Begin time; only exact unprefixed UTC begin times retained; no signal-spike inference.",
                "source_provenance": catalogue[["source"]].drop_duplicates().to_dict("records") if len(catalogue) else [],
            }
            (run_dir / "event_catalogue_report.json").write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8")
            log.info("Stage '%s' completed: %d catalogue events; %d in coverage.",
                     st.name, summary["events_total"], summary["events_in_observation_overlap"])
        elif st.name == "labels":
            try:
                timeline_stream = max(
                    (stream for stream in streams.values() if "timestamp" in stream.dataframe.columns),
                    key=lambda stream: len(stream.dataframe),
                )
                label_builder = OnsetLabelBuilder()
                labels = label_builder.build_labels(
                    timeline_stream.dataframe["timestamp"], catalogue, cfg.labels.horizons_minutes
                )
            except (DataUndetermined, ValueError, KeyError) as exc:
                log.warning("Label construction did not run: %s", exc)
                return st
            labels.to_csv(run_dir / "forecast_labels.csv", index=False)
            label_summary = {"rows": int(len(labels)), "horizons_minutes": cfg.labels.horizons_minutes,
                             "columns": list(labels.columns), "positive_counts": {}, "negative_counts": {},
                             "invalid_counts": {}, "valid_counts": {},
                             "timeline_quality": label_builder.last_report}
            for horizon in cfg.labels.horizons_minutes:
                y = labels[f"y_{horizon}m"]
                label_summary["positive_counts"][f"{horizon}m"] = int((y == 1).sum())
                label_summary["negative_counts"][f"{horizon}m"] = int((y == 0).sum())
                label_summary["invalid_counts"][f"{horizon}m"] = int(y.isna().sum())
                label_summary["valid_counts"][f"{horizon}m"] = int(labels[f"valid_{horizon}m"].sum())
            (run_dir / "label_report.json").write_text(
                json.dumps(label_summary, indent=2, default=str), encoding="utf-8")
            log.info("Stage '%s' completed: %d timestamps; labels written for horizons %s.",
                     st.name, len(labels), cfg.labels.horizons_minutes)
        elif st.name == "causal_features":
            try:
                feature_builder = CausalFeatureBuilder(cfg, run_dir)
                features, provenance = feature_builder.build(streams, TransformationLog(run_dir))
            except (DataUndetermined, ValueError, KeyError) as exc:
                log.warning("Causal feature construction did not run: %s", exc)
                return st
            features.to_csv(run_dir / "causal_features.csv", index=False)
            (run_dir / "feature_provenance.json").write_text(
                json.dumps(provenance, indent=2, default=str), encoding="utf-8")
            log.info("Stage '%s' completed: %d timestamps, %d features.",
                     st.name, len(features), provenance["feature_count"])
        else:
            log.info("Stage '%s' implemented; orchestration is added in a later phase.", st.name)
    return None
