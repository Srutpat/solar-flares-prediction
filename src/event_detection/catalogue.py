"""Phase 5: retrieve/validate external flare events and construct onset labels.

No flare events are inferred from SoLEXS/HEL1OS signal spikes. By default, real
NOAA/SWPC daily edited solar-event reports are downloaded and parsed. A curated
CSV may override automatic retrieval for reproducibility or alternative catalogues.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.core.interfaces import EventCatalogueBuilder
from src.core.errors import DataUndetermined
from src.core.logging_setup import get_logger
from src.event_detection.noaa_catalogue import download_swpc_reports


class ReferenceEventCatalogueBuilder(EventCatalogueBuilder):
    """Load a documented reference catalogue; never invent event labels."""

    PHASE = 5
    IMPLEMENTED = True
    REQUIRED_COLUMNS = ("onset_utc", "source")
    OPTIONAL_COLUMNS = ("event_id", "peak_utc", "end_utc", "class", "notes")

    def __init__(self, cfg=None, cache_dir=None):
        self.cfg = cfg
        self.cache_dir = Path(cache_dir) if cache_dir else Path("experiments/runs/phase5_catalogue_cache")

    def build(self, ts, log):
        cfg = getattr(self, "cfg", None)
        path_value = getattr(getattr(cfg, "labels", None), "reference_catalogue_path", None)
        provenance = []
        path = None
        if path_value:
            path = Path(path_value).expanduser()
            if not path.is_file():
                raise DataUndetermined(f"Reference flare catalogue not found: {path}")
            try:
                catalogue = pd.read_csv(path)
            except Exception as exc:
                raise DataUndetermined(f"Could not read reference catalogue CSV '{path}': {exc}") from exc
            source_kind = "user_supplied_validated_csv"
        else:
            timeline = _timeline_from_streams(ts)
            if timeline.empty:
                raise DataUndetermined("No synchronized timeline is available to retrieve a date-matched catalogue.")
            start, end = timeline.min(), timeline.max()
            # No manual user catalogue is required: retrieve the real NOAA/SWPC
            # edited daily event reports covering the actual synchronized interval.
            try:
                catalogue, provenance = download_swpc_reports(start, end, self.cache_dir / "noaa_swpc")
            except Exception as exc:
                raise DataUndetermined(
                    "Could not automatically retrieve NOAA/SWPC daily event reports. "
                    "No synthetic labels were created. Check internet access, or provide a CSV via "
                    "--reference-catalogue. Original retrieval error: " + str(exc)
                ) from exc
            source_kind = "NOAA_SWPC_daily_event_reports"

        missing = [c for c in self.REQUIRED_COLUMNS if c not in catalogue.columns]
        if missing:
            raise DataUndetermined(
                f"Reference catalogue is missing required column(s): {missing}. "
                "Required columns are onset_utc and source."
            )
        timeline = _timeline_from_streams(ts)
        if timeline.empty:
            raise DataUndetermined("No synchronized timeline is available to compare against the catalogue.")
        start, end = timeline.min(), timeline.max()
        coverage_segments, cadence_seconds, gap_threshold_seconds = _coverage_segments(timeline)
        # A successfully retrieved report with zero exact-time XRA rows is a
        # legitimate zero-event result, not a reason to fabricate positives.
        catalogue = catalogue.copy()
        catalogue["onset_utc"] = pd.to_datetime(catalogue["onset_utc"], utc=True, errors="coerce")
        if catalogue["onset_utc"].isna().any():
            bad = catalogue.index[catalogue["onset_utc"].isna()].tolist()[:10]
            raise DataUndetermined(f"Invalid onset_utc value(s) at CSV row index(es): {bad}")
        for col in ("peak_utc", "end_utc"):
            if col in catalogue.columns:
                catalogue[col] = pd.to_datetime(catalogue[col], utc=True, errors="coerce")
        if catalogue["source"].isna().any() or catalogue["source"].astype(str).str.strip().eq("").any():
            raise DataUndetermined("Every catalogue event must identify its reference source in 'source'.")
        if "event_id" not in catalogue.columns:
            catalogue.insert(0, "event_id", [f"event_{i:05d}" for i in range(1, len(catalogue) + 1)])
        if catalogue["event_id"].duplicated().any():
            raise DataUndetermined("event_id values must be unique in the reference catalogue.")
        if "peak_utc" in catalogue.columns:
            bad = catalogue["peak_utc"].notna() & (catalogue["peak_utc"] < catalogue["onset_utc"])
            if bad.any():
                raise DataUndetermined("Catalogue has peak_utc earlier than onset_utc.")
        if "end_utc" in catalogue.columns:
            bad = catalogue["end_utc"].notna() & (catalogue["end_utc"] < catalogue["onset_utc"])
            if bad.any():
                raise DataUndetermined("Catalogue has end_utc earlier than onset_utc.")

        catalogue = catalogue.sort_values("onset_utc", kind="stable").reset_index(drop=True)
        # Keep the full validated input, but only treat onsets inside an actual
        # observed segment as label-eligible. Global min/max would incorrectly
        # include events falling inside GTI/data gaps.
        def onset_is_observed(onset):
            return any(seg_start <= onset <= seg_end for seg_start, seg_end in coverage_segments)
        catalogue["within_observation_overlap"] = catalogue["onset_utc"].map(onset_is_observed).astype(bool)
        catalogue["observation_segment_count"] = int(len(coverage_segments))
        catalogue["catalogue_path"] = str(path.resolve()) if path is not None else ""
        catalogue["catalogue_source_kind"] = source_kind
        if log is not None:
            log.record("event_catalogue", "load_reference_catalogue", len(catalogue), len(catalogue),
                       source_kind=source_kind, source_path=str(path.resolve()) if path is not None else "NOAA/SWPC auto-retrieval",
                       report_provenance=provenance, overlap_start_utc=start.isoformat(),
                       overlap_end_utc=end.isoformat(), observed_segments=[
                           {"start_utc": a.isoformat(), "end_utc": b.isoformat()} for a, b in coverage_segments
                       ], median_cadence_seconds=cadence_seconds, gap_threshold_seconds=gap_threshold_seconds,
                       events_in_overlap=int(catalogue["within_observation_overlap"].sum()),
                       events_outside_overlap=int((~catalogue["within_observation_overlap"]).sum()))
        get_logger("event_catalogue").info(
            "Loaded %d reference events from %s; %d onset(s) fall inside synchronized coverage.",
            len(catalogue), source_kind, int(catalogue["within_observation_overlap"].sum()))
        return catalogue


def _coverage_segments(timeline: pd.Series, gap_factor: float = 3.0,
                       minimum_gap_seconds: float = 5.0) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp]], float | None, float]:
    """Split an observation timeline at large gaps using its median cadence."""
    import numpy as np
    stamps = pd.to_datetime(pd.Series(timeline), utc=True, errors="coerce").dropna().drop_duplicates().sort_values()
    if stamps.empty:
        return [], None, float(minimum_gap_seconds)
    # Normalize datetime integers to nanoseconds before calculating gaps.
    ns = stamps.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    gaps = np.diff(ns) / 1e9
    positive = gaps[gaps > 0]
    cadence = float(np.median(positive)) if len(positive) else None
    threshold = max(minimum_gap_seconds, gap_factor * cadence if cadence is not None else minimum_gap_seconds)
    breaks = [0]
    breaks.extend((np.flatnonzero(gaps > threshold) + 1).tolist())
    breaks.append(len(stamps))
    segments = [(stamps.iloc[breaks[i]], stamps.iloc[breaks[i + 1] - 1])
                for i in range(len(breaks) - 1)]
    return segments, cadence, float(threshold)


def _timeline_from_streams(ts) -> pd.Series:
    if isinstance(ts, dict):
        candidates = []
        for stream in ts.values():
            frame = getattr(stream, "dataframe", None)
            if frame is not None and "timestamp" in frame.columns and not frame.empty:
                stamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
                if not stamps.empty:
                    candidates.append(stamps)
        if not candidates:
            return pd.Series([], dtype="datetime64[ns, UTC]")
        combined = pd.concat(candidates, ignore_index=True).drop_duplicates().sort_values()
        return combined.reset_index(drop=True)
    if isinstance(ts, pd.DataFrame) and "timestamp" in ts.columns:
        return pd.to_datetime(ts["timestamp"], utc=True, errors="coerce").dropna()
    raise DataUndetermined("Cannot extract timestamp timeline from synchronized streams.")
