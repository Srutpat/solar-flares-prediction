"""Phase 5: causal forecast labels from validated event onset times.

Labels are created only from an external reference event catalogue. A row is
valid for horizon H only when its entire future interval lies inside one
continuous observed segment; GTI/data gaps therefore cannot silently become
negative examples.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.interfaces import LabelBuilder
from src.core.errors import DataUndetermined
from src.core.logging_setup import get_logger


class OnsetLabelBuilder(LabelBuilder):
    """For each horizon H, y_H(t)=1 iff onset is in (t, t+H].

    Gaps are detected from the observed timeline using a robust median cadence.
    Forecast rows whose future horizon crosses a gap or the observation tail are
    marked invalid (nullable label), never converted to negative labels.
    """
    PHASE = 5
    IMPLEMENTED = True

    def __init__(self, gap_factor: float = 3.0, minimum_gap_seconds: float = 5.0):
        if gap_factor <= 1:
            raise ValueError("gap_factor must be greater than 1")
        if minimum_gap_seconds <= 0:
            raise ValueError("minimum_gap_seconds must be positive")
        self.gap_factor = float(gap_factor)
        self.minimum_gap_seconds = float(minimum_gap_seconds)
        self.last_report: dict = {}

    def build_labels(self, timeline, catalogue, horizons_minutes):
        stamps = pd.to_datetime(pd.Series(timeline), utc=True, errors="coerce").dropna()
        stamps = stamps.drop_duplicates().sort_values().reset_index(drop=True)
        if stamps.empty:
            raise DataUndetermined("Cannot create labels: synchronized timeline is empty.")
        if "onset_utc" not in catalogue.columns:
            raise DataUndetermined("Cannot create labels: catalogue has no onset_utc column.")

        event_rows = catalogue
        if "within_observation_overlap" in catalogue.columns:
            event_rows = catalogue.loc[catalogue["within_observation_overlap"].astype(bool)]
        events = (pd.to_datetime(event_rows["onset_utc"], utc=True, errors="coerce")
                  .dropna().drop_duplicates().sort_values())

       # Keep timestamp and horizon arithmetic in the same nanosecond unit.
                # Keep timestamp and horizon arithmetic in the same nanosecond unit.
                # Keep timestamp and horizon arithmetic in the same nanosecond unit.
        stamp_ns = stamps.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        diffs_s = np.diff(stamp_ns) / 1e9
        positive_diffs = diffs_s[np.isfinite(diffs_s) & (diffs_s > 0)]
        cadence = float(np.median(positive_diffs)) if len(positive_diffs) else None
        gap_threshold = max(self.minimum_gap_seconds,
                            self.gap_factor * cadence if cadence is not None else self.minimum_gap_seconds)
        # A new segment starts immediately after a gap larger than the threshold.
        segment_ids = np.zeros(len(stamp_ns), dtype=np.int64)
        if len(diffs_s):
            segment_ids[1:] = np.cumsum(diffs_s > gap_threshold)
        segment_ends = {}
        for sid in np.unique(segment_ids):
            segment_ends[int(sid)] = int(np.flatnonzero(segment_ids == sid)[-1])

        out = pd.DataFrame({"timestamp": stamps, "segment_id": segment_ids})
        event_ns = events.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        for horizon in horizons_minutes:
            horizon = int(horizon)
            if horizon <= 0:
                raise ValueError(f"Forecast horizon must be positive; got {horizon}.")
            future_ns = stamp_ns + int(pd.Timedelta(minutes=horizon).value)
            end_indices = np.array([segment_ends[int(sid)] for sid in segment_ids], dtype=np.int64)
            valid = future_ns <= stamp_ns[end_indices]
            first_after_t = event_ns.searchsorted(stamp_ns, side="right")
            first_after_tH = event_ns.searchsorted(future_ns, side="right")
            positive = first_after_tH > first_after_t
            labels = pd.array(positive, dtype="Int64")
            labels[~valid] = pd.NA
            out[f"y_{horizon}m"] = labels
            out[f"valid_{horizon}m"] = valid

        self.last_report = {
            "timeline_rows": int(len(stamps)),
            "reference_events_used": int(len(events)),
            "median_cadence_seconds": cadence,
            "gap_threshold_seconds": float(gap_threshold),
            "continuous_segments": int(len(segment_ends)),
            "horizons_minutes": [int(h) for h in horizons_minutes],
        }
        get_logger("labels").info(
            "Built labels for %d timestamps, %d continuous segments, %d reference events and horizons %s.",
            len(out), len(segment_ends), len(events), horizons_minutes)
        return out
