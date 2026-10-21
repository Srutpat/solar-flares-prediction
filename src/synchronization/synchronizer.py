"""Phase 4: causal temporal synchronization of SoLEXS and HEL1OS streams.

The implementation uses a backward as-of alignment when both instruments are
present. For each reference timestamp, only the most recent observation at or
before that timestamp is eligible. No interpolation or forward-looking value is
used. Single-instrument arms pass through unchanged because there is nothing to
synchronize.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

from src.core.errors import NotImplementedPhase
from src.core.interfaces import Synchronizer
from src.core.logging_setup import TransformationLog


@dataclass
class StreamSyncSummary:
    stream_id: str
    instrument: str | None
    rows_before: int
    rows_after: int
    matched_rows: int | None
    unmatched_rows: int | None
    max_observation_age_seconds: float | None
    median_observation_age_seconds: float | None

    def to_dict(self):
        return asdict(self)


@dataclass
class SynchronizationReport:
    method: str
    max_staleness_seconds: float | None
    reference_stream: str | None
    overlap_start_utc: str | None
    overlap_end_utc: str | None
    reference_rows: int
    streams: dict[str, StreamSyncSummary]
    notes: list[str]

    def to_dict(self):
        return {
            "method": self.method,
            "max_staleness_seconds": self.max_staleness_seconds,
            "reference_stream": self.reference_stream,
            "overlap_start_utc": self.overlap_start_utc,
            "overlap_end_utc": self.overlap_end_utc,
            "reference_rows": self.reference_rows,
            "streams": {k: v.to_dict() for k, v in self.streams.items()},
            "notes": self.notes,
        }


def _cadence_seconds(frame: pd.DataFrame) -> float | None:
    ts = pd.DatetimeIndex(frame["timestamp"].dropna())
    if len(ts) < 2:
        return None
    diffs = np.diff(ts.view("int64")) / 1e9
    diffs = diffs[np.isfinite(diffs)]
    return float(np.median(diffs)) if len(diffs) else None


def _native_endian_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose NumPy-backed numeric columns use native byte order.

    FITS stores numeric arrays in big-endian order. Some pandas operations (notably
    merge_asof on certain pandas/NumPy versions) reject those buffers on little-endian
    systems, so normalize every numeric column immediately after ingestion.
    """
    out = frame.copy()
    for col in out.columns:
        try:
            values = out[col].to_numpy(copy=True)
            dtype = values.dtype
            if dtype.kind in "iufc" and not dtype.isnative:
                native_dtype = dtype.newbyteorder("=")
                out[col] = pd.Series(values.astype(native_dtype, copy=True), index=out.index)
        except (TypeError, ValueError):
            # Non-numeric/object or extension columns are left unchanged.
            continue
    return out


def _instrument_streams(streams, instrument):
    return [s for s in streams.values() if (s.instrument or "").strip().upper() == instrument.upper()]


class CausalSynchronizer(Synchronizer):
    """Causally align all available streams to one common UTC reference timeline."""

    PHASE = 4
    IMPLEMENTED = True

    def __init__(self, cfg=None, run_dir=None):
        self.cfg = cfg
        self.run_dir = run_dir
        self.last_report: SynchronizationReport | None = None

    def synchronize(self, streams: dict, log=None):
        if not streams:
            raise ValueError("Phase 4 received no standardized streams.")

        # Normalize all FITS-derived numeric columns before any pandas merge or
        # selection. Converting only the right-hand merge frame is insufficient
        # on some pandas/NumPy versions because other copied frames may retain
        # non-native buffers.
        streams = {
            key: deepcopy(stream)
            for key, stream in streams.items()
        }
        for stream in streams.values():
            stream.dataframe = _native_endian_frame(stream.dataframe)

        sync_cfg = getattr(self.cfg, "synchronization", None) if self.cfg else None
        method = getattr(sync_cfg, "method", None) if sync_cfg else None
        max_staleness = getattr(sync_cfg, "max_staleness_seconds", None) if sync_cfg else None

        instruments = {str(s.instrument or "").strip().upper() for s in streams.values()}
        instruments.discard("")

        # A/B arms do not need cross-instrument synchronization. Preserve their
        # validated/preprocessed streams exactly, while still emitting a report.
        if len(instruments) < 2:
            output = {k: deepcopy(v) for k, v in streams.items()}
            summaries = {
                k: StreamSyncSummary(k, v.instrument, len(v.dataframe), len(v.dataframe),
                                     None, None, None, None)
                for k, v in output.items()
            }
            self.last_report = SynchronizationReport(
                method="single_instrument_passthrough",
                max_staleness_seconds=max_staleness,
                reference_stream=next(iter(output), None),
                overlap_start_utc=None, overlap_end_utc=None,
                reference_rows=len(next(iter(output.values())).dataframe) if output else 0,
                streams=summaries,
                notes=["Only one instrument is present; no cross-instrument alignment was required."],
            )
            return output

        if method != "asof_backward":
            raise ValueError(
                "Phase 4 currently implements only synchronization.method='asof_backward'. "
                "Set it explicitly after dataset inspection; no forward-looking method is allowed."
            )
        if max_staleness is None or float(max_staleness) <= 0:
            raise ValueError("Phase 4 requires synchronization.max_staleness_seconds > 0 for causal alignment.")

        solexs = _instrument_streams(streams, "SoLEXS")
        helios = _instrument_streams(streams, "HEL1OS")
        if not solexs or not helios:
            raise ValueError("Combined synchronization requires at least one SoLEXS and one HEL1OS stream.")

        # Use the longest SoLEXS stream as the common reference. The current
        # inspected archive contains one SoLEXS light curve, so this is a
        # deterministic general rule without selecting a HEL1OS energy band.
        reference = max(solexs, key=lambda s: len(s.dataframe))
        ref_df = reference.dataframe.copy()
        ref_df["timestamp"] = pd.to_datetime(ref_df["timestamp"], utc=True)
        ref_df = ref_df.sort_values("timestamp", kind="stable").drop_duplicates("timestamp", keep="last")

        # The reference timeline is already GTI-filtered by Phase 3. Do not
        # calculate one global intersection across every input stream: separate
        # HEL1OS ZIPs can represent consecutive time segments (e.g. 00:00-12:00
        # and 12:00-24:00) and are currently represented as separate streams.
        # Intersecting *all* stream ranges incorrectly makes the overlap empty
        # when a later segment does not overlap the SoLEXS observing window.
        ref_start = ref_df["timestamp"].min()
        ref_end = ref_df["timestamp"].max()
        if pd.isna(ref_start) or pd.isna(ref_end):
            raise ValueError("The GTI-filtered SoLEXS reference stream has no valid timestamps.")

        active_streams = {}
        skipped_nonoverlap = []
        for stream_id, stream in streams.items():
            if stream_id == reference.stream_id:
                continue
            stamps = pd.to_datetime(stream.dataframe["timestamp"], utc=True, errors="coerce").dropna()
            if stamps.empty:
                skipped_nonoverlap.append(stream_id)
                continue
            stream_start, stream_end = stamps.min(), stamps.max()
            if stream_end < ref_start or stream_start > ref_end:
                skipped_nonoverlap.append(stream_id)
                continue
            active_streams[stream_id] = stream

        active_helios = {
            stream_id: stream for stream_id, stream in active_streams.items()
            if (stream.instrument or "").strip().upper() == "HEL1OS"
        }
        if not active_helios:
            coverages = []
            for stream_id, stream in streams.items():
                if (stream.instrument or "").strip().upper() != "HEL1OS":
                    continue
                stamps = pd.to_datetime(stream.dataframe["timestamp"], utc=True, errors="coerce").dropna()
                if not stamps.empty:
                    coverages.append(f"{stream_id}=[{stamps.min().isoformat()}, {stamps.max().isoformat()}]")
            raise ValueError(
                "No HEL1OS stream overlaps the GTI-filtered SoLEXS reference interval "
                f"[{ref_start.isoformat()}, {ref_end.isoformat()}]. HEL1OS coverage: "
                + ("; ".join(coverages) if coverages else "no valid HEL1OS timestamps")
            )

        # Report the envelope of the SoLEXS interval and the HEL1OS segments
        # that actually intersect it. Individual rows are still matched only
        # backward, within the configured staleness tolerance.
        active_starts = [pd.to_datetime(s.dataframe["timestamp"], utc=True).min()
                         for s in active_helios.values()]
        active_ends = [pd.to_datetime(s.dataframe["timestamp"], utc=True).max()
                       for s in active_helios.values()]
        overlap_start = max(ref_start, min(active_starts))
        overlap_end = min(ref_end, max(active_ends))
        if overlap_start > overlap_end:
            raise ValueError(
                "No overlapping observation interval remains after considering HEL1OS segments "
                "that intersect the GTI-filtered SoLEXS timeline."
            )

        ref_df = ref_df[(ref_df["timestamp"] >= overlap_start) & (ref_df["timestamp"] <= overlap_end)].copy()
        ref_df = ref_df.reset_index(drop=True)
        if ref_df.empty:
            raise ValueError("The common observation envelope contains no valid SoLEXS reference rows.")

        output = {}
        summaries = {}
        output_ref = deepcopy(reference)
        output_ref.dataframe = ref_df
        output_ref.metadata = dict(output_ref.metadata)
        output_ref.metadata.update({
            "phase4_role": "reference_timeline",
            "synchronization_method": method,
            "overlap_start_utc": overlap_start.isoformat(),
            "overlap_end_utc": overlap_end.isoformat(),
        })
        output[reference.stream_id] = output_ref
        summaries[reference.stream_id] = StreamSyncSummary(
            reference.stream_id, reference.instrument, len(reference.dataframe), len(ref_df),
            len(ref_df), 0, 0.0, 0.0
        )

        # Causal as-of matching: a HEL1OS observation is usable only if its
        # timestamp is <= the reference timestamp and is no older than the
        # configured tolerance. This avoids interpolation and look-ahead.
        left = pd.DataFrame({"timestamp": ref_df["timestamp"]})
        for stream_id, stream in active_streams.items():
            right = stream.dataframe.copy()
            right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True)
            right = right.sort_values("timestamp", kind="stable").drop_duplicates("timestamp", keep="last")
            right = right[(right["timestamp"] >= overlap_start - pd.Timedelta(seconds=float(max_staleness))) &
                          (right["timestamp"] <= overlap_end)]
            if right.empty:
                raise ValueError(f"{stream_id}: no observations available within the overlap interval.")

            cols = list(stream.measurement_columns)
            right_use = right[["timestamp"] + cols].copy()
            merged = pd.merge_asof(
                left, right_use, on="timestamp", direction="backward",
                tolerance=pd.Timedelta(seconds=float(max_staleness)),
                allow_exact_matches=True,
            )
            matched = merged[cols].notna().any(axis=1)
            # Observation age is computed from the matched source timestamp.
            right_age = right[["timestamp"]].rename(columns={"timestamp": "source_timestamp"})
            age = pd.merge_asof(
                left, right_age, left_on="timestamp", right_on="source_timestamp",
                direction="backward", tolerance=pd.Timedelta(seconds=float(max_staleness)),
                allow_exact_matches=True,
            )
            age_seconds = (age["timestamp"] - age["source_timestamp"]).dt.total_seconds()
            valid_age = age_seconds[age_seconds.notna()]

            cleaned = deepcopy(stream)
            cleaned.dataframe = merged
            cleaned.metadata = dict(cleaned.metadata)
            cleaned.metadata.update({
                "phase4_role": "causally_aligned",
                "synchronization_method": method,
                "reference_stream": reference.stream_id,
                "max_staleness_seconds": float(max_staleness),
                "matched_rows": int(matched.sum()),
                "unmatched_rows": int((~matched).sum()),
            })
            output[stream_id] = cleaned
            summaries[stream_id] = StreamSyncSummary(
                stream_id, stream.instrument, len(stream.dataframe), len(merged),
                int(matched.sum()), int((~matched).sum()),
                float(valid_age.max()) if len(valid_age) else None,
                float(valid_age.median()) if len(valid_age) else None,
            )

        self.last_report = SynchronizationReport(
            method=method,
            max_staleness_seconds=float(max_staleness),
            reference_stream=reference.stream_id,
            overlap_start_utc=overlap_start.isoformat(),
            overlap_end_utc=overlap_end.isoformat(),
            reference_rows=len(ref_df),
            streams=summaries,
            notes=[
                "Reference timeline selected from the longest GTI-filtered SoLEXS stream.",
                "HEL1OS observations are matched backward only; no future observation can populate an earlier timestamp.",
                "No interpolation is performed.",
                "No energy band or detector is selected by Phase 4.",
                f"Skipped {len(skipped_nonoverlap)} stream(s) whose time coverage did not overlap the SoLEXS reference window.",
                *( ["Skipped stream IDs: " + ", ".join(skipped_nonoverlap)] if skipped_nonoverlap else [] ),
            ],
        )
        return output

    def report(self) -> dict[str, Any]:
        if self.last_report is None:
            return {}
        return self.last_report.to_dict()
