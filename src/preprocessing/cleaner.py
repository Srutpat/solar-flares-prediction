"""Phase 3 preprocessing: chronological ordering and explicit duplicate policy."""
from __future__ import annotations

from copy import deepcopy
import numpy as np
import pandas as pd

from src.core.interfaces import Preprocessor
from src.core.logging_setup import TransformationLog


def _native_endian_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy FITS-derived numeric columns into native byte order before pandas indexing.

    Astropy exposes FITS numeric arrays in big-endian order. On little-endian
    Windows builds, pandas boolean row selection can raise
    ``Big-endian buffer not supported on little-endian compiler`` unless the
    underlying numeric buffers are normalized before any filtering.
    """
    out = frame.copy()
    for col in out.columns:
        try:
            values = out[col].to_numpy(copy=True)
            dtype = values.dtype
            if dtype.kind in "iufc" and not dtype.isnative:
                native_dtype = dtype.newbyteorder("=")
                # Explicit dtype conversion creates a native-endian owned array.
                native_values = np.array(values, dtype=native_dtype, copy=True)
                out[col] = pd.Series(native_values, index=out.index)
        except (TypeError, ValueError):
            # Leave non-numeric / extension columns unchanged.
            continue
    return out


class TimeSeriesPreprocessor(Preprocessor):
    PHASE = 3
    IMPLEMENTED = True

    def __init__(self, cfg=None, run_dir=None):
        self.cfg = cfg
        self.run_dir = run_dir

    def apply(self, ts, log=None):
        policy = getattr(self.cfg.preprocessing, "duplicate_timestamp_policy", "report_only") if self.cfg else "report_only"
        do_sort = getattr(self.cfg.preprocessing, "sort_chronologically", True) if self.cfg else True
        transform_log = log if isinstance(log, TransformationLog) else None
        output = {}
        for stream_id, stream in ts.items():
            # Normalize endian-sensitive FITS columns before sort_values or .loc[mask].
            df = _native_endian_frame(stream.dataframe)
            before = len(df)
            if do_sort:
                df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
            duplicates_before = int(df["timestamp"].duplicated(keep=False).sum())
            if policy == "keep_first":
                df = df.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
            elif policy == "keep_last":
                df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
            elif policy == "error" and duplicates_before:
                raise ValueError(f"{stream_id}: duplicate timestamps found and policy='error'")

            # Apply only the explicitly provided SoLEXS good-time intervals.
            # Never treat a finite value outside a declared GTI as valid data.
            gti_status = stream.metadata.get("gti_status")
            gti_intervals = stream.metadata.get("gti_intervals_utc", [])
            gti_rows_removed = 0
            if (getattr(stream, "instrument", "") or "").lower() in {"solexs", "hel1os"} and gti_status == "available":
                valid = pd.Series(False, index=df.index)
                stamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                for start_value, stop_value in gti_intervals:
                    start = pd.to_datetime(start_value, utc=True)
                    stop = pd.to_datetime(stop_value, utc=True)
                    valid |= stamps.between(start, stop, inclusive="both")
                gti_rows_removed = int((~valid).sum())
                df = df.loc[valid].reset_index(drop=True)
            # report_only deliberately preserves missing values; no imputation is
            # performed in Phase 3 because the correct treatment depends on the
            # verified cadence/gap structure and is a research decision.
            cleaned = deepcopy(stream)
            cleaned.dataframe = df
            output[stream_id] = cleaned
            if transform_log:
                transform_log.record(
                    "preprocessing", "chronological_sort_and_duplicate_policy",
                    rows_before=before, rows_after=len(df),
                    stream_id=stream_id, duplicate_policy=policy,
                    duplicates_detected=duplicates_before,
                    gti_status=stream.metadata.get("gti_status", "not_applicable"),
                    gti_rows_removed=gti_rows_removed,
                    gti_source=stream.metadata.get("gti_source"),
                    missing_values="preserved; no imputation in Phase 3")
        return output
