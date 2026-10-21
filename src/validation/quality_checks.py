"""Phase 3 data-quality validation for standardized time-series streams."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.core.interfaces import QualityValidator


@dataclass
class StreamQualityReport:
    stream_id: str
    instrument: str | None
    detector: str | None
    rows: int
    timestamp_column: str
    timestamp_missing: int
    timestamp_duplicates: int
    timestamp_non_monotonic: int
    cadence: dict[str, Any]
    gaps: dict[str, Any]
    measurements: dict[str, dict[str, Any]]

    def to_dict(self):
        return asdict(self)


@dataclass
class QualityReport:
    streams: dict[str, StreamQualityReport]
    summary: dict[str, Any]
    warnings: list[str]

    def to_dict(self):
        return {
            "streams": {k: v.to_dict() for k, v in self.streams.items()},
            "summary": self.summary,
            "warnings": self.warnings,
        }


def _cadence(ts: pd.Series) -> dict[str, Any]:
    clean = pd.DatetimeIndex(ts.dropna())
    if len(clean) < 2:
        return {"n_intervals": 0, "median_seconds": None, "min_seconds": None,
                "max_seconds": None, "irregular_fraction": None}
    diffs = np.diff(clean.view("int64")) / 1e9
    finite = diffs[np.isfinite(diffs)]
    if not len(finite):
        return {"n_intervals": 0, "median_seconds": None, "min_seconds": None,
                "max_seconds": None, "irregular_fraction": None}
    med = float(np.median(finite))
    irregular = float(np.mean(np.abs(finite - med) > max(1e-9, med * 1e-6)))
    return {"n_intervals": int(len(finite)), "median_seconds": med,
            "min_seconds": float(np.min(finite)), "max_seconds": float(np.max(finite)),
            "irregular_fraction": irregular}


def _gap_summary(ts: pd.Series) -> dict[str, Any]:
    clean = pd.DatetimeIndex(ts.dropna())
    if len(clean) < 2:
        return {"count": 0, "threshold_seconds": None, "largest_seconds": None}
    diffs = np.diff(clean.view("int64")) / 1e9
    finite = diffs[np.isfinite(diffs)]
    if not len(finite):
        return {"count": 0, "threshold_seconds": None, "largest_seconds": None}
    med = float(np.median(finite))
    threshold = max(2.0 * med, med + 1e-9)
    gaps = finite > threshold
    return {"count": int(gaps.sum()), "threshold_seconds": threshold,
            "largest_seconds": float(np.max(finite)),
            "total_gap_seconds": float(np.sum(finite[gaps] - med)) if gaps.any() else 0.0}


def _measurement_summary(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    missing = int(numeric.isna().sum())
    raw = np.asarray(numeric, dtype=float)
    finite_mask = np.isfinite(raw)
    nonfinite = int((~finite_mask & ~np.isnan(raw)).sum())
    finite = raw[finite_mask]
    return {
        "dtype": str(series.dtype),
        "missing_or_non_numeric": missing,
        "non_finite": nonfinite,
        "finite_count": int(finite.size),
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
    }


class DataQualityValidator(QualityValidator):
    """Inspect standardized streams without changing their values."""

    PHASE = 3
    IMPLEMENTED = True

    def check(self, ts, log=None):
        reports = {}
        warnings = []
        total_rows = total_missing = total_duplicates = total_gaps = 0
        for stream_id, stream in ts.items():
            df = stream.dataframe
            timestamps = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            duplicate_count = int(timestamps.duplicated(keep=False).sum())
            non_monotonic = int((timestamps.dropna().diff().dt.total_seconds().dropna() < 0).sum())
            measurements = {c: _measurement_summary(df[c]) for c in stream.measurement_columns}
            missing = sum(v["missing_or_non_numeric"] for v in measurements.values())
            gaps = _gap_summary(timestamps)
            total_rows += len(df); total_missing += missing
            total_duplicates += duplicate_count; total_gaps += gaps["count"]
            if duplicate_count:
                warnings.append(f"{stream_id}: {duplicate_count} duplicated timestamp rows detected")
            if non_monotonic:
                warnings.append(f"{stream_id}: {non_monotonic} backward timestamp intervals detected")
            if gaps["count"]:
                warnings.append(f"{stream_id}: {gaps['count']} large time gaps detected")
            reports[stream_id] = StreamQualityReport(
                stream_id=stream_id, instrument=stream.instrument, detector=stream.detector,
                rows=len(df), timestamp_column="timestamp",
                timestamp_missing=int(timestamps.isna().sum()),
                timestamp_duplicates=duplicate_count,
                timestamp_non_monotonic=non_monotonic,
                cadence=_cadence(timestamps), gaps=gaps, measurements=measurements)
        summary = {"streams": len(reports), "rows": total_rows,
                   "measurement_missing_or_non_numeric": total_missing,
                   "duplicate_timestamp_rows": total_duplicates, "large_gap_count": total_gaps}
        return QualityReport(reports, summary, warnings)
