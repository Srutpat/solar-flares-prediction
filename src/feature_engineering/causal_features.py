"""Phase 6: causal, data-supported feature construction.

Every generated feature at timestamp t uses measurements at t or earlier. Streams
are joined by exact timestamp to the selected reference timeline; this phase does
not interpolate or forward-fill measurements.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from src.core.errors import DataUndetermined
from src.core.interfaces import FeatureBuilder
from src.core.logging_setup import TransformationLog


_ERROR_NAME = re.compile(r"(^|_)(stat_?err(or)?|err(or)?|sigma|uncertainty|quality|flag)(_|$)", re.I)


def _safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower() or "unnamed"


def _numeric_measurement_columns(stream) -> tuple[list[str], list[str]]:
    available = [c for c in getattr(stream, "measurement_columns", [])
                 if c in stream.dataframe.columns]
    usable, excluded = [], []
    for col in available:
        if _ERROR_NAME.search(str(col)):
            excluded.append(str(col))
            continue
        if pd.api.types.is_numeric_dtype(stream.dataframe[col]):
            usable.append(str(col))
        else:
            excluded.append(str(col))
    return usable, excluded


class CausalFeatureBuilder(FeatureBuilder):
    """Create per-stream levels, rates, and trailing rolling statistics."""

    PHASE = 6
    IMPLEMENTED = True

    def __init__(self, cfg=None, run_dir=None):
        self.cfg = cfg
        self.run_dir = run_dir
        self.last_provenance: dict[str, Any] = {}

    def build(self, ts, log=None) -> tuple[pd.DataFrame, dict[str, Any]]:
        if not isinstance(ts, dict) or not ts:
            raise DataUndetermined("Phase 6 needs the standardized stream dictionary from Phase 2/4.")
        features_cfg = getattr(self.cfg, "features", None) if self.cfg is not None else None
        history_minutes = getattr(features_cfg, "history_window_minutes", None) if features_cfg else None
        if history_minutes is None or not np.isfinite(float(history_minutes)) or float(history_minutes) <= 0:
            raise DataUndetermined(
                "Phase 6 requires features.history_window_minutes to be set to a positive number of minutes."
            )
        groups = list(getattr(features_cfg, "groups", ["levels", "rates", "rolling_stats"]))
        rolling_windows = getattr(features_cfg, "rolling_window_minutes", None) or [float(history_minutes)]
        rolling_windows = sorted({float(w) for w in rolling_windows})
        if any(not np.isfinite(w) or w <= 0 for w in rolling_windows):
            raise ValueError("features.rolling_window_minutes must contain positive finite values.")
        rolling_windows = [w for w in rolling_windows if w <= float(history_minutes)]
        if "rolling_stats" in groups and not rolling_windows:
            raise DataUndetermined(
                "No valid rolling windows remain: set rolling_window_minutes at or below history_window_minutes."
            )

        candidates = [s for s in ts.values()
                      if hasattr(s, "dataframe") and "timestamp" in s.dataframe.columns]
        if not candidates:
            raise DataUndetermined("Phase 6 found no stream with a timestamp column.")
        reference = next((s for s in candidates
                          if getattr(s, "metadata", {}).get("phase4_role") == "reference_timeline"), None)
        if reference is None:
            reference = max(candidates, key=lambda s: len(s.dataframe))
        timeline = pd.to_datetime(reference.dataframe["timestamp"], utc=True, errors="coerce")
        timeline = timeline.dropna().drop_duplicates().sort_values().reset_index(drop=True)
        if timeline.empty:
            raise DataUndetermined("Phase 6 reference timeline has no valid timestamps.")
        result = pd.DataFrame({"timestamp": timeline})

        source_report = []
        feature_columns: dict[str, list[str]] = {name: [] for name in groups}
        eligible_channels = []
        for stream_id, stream in ts.items():
            if not hasattr(stream, "dataframe") or "timestamp" not in stream.dataframe.columns:
                continue
            usable, excluded = _numeric_measurement_columns(stream)
            source_report.append({
                "stream_id": stream_id,
                "instrument": getattr(stream, "instrument", None),
                "detector": getattr(stream, "detector", None),
                "source_file": getattr(stream, "source_file", None),
                "measurement_columns_considered": list(getattr(stream, "measurement_columns", [])),
                "measurement_columns_used": usable,
                "measurement_columns_excluded": excluded,
                "rows": int(len(stream.dataframe)),
            })
            if not usable:
                continue
            frame = stream.dataframe.copy()
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")
            frame = frame.drop_duplicates("timestamp", keep="last")
            tag = _safe_name(stream_id)
            selected = frame[["timestamp"] + usable].rename(
                columns={col: f"raw__{tag}__{_safe_name(col)}" for col in usable}
            )
            result = result.merge(selected, on="timestamp", how="left", validate="one_to_one")
            for col in usable:
                raw_name = f"raw__{tag}__{_safe_name(col)}"
                values = pd.to_numeric(result[raw_name], errors="coerce")
                channel = {
                    "stream_id": stream_id,
                    "instrument": str(getattr(stream, "instrument", "") or "").strip(),
                    "column": col,
                    "raw_feature": raw_name,
                    "values": values,
                }
                eligible_channels.append(channel)

        if not eligible_channels:
            raise DataUndetermined("Phase 6 found no numeric measurement channels suitable for forecasting.")

        # Detect gaps from the reference timeline. Rolling statistics are restarted
        # after a gap so they cannot silently span unobserved intervals.
        stamp_ns = result["timestamp"].astype("int64").to_numpy()
        diffs = np.diff(stamp_ns) / 1e9
        positive = diffs[np.isfinite(diffs) & (diffs > 0)]
        cadence = float(np.median(positive)) if len(positive) else None
        gap_threshold = max(5.0, 3.0 * cadence) if cadence is not None else 5.0
        segment_ids = np.zeros(len(result), dtype=np.int64)
        if len(diffs):
            segment_ids[1:] = np.cumsum(diffs > gap_threshold)
        result["segment_id"] = segment_ids

        for channel in eligible_channels:
            raw_name = channel["raw_feature"]
            base = f"{_safe_name(channel['stream_id'])}__{_safe_name(channel['column'])}"
            values = pd.to_numeric(result[raw_name], errors="coerce")
            if "levels" in groups:
                name = f"level__{base}"
                result[name] = values
                feature_columns["levels"].append(name)
            if "rates" in groups:
                elapsed = result["timestamp"].diff().dt.total_seconds()
                rate = values.diff().div(elapsed.where(elapsed > 0))
                # Do not calculate a rate across an observation gap.
                rate.loc[result["segment_id"].diff().fillna(0).ne(0)] = np.nan
                name = f"rate_per_second__{base}"
                result[name] = rate
                feature_columns["rates"].append(name)
            if "rolling_stats" in groups:
                for window in rolling_windows:
                    mean_name = f"rolling_mean_{_safe_name(window)}m__{base}"
                    std_name = f"rolling_std_{_safe_name(window)}m__{base}"
                    mean_values = pd.Series(np.nan, index=result.index, dtype=float)
                    std_values = pd.Series(np.nan, index=result.index, dtype=float)
                    for _, indices in result.groupby("segment_id", sort=False).groups.items():
                        idx = list(indices)
                        segment = pd.Series(values.loc[idx].to_numpy(),
                                            index=pd.DatetimeIndex(result.loc[idx, "timestamp"]))
                        rolling = segment.rolling(f"{window}min", min_periods=2, closed="both")
                        mean_values.loc[idx] = rolling.mean().to_numpy()
                        std_values.loc[idx] = rolling.std(ddof=0).to_numpy()
                    result[mean_name] = mean_values
                    result[std_name] = std_values
                    feature_columns["rolling_stats"].extend([mean_name, std_name])

        cross_features = []
        if "cross_instrument" in groups:
            solexs_channels = [c for c in eligible_channels
                               if c["instrument"].strip().upper() == "SOLEXS"]
            helios_channels = [c for c in eligible_channels
                               if c["instrument"].strip().upper() == "HEL1OS"]
            for soft in solexs_channels:
                for hard in helios_channels:
                    soft_values = pd.to_numeric(soft["values"], errors="coerce")
                    hard_values = pd.to_numeric(hard["values"], errors="coerce")
                    for window in rolling_windows:
                        name = (
                            f"rolling_corr_{_safe_name(window)}m__"
                            f"{_safe_name(soft['stream_id'])}_{_safe_name(soft['column'])}__"
                            f"{_safe_name(hard['stream_id'])}_{_safe_name(hard['column'])}"
                        )
                        corr_values = pd.Series(np.nan, index=result.index, dtype=float)
                        for _, indices in result.groupby("segment_id", sort=False).groups.items():
                            idx = list(indices)
                            time_index = pd.DatetimeIndex(result.loc[idx, "timestamp"])
                            left = pd.Series(soft_values.loc[idx].to_numpy(), index=time_index)
                            right = pd.Series(hard_values.loc[idx].to_numpy(), index=time_index)
                            corr_values.loc[idx] = left.rolling(
                                f"{window}min", min_periods=3, closed="both"
                            ).corr(right).to_numpy()
                        result[name] = corr_values
                        cross_features.append(name)
                        feature_columns["cross_instrument"].append(name)
            cross_instrument_note = (
                f"Generated {len(cross_features)} trailing rolling Pearson correlation feature(s) "
                "between SoLEXS and HEL1OS channels. Correlation is scale-invariant; no cross-instrument "
                "ratios or differences are assumed. Windows use current/past observations only."
                if cross_features else
                "Not generated: both SoLEXS and HEL1OS numeric signal channels are required."
            )
        else:
            cross_instrument_note = "Not requested in features.groups."
        result = result.drop(columns=[c for c in result.columns if c.startswith("raw__")])
        result = result.sort_values("timestamp", kind="stable").reset_index(drop=True)
        provenance = {
            "phase": 6,
            "reference_stream_id": getattr(reference, "stream_id", None),
            "reference_instrument": getattr(reference, "instrument", None),
            "rows": int(len(result)),
            "timestamp_start_utc": result["timestamp"].min().isoformat(),
            "timestamp_end_utc": result["timestamp"].max().isoformat(),
            "median_cadence_seconds": cadence,
            "gap_threshold_seconds": float(gap_threshold),
            "continuous_segments": int(result["segment_id"].nunique()),
            "history_window_minutes": float(history_minutes),
            "rolling_window_minutes": rolling_windows if "rolling_stats" in groups else [],
            "requested_groups": groups,
            "generated_feature_columns": {k: v for k, v in feature_columns.items()},
            "feature_count": int(sum(len(v) for v in feature_columns.values())),
            "source_streams": source_report,
            "cross_instrument_features": cross_instrument_note,
            "causality_policy": "exact timestamp joins; no interpolation or forward fill; trailing windows only; rates use current and previous observation; rolling windows restart at detected gaps.",
        }
        self.last_provenance = provenance
        if isinstance(log, TransformationLog):
            log.record(
                "causal_features", "build_causal_features",
                rows_before=len(timeline), rows_after=len(result),
                feature_count=provenance["feature_count"],
                requested_groups=groups,
                history_window_minutes=float(history_minutes),
                rolling_window_minutes=rolling_windows,
                reference_stream_id=provenance["reference_stream_id"],
                causality_policy=provenance["causality_policy"],
            )
        return result, provenance
