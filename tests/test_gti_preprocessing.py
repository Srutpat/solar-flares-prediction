from types import SimpleNamespace
import pandas as pd

from src.preprocessing.cleaner import TimeSeriesPreprocessor


class _Cfg:
    class preprocessing:
        duplicate_timestamp_policy = "report_only"
        sort_chronologically = True


def test_solexs_rows_outside_declared_gti_are_removed():
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-10-17T00:00:21Z", "2026-10-17T00:00:22Z",
            "2026-10-17T00:00:23Z", "2026-10-17T04:00:42Z",
        ], utc=True),
        "COUNTS": [100, 101, 102, 999],
    })
    stream = SimpleNamespace(
        dataframe=frame, instrument="SoLEXS", metadata={
            "gti_status": "available",
            "gti_intervals_utc": [("2026-10-17T00:00:22+00:00", "2026-10-17T04:00:41+00:00")],
            "gti_source": "test-gti",
        }
    )
    cleaned = TimeSeriesPreprocessor(_Cfg()).apply({"solexs": stream})["solexs"]
    assert cleaned.dataframe["COUNTS"].tolist() == [101, 102]


def test_solexs_empty_gti_does_not_mark_rows_as_valid():
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2026-10-17T00:00:22Z"], utc=True), "COUNTS": [1]})
    stream = SimpleNamespace(dataframe=frame, instrument="SoLEXS", metadata={"gti_status": "available", "gti_intervals_utc": []})
    cleaned = TimeSeriesPreprocessor(_Cfg()).apply({"solexs": stream})["solexs"]
    assert cleaned.dataframe.empty


def test_helios_rows_outside_declared_gti_are_removed():
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2026-10-17T00:00:00Z", "2026-10-17T00:00:01Z", "2026-10-17T00:00:02Z"], utc=True), "COUNTS": [1, 2, 3]})
    stream = SimpleNamespace(dataframe=frame, instrument="HEL1OS", metadata={"gti_status": "available", "gti_intervals_utc": [("2026-10-17T00:00:01+00:00", "2026-10-17T00:00:02+00:00")]})
    cleaned = TimeSeriesPreprocessor(_Cfg()).apply({"helios": stream})["helios"]
    assert cleaned.dataframe["COUNTS"].tolist() == [2, 3]


def test_big_endian_numeric_column_is_normalized_before_gti_boolean_filter():
    import numpy as np

    frame = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-10-17T00:00:21Z", "2026-10-17T00:00:22Z", "2026-10-17T00:00:23Z"
        ], utc=True),
        "COUNTS": np.array([100.0, 101.0, 102.0], dtype=">f8"),
    })
    assert not frame["COUNTS"].to_numpy().dtype.isnative
    stream = SimpleNamespace(
        dataframe=frame, instrument="SoLEXS", metadata={
            "gti_status": "available",
            "gti_intervals_utc": [("2026-10-17T00:00:22+00:00", "2026-10-17T00:00:23+00:00")],
            "gti_source": "big-endian-regression-test",
        }
    )
    cleaned = TimeSeriesPreprocessor(_Cfg()).apply({"solexs": stream})["solexs"]
    assert cleaned.dataframe["COUNTS"].tolist() == [101.0, 102.0]
    assert cleaned.dataframe["COUNTS"].to_numpy().dtype.isnative
