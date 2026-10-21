import pandas as pd

from src.data_adapter.adapter import StandardizedStream
from src.validation.quality_checks import DataQualityValidator
from src.preprocessing.cleaner import TimeSeriesPreprocessor


def _stream():
    ts = pd.to_datetime([
        "2026-09-25T00:00:02Z",
        "2026-09-25T00:00:00Z",
        "2026-09-25T00:00:01Z",
        "2026-09-25T00:00:01Z",
    ], utc=True)
    return {
        "s": StandardizedStream(
            stream_id="s", instrument="SoLEXS", detector="SDD2",
            product_type="lightcurve", extension="RATE",
            source_file="test.lc.gz", measurement_columns=["COUNTS"],
            dataframe=pd.DataFrame({"timestamp": ts, "COUNTS": [1.0, 2.0, float("nan"), 4.0]}),
        )
    }


def test_quality_report_detects_missing_and_duplicates():
    report = DataQualityValidator().check(_stream())
    r = report.streams["s"]
    assert r.timestamp_duplicates == 2
    assert r.measurements["COUNTS"]["missing_or_non_numeric"] == 1


def test_report_only_preserves_missing_and_removes_no_duplicates():
    streams = _stream()
    out = TimeSeriesPreprocessor().apply(streams)
    assert len(out["s"].dataframe) == 4
    assert out["s"].dataframe["COUNTS"].isna().sum() == 1


def test_keep_first_duplicate_policy():
    class Cfg:
        class P:
            duplicate_timestamp_policy = "keep_first"
            sort_chronologically = True
        preprocessing = P()
    out = TimeSeriesPreprocessor(Cfg()).apply(_stream())
    df = out["s"].dataframe
    assert len(df) == 3
    assert df["timestamp"].is_monotonic_increasing
