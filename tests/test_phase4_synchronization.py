import pandas as pd

from src.data_adapter.adapter import StandardizedStream
from src.synchronization.synchronizer import CausalSynchronizer


class Cfg:
    class Sync:
        method = "asof_backward"
        max_staleness_seconds = 1.0
    synchronization = Sync()


def _stream(stream_id, instrument, times, values, detector):
    return StandardizedStream(
        stream_id=stream_id,
        instrument=instrument,
        detector=detector,
        product_type="lightcurve",
        extension="RATE",
        source_file=f"{stream_id}.fits",
        measurement_columns=["RATE"],
        dataframe=pd.DataFrame({
            "timestamp": pd.to_datetime(times, utc=True),
            "RATE": values,
        }),
    )


def test_asof_backward_never_uses_future_observation():
    streams = {
        "solexs": _stream(
            "solexs", "SoLEXS",
            ["2026-09-25T00:00:13Z", "2026-09-25T00:00:14Z", "2026-09-25T00:00:15Z"],
            [10, 11, 12], "SDD2",
        ),
        "helios": _stream(
            "helios", "HEL1OS",
            ["2026-09-25T00:00:13.700Z", "2026-09-25T00:00:14.700Z", "2026-09-25T00:00:15.700Z"],
            [20, 21, 22], "CZT1",
        ),
    }
    sync = CausalSynchronizer(Cfg())
    out = sync.synchronize(streams)
    df = out["helios"].dataframe
    # The common overlap begins at 13.700, so the first usable reference
    # timestamp is 14.000 and can only use the 13.700 HEL1OS observation.
    assert df.loc[0, "RATE"] == 20
    assert df.loc[1, "RATE"] == 21
    assert sync.report()["streams"]["helios"]["unmatched_rows"] == 0


def test_overlap_is_used_and_reported():
    streams = {
        "solexs": _stream(
            "solexs", "SoLEXS",
            ["2026-09-25T00:00:00Z", "2026-09-25T00:00:01Z", "2026-09-25T00:00:02Z"],
            [1, 2, 3], "SDD2",
        ),
        "helios": _stream(
            "helios", "HEL1OS",
            ["2026-09-25T00:00:01.700Z", "2026-09-25T00:00:02.700Z"],
            [4, 5], "CZT1",
        ),
    }
    sync = CausalSynchronizer(Cfg())
    out = sync.synchronize(streams)
    assert len(out["solexs"].dataframe) == 1
    report = sync.report()
    assert report["method"] == "asof_backward"
    assert report["overlap_start_utc"].startswith("2026-09-25T00:00:01")


def test_single_instrument_is_passthrough():
    streams = {
        "solexs": _stream(
            "solexs", "SoLEXS",
            ["2026-09-25T00:00:00Z", "2026-09-25T00:00:01Z"],
            [1, 2], "SDD2",
        )
    }
    sync = CausalSynchronizer(Cfg())
    out = sync.synchronize(streams)
    assert out["solexs"].dataframe.equals(streams["solexs"].dataframe)
    assert sync.report()["method"] == "single_instrument_passthrough"


def test_nonoverlapping_later_archive_segment_is_skipped_not_used_in_global_intersection():
    streams = {
        "solexs": _stream(
            "solexs", "SoLEXS",
            ["2026-10-17T00:00:00Z", "2026-10-17T00:00:01Z", "2026-10-17T00:00:02Z"],
            [1, 2, 3], "SDD2",
        ),
        "helios_first_half": _stream(
            "helios_first_half", "HEL1OS",
            ["2026-10-17T00:00:00Z", "2026-10-17T00:00:01Z", "2026-10-17T00:00:02Z"],
            [4, 5, 6], "CZT1",
        ),
        "helios_second_half": _stream(
            "helios_second_half", "HEL1OS",
            ["2026-10-17T12:00:00Z", "2026-10-17T12:00:01Z"],
            [7, 8], "CZT1",
        ),
    }
    sync = CausalSynchronizer(Cfg())
    out = sync.synchronize(streams)
    assert "helios_first_half" in out
    assert "helios_second_half" not in out
    assert len(out["solexs"].dataframe) == 3
    assert any("Skipped 1 stream(s)" in note for note in sync.report()["notes"])
