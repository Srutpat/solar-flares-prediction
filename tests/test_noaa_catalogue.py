from datetime import date

from src.event_detection.noaa_catalogue import parse_swpc_event_report


def test_parse_exact_xray_begin_and_class():
    report = """# Example NOAA SWPC report\n#Event Begin Max End Obs Q Type Loc/Frq Particulars Reg#\n  1 0012 0018 0024 G16 5 XRA 1-8A C2.4 4551\n  2 B0120 0124 0130 G16 5 XRA 1-8A C1.0 4551\n  3 0140 U0144 0150 G16 5 XRA 1-8A M1.2 4551\n  4 0200 0205 0210 G16 5 OPT 1 2N 4551\n"""
    df = parse_swpc_event_report(report, date(2026, 10, 17), "https://example.test/report.txt")
    assert len(df) == 2  # exact Begin times are retained even if the peak is uncertain
    assert df.loc[0, "onset_utc"] == "2026-10-17T00:12:00+00:00"
    assert df.loc[0, "class"] == "C2.4"
    assert df.loc[1, "onset_utc"] == "2026-10-17T01:40:00+00:00"
    assert df.loc[1, "class"] == "M1.2"
    assert "NOAA SWPC" in df.loc[0, "source"]


def test_empty_report_returns_schema_not_fake_events():
    df = parse_swpc_event_report("# no xray rows\n", date(2026, 10, 17), "https://example.test/report.txt")
    assert df.empty
    assert {"onset_utc", "source", "class"}.issubset(df.columns)


def test_parse_plus_marker_does_not_corrupt_event_id():
    report = """# Event Begin Max End Obs Q Type Loc/Frq Particulars Reg#
  42 + 0310 0314 0320 G18 5 XRA 1-8A C3.1 4551
"""
    df = parse_swpc_event_report(report, date(2026, 10, 17), "https://example.test/report.txt")
    assert len(df) == 1
    assert "-42-" in df.loc[0, "event_id"]
    assert df.loc[0, "onset_utc"] == "2026-10-17T03:10:00+00:00"
