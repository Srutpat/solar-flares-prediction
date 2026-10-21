"""Retrieve and parse NOAA/SWPC daily edited solar event reports.

Only XRA rows with an exact (unprefixed) begin time are emitted as onset labels.
Times marked before/after/uncertain are excluded because they do not provide an
exact onset. The downloaded report is retained as provenance in the run folder.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import re

import pandas as pd


NOAA_BASE = "https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/solar_event_reports"
MIRROR_BASE = "https://iswa.ccmc.gsfc.nasa.gov/iswa_data_tree/ancillary/solar/noaa-swpc/events"


def parse_swpc_event_report(text: str, report_date: date, source_url: str) -> pd.DataFrame:
    """Parse NOAA fixed-width/whitespace event reports into standardized flare rows."""
    records = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        tokens = line.split()
        try:
            idx = tokens.index("XRA")
        except ValueError:
            continue
        # Report columns: Event, Begin, Max, End, Obs, Q, Type, Loc/Frq, Particulars, Reg#.
        if idx < 6 or len(tokens) <= idx + 2:
            continue
        # Some reports put a standalone '+' after the event number. It means
        # SWPC selected this report to represent multiple reports for the same
        # event; it must not shift the event ID or time columns.
        if idx >= 7 and tokens[idx - 6] == "+":
            event_id = tokens[idx - 7]
        else:
            event_id = tokens[idx - 6]
        begin, peak, end = tokens[idx - 5], tokens[idx - 4], tokens[idx - 3]
        flare_class = tokens[idx + 2] if re.fullmatch(r"[BCMX]\d+(?:\.\d+)?", tokens[idx + 2], re.I) else ""
        # Prefixes A/B/U indicate after/before/uncertain; these are not exact onset times.
        if not re.fullmatch(r"\d{4}", begin):
            continue
        hour, minute = int(begin[:2]), int(begin[2:])
        if hour > 23 or minute > 59:
            continue
        onset = datetime(report_date.year, report_date.month, report_date.day, hour, minute, tzinfo=timezone.utc)
        def parse_event_time(value: str):
            if not re.fullmatch(r"\d{4}", value):
                return ""
            h, m = int(value[:2]), int(value[2:])
            if h > 23 or m > 59:
                return ""
            d = report_date + timedelta(days=1) if h * 60 + m < hour * 60 + minute else report_date
            return datetime(d.year, d.month, d.day, h, m, tzinfo=timezone.utc).isoformat()
        records.append({
            "event_id": f"SWPC-{report_date:%Y%m%d}-{event_id}-{line_no}",
            "onset_utc": onset.isoformat(),
            "peak_utc": parse_event_time(peak),
            "end_utc": parse_event_time(end),
            "class": flare_class,
            "source": f"NOAA SWPC Solar Event Report ({report_date.isoformat()}); {source_url}",
            "notes": f"XRA row {line_no}; only exact Begin times retained",
        })
    columns = ["event_id", "onset_utc", "peak_utc", "end_utc", "class", "source", "notes"]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records, columns=columns).drop_duplicates(subset=["onset_utc", "class"]).reset_index(drop=True)


def _urls_for_date(day: date) -> list[str]:
    stamp = day.strftime("%Y%m%d")
    return [
        f"{NOAA_BASE}/{day:%Y/%m}/{stamp}events.txt",
        f"{MIRROR_BASE}/{day:%Y/%m}/{stamp}events.txt",
    ]


def download_swpc_reports(start: pd.Timestamp, end: pd.Timestamp, cache_dir: str | Path) -> tuple[pd.DataFrame, list[dict]]:
    """Download each UTC daily report intersecting [start, end], cache raw sources, parse events."""
    start = pd.to_datetime(start, utc=True)
    end = pd.to_datetime(end, utc=True)
    if pd.isna(start) or pd.isna(end) or end < start:
        raise ValueError("Invalid observation coverage for NOAA catalogue retrieval.")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    reports, provenance = [], []
    day = start.date()
    final_day = end.date()
    while day <= final_day:
        last_error = None
        report_text = None
        selected_url = None
        for url in _urls_for_date(day):
            try:
                req = Request(url, headers={"User-Agent": "SolarFlareResearchFramework/0.1 (research data retrieval)"})
                with urlopen(req, timeout=25) as response:
                    report_text = response.read().decode("utf-8", errors="replace")
                selected_url = url
                break
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                last_error = exc
        if report_text is None:
            raise RuntimeError(f"Could not retrieve NOAA/SWPC report for {day.isoformat()}: {last_error}")
        raw_path = cache_dir / f"{day:%Y%m%d}events.txt"
        raw_path.write_text(report_text, encoding="utf-8")
        parsed = parse_swpc_event_report(report_text, day, selected_url)
        reports.append(parsed)
        provenance.append({"date_utc": day.isoformat(), "url": selected_url, "raw_report": str(raw_path),
                           "parsed_xra_events": int(len(parsed))})
        day += timedelta(days=1)
    if reports:
        catalogue = pd.concat(reports, ignore_index=True)
    else:
        catalogue = pd.DataFrame(columns=["event_id", "onset_utc", "peak_utc", "end_utc", "class", "source", "notes"])
    return catalogue, provenance
