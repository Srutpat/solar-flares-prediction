from datetime import date
from pathlib import Path
from src.event_detection.noaa_catalogue import download_swpc_reports

start, end = date(2026, 9, 1), date(2026, 10, 18)
cat, prov = download_swpc_reports(start, end, Path("cache_scan/noaa_swpc"))
print(cat.columns.tolist())
print(cat.head(10))
print(prov)