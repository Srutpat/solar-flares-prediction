import pandas as pd
from pathlib import Path
from src.event_detection.noaa_catalogue import download_swpc_reports

start = pd.Timestamp("2026-09-01", tz="UTC")
end = pd.Timestamp("2026-10-18", tz="UTC")
cat, _ = download_swpc_reports(start, end, Path("cache_scan/noaa_swpc"))  # cached, so fast

cat["onset_utc"] = pd.to_datetime(cat["onset_utc"], utc=True)
cat["day"] = cat["onset_utc"].dt.date
cat["cls"] = cat["class"].astype(str).str.strip().str[0].str.upper()

# counts per day and class
counts = cat.groupby(["day", "cls"]).size().unstack(fill_value=0)
for c in "BCMX":
    if c not in counts:
        counts[c] = 0
counts = counts[["B", "C", "M", "X"]]
print(counts.to_string())

# M/X onset times (UTC) so you can check them against instrument coverage
big = cat[cat["cls"].isin(["M", "X"])][["day", "onset_utc", "class"]]
print("\nM/X flares:")
print(big.to_string(index=False))

# save for later
cat.to_csv("cache_scan/flare_catalogue_sep_oct.csv", index=False)
counts.to_csv("cache_scan/flares_per_day.csv")