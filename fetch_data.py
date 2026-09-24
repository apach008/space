"""Download the raw files the site is built from into data/.

Run once to take a snapshot. The satellite catalog changes every day, so
running this again replaces the snapshot and the numbers in the report
page have to be updated from the new prepare_data.py output.

Needs pandas and lxml.
"""
import io
import urllib.request
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "space-catalog-site"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


# 1. The satellite catalog (SATCAT), one row per cataloged object.
(DATA / "satcat.csv").write_bytes(get("https://celestrak.org/pub/satcat.csv"))

# 2. What the owner and launch site codes mean. CelesTrak publishes these as
#    HTML tables, so they are read with pandas and saved as CSV.
for page, out, cols in [
    ("sources", "owner_codes.csv", ["code", "owner_name"]),
    ("launchsites", "launch_site_codes.csv", ["code", "site_name"]),
]:
    html = get(f"https://celestrak.org/satcat/{page}.php").decode("utf-8", errors="replace")
    t = pd.read_html(io.StringIO(html))[0]
    t.columns = cols
    t.to_csv(DATA / out, index=False, lineterminator="\n")

# 3. Land outlines for the globe, from Natural Earth (public domain).
(DATA / "ne_50m_land.geojson").write_bytes(get(
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_land.geojson"))

for f in sorted(DATA.iterdir()):
    print(f"{f.name:28s} {f.stat().st_size:>12,d} bytes")
