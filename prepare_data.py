"""Build the data the two pages use and print the numbers the report quotes.

Reads the raw files that fetch_data.py saved in data/ and writes

    data/satcat_site.csv       the 17 columns both pages load, one row per object
    data/report_numbers.json   the aggregates behind every report chart
    data/land_mask.png         the land/sea mask the globe is drawn from

It then copies those three files into index.html and dashboard.html,
between the DATA markers, so each page runs on its own when opened from
disk. The numbers written in the report text are typed by hand from the
tables this script prints.

Needs pandas and Pillow.
"""
import base64
import gzip
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).parent
DATA = ROOT / "data"
SNAPSHOT = "2026-09-23"          # the day data/satcat.csv was downloaded
SNAP = pd.Timestamp(SNAPSHOT)
R_EARTH = 6378.137               # km, equatorial radius

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)

# ---------------------------------------------------------------------------
# Load and label
# ---------------------------------------------------------------------------
raw = pd.read_csv(DATA / "satcat.csv", dtype={"NORAD_CAT_ID": int})
owners = pd.read_csv(DATA / "owner_codes.csv")
sites = pd.read_csv(DATA / "launch_site_codes.csv")

df = raw.rename(columns=str.lower).copy()
df["launch_date"] = pd.to_datetime(df.launch_date)
df["decay_date"] = pd.to_datetime(df.decay_date)
df["launch_year"] = df.launch_date.dt.year
df["decayed"] = df.decay_date.notna()

TYPE_NAMES = {"PAY": "Payload", "R/B": "Rocket body", "DEB": "Debris", "UNK": "Unknown"}
df["type"] = df.object_type.map(TYPE_NAMES)

# Short names for the owners that appear in charts; the rest use CelesTrak's name.
SHORT_OWNER = {
    "US": "United States", "CIS": "Russia (USSR)", "PRC": "China", "FR": "France",
    "JPN": "Japan", "IND": "India", "UK": "United Kingdom", "ESA": "ESA",
    "ISS": "ISS partners", "GLOB": "Globalstar", "ORB": "Orbcomm", "IRID": "Iridium",
    "SES": "SES", "EUTE": "Eutelsat", "O3B": "O3b", "ITSO": "Intelsat", "TBD": "Unassigned",
    "UNK": "Unknown owner", "GER": "Germany", "CA": "Canada", "SKOR": "South Korea",
    "ISRA": "Israel", "IT": "Italy", "BRAZ": "Brazil", "SPN": "Spain", "AUS": "Australia",
}
owner_name = dict(zip(owners.code, owners.owner_name))
df["owner_name"] = [SHORT_OWNER.get(c, owner_name.get(c, c)) for c in df.owner]

SHORT_SITE = {
    "AFETR": "Cape Canaveral", "AFWTR": "Vandenberg", "TYMSC": "Baikonur",
    "PLMSC": "Plesetsk", "TAISC": "Taiyuan", "FRGUI": "Kourou", "JSC": "Jiuquan",
    "SRILR": "Sriharikota", "XICLF": "Xichang", "TANSC": "Tanegashima",
    "VOSTO": "Vostochny", "WSC": "Wenchang", "RLLB": "Mahia (Rocket Lab)",
    "KYMSC": "Kapustin Yar", "WLPIS": "Wallops Island", "KSCUT": "Uchinoura",
    "YSLA": "Yellow Sea (sea launch)", "DLS": "Dombarovsky", "SEAL": "Sea Launch",
    "ERAS": "Eastern Range air launch", "WRAS": "Western Range air launch",
    "HGSTR": "Hammaguira", "KODAK": "Kodiak", "NSC": "Naro", "SCSLA": "South China Sea launch",
    "YAVNE": "Palmachim", "SVOBO": "Svobodny", "SEMLS": "Semnan", "SNMLP": "San Marco",
    "SMTS": "Shahrud", "SUBL": "Submarine launch", "YUN": "Sohae", "KWAJ": "Kwajalein",
    "ANDSP": "Andoya", "WOMRA": "Woomera", "CAS": "Canary Islands air launch",
    "JJSLA": "Jeju sea launch",
}
site_name = dict(zip(sites.code, sites.site_name))
df["site_name"] = [SHORT_SITE.get(c, site_name.get(c, c)) for c in df.launch_site]

# Orbit regime from the catalog's apogee and perigee (km above the surface).
peri, apo = df.perigee, df.apogee
ecc = (apo - peri) / (apo + peri + 2 * R_EARTH)
df["eccentricity"] = ecc.round(5)
regime = pd.Series("No orbit data", index=df.index)
has = apo.notna() & peri.notna()
regime[has & (ecc >= 0.25)] = "Highly elliptical"
regime[has & (ecc < 0.25) & (apo < 2000)] = "Low Earth orbit"
regime[has & (ecc < 0.25) & (apo >= 2000) & (peri < 35000)] = "Medium Earth orbit"
regime[has & (ecc < 0.25) & (peri >= 35000)] = "Geosynchronous and above"
regime[df.orbit_center != "EA"] = "Beyond Earth orbit"
# Docked objects carry the NORAD number of what they are docked to as their center.
regime[df.orbit_center.str.fullmatch(r"\d+")] = "Docked"
df["regime"] = regime

# An active payload is a payload with an active operational status. The
# status applies to payloads only: the two rocket bodies that carry "+" in
# this snapshot are not counted as active.
ACTIVE = {"+", "P", "B", "S", "X"}
is_pay = df.type == "Payload"
status = pd.Series("No status (debris, stages)", index=df.index)
status[is_pay & df.ops_status_code.isin(ACTIVE)] = "Active"
status[is_pay & (df.ops_status_code == "-")] = "Inactive"
status[df.decayed] = "Reentered"
df["status"] = status

df["in_orbit"] = ~df.decayed
df["in_earth_orbit"] = ~df.decayed & ((df.orbit_center == "EA") | df.orbit_center.str.fullmatch(r"\d+"))
# A Starlink satellite is a payload whose name starts with STARLINK (two debris
# pieces also carry the name and are left out).
df["starlink"] = is_pay & df.object_name.str.startswith("STARLINK", na=False)
df["launch_id"] = df.object_id.str[:8]
df["mean_alt"] = (df.apogee + df.perigee) / 2
end = df.decay_date.fillna(SNAP)
df["years_in_orbit"] = ((end - df.launch_date).dt.days / 365.25).round(2)

# Payloads released from the space station carry its launch date (1998-11-20).
from_station = (df.type == "Payload") & (df.launch_date == "1998-11-20") & (df.norad_cat_id != 25544)

# When each object entered the catalog.
# Payloads and rocket bodies enter on their launch date. A debris piece carries
# the launch date of the object it broke off, which can be decades before the
# breakup (Fengyun 1C debris is dated 1999, the breakup was in 2007), and the
# payloads released from the space station carry the station's 1998 date. For
# those, and for unidentified objects, the date is estimated from the catalog
# number: catalog numbers are handed out in order, so the estimate is taken
# from the launch dates of the payloads and rocket bodies numbered around it.
# In catalog-number order: a rolling median over 51 neighbors, then a running
# maximum (so the estimate never goes back in time as the numbers rise), then
# interpolated at each catalog number. The object enters on the later of its
# launch date and that estimate less 30 days (new objects often receive
# numbers some days after they are first tracked), and never after its
# reentry date.
ref = df[df.type.isin(["Payload", "Rocket body"]) & ~from_station].sort_values("norad_cat_id")
ref_days = (ref.launch_date - pd.Timestamp("1957-01-01")).dt.days
ref_days = ref_days.rolling(51, center=True, min_periods=1).median().cummax()
est = np.interp(df.norad_cat_id, ref.norad_cat_id, ref_days)
df["cataloged"] = pd.Timestamp("1957-01-01") + pd.to_timedelta(est.round(), unit="D")
estimated = df.type.isin(["Debris", "Unknown"]) | from_station
guess = df.launch_date.where(df.launch_date >= df.cataloged - pd.Timedelta(days=30),
                             df.cataloged - pd.Timedelta(days=30))
guess = guess.where(~(df.decayed & (guess > df.decay_date)), df.decay_date)
df["appear"] = df.launch_date.where(~estimated, guess)
late = df[df.appear > df.launch_date + pd.Timedelta(days=365)]
print(f"\nObjects that enter the catalog more than a year after their launch date: {len(late):,} "
      f"({late.type.value_counts().to_dict()})")
print(f"Estimated entries capped at the reentry date: {int((estimated & df.decayed & (df.appear == df.decay_date)).sum()):,}")
for nm in ["FENGYUN 1C DEB", "COSMOS 1408 DEB", "IRIDIUM 33 DEB", "COSMOS 2251 DEB"]:
    x = df[df.object_name == nm].appear
    print(f"  {nm}: appear median {x.median().date()}, 10th pct {x.quantile(.1).date()}")

# The station payloads are counted in the year they entered the catalog.
# Every other object is counted in its launch year.
df["year"] = df.launch_year.where(~from_station, df.appear.dt.year)
print(f"Payloads with the station launch date (station itself excluded): {int(from_station.sum())}")

print(f"Rows: {len(df):,}   columns in the raw file: {raw.shape[1]}")
print(f"Launch dates {df.launch_date.min().date()} to {df.launch_date.max().date()}")
print(f"Owners: {df.owner.nunique()}   launch sites: {df.launch_site.nunique()}")
print(df.type.value_counts().to_string(), "\n")

# ---------------------------------------------------------------------------
# Headline numbers
# ---------------------------------------------------------------------------
eo = df[df.in_earth_orbit]
active = df[df.status == "Active"]
head = {
    "cataloged": len(df),
    "in_earth_orbit": len(eo),
    "active_payloads": len(active),
    "debris_in_orbit": int((eo.type == "Debris").sum()),
    "debris_share_in_orbit": round((eo.type == "Debris").mean() * 100, 1),
    "starlink_active": int(active.starlink.sum()),
    "starlink_share_active": round(active.starlink.mean() * 100, 1),
    "owners": int(df.owner.nunique()),
    "reentered": int(df.decayed.sum()),
}
print("HEADLINE", json.dumps(head, indent=1))

report = {"snapshot": SNAPSHOT, "head": head}

# 1. Objects cataloged per launch year, by type -------------------------------
by_year = pd.crosstab(df.year, df.type).reindex(columns=list(TYPE_NAMES.values()), fill_value=0)
by_year["Total"] = by_year.sum(axis=1)
print("\n[1] Objects cataloged by launch year (last 12)\n", by_year.tail(12))
pay_year = df[df.type == "Payload"].groupby("year").size()
print("Payloads by year, 2015-2026:\n", pay_year.loc[2015:].to_string())
print(f"Payloads launched since 2020: {pay_year.loc[2020:].sum():,} of {pay_year.sum():,} "
      f"({pay_year.loc[2020:].sum() / pay_year.sum() * 100:.1f}%)")
print(f"Payloads launched 1957-2019: {pay_year.loc[:2019].sum():,}")
report["by_year"] = {"years": by_year.index.tolist(),
                     **{k: by_year[k].tolist() for k in TYPE_NAMES.values()}}

# 2. Starlink ---------------------------------------------------------------
pay = df[df.type == "Payload"]
sl = pd.crosstab(pay.year, pay.starlink).loc[2015:]
sl.columns = ["Other payloads", "Starlink"]
print("\n[2] Payloads by year, Starlink vs other\n", sl)
sl_all = df[df.starlink]
print(f"Starlink cataloged {len(sl_all):,}, in orbit {sl_all.in_orbit.sum():,}, "
      f"reentered {sl_all.decayed.sum():,}, active {int((sl_all.status == 'Active').sum()):,}")
print(f"Payloads since 2019 that are Starlink: {pay[pay.year >= 2019].starlink.mean() * 100:.1f}%")
print("Active payloads by owner (top 8):\n", active.owner_name.value_counts().head(8).to_string())
report["starlink"] = {"years": sl.index.tolist(), "starlink": sl.Starlink.tolist(),
                      "other": sl["Other payloads"].tolist()}

# 3. Objects in Earth orbit at the start of each year, by type -----------------
earthish = df[(df.orbit_center == "EA") | df.orbit_center.str.fullmatch(r"\d+")]
def in_orbit_at(t):
    """Objects in Earth orbit at time t: entered on or before t and not yet reentered.
    The same rule as both pages (appear <= t < decay)."""
    return earthish[(earthish.appear <= t) & (earthish.decay_date.isna() | (earthish.decay_date > t))]
years = list(range(1958, 2027))
rows = []
for y in years:
    t = pd.Timestamp(f"{y}-01-01")
    alive = in_orbit_at(t)
    rows.append(alive.type.value_counts().reindex(TYPE_NAMES.values(), fill_value=0))
alive_df = pd.DataFrame(rows, index=years)
alive_df.loc[2026.72] = eo.type.value_counts().reindex(TYPE_NAMES.values(), fill_value=0)
print("\n[3] Objects in Earth orbit on 1 January (selected years)\n",
      alive_df.loc[[1970, 1980, 1990, 2000, 2007, 2008, 2010, 2020, 2021, 2022, 2024, 2025, 2026, 2026.72]])
# The month in which payloads in orbit last overtook debris in orbit.
months = pd.date_range("1958-01-01", SNAPSHOT, freq="MS")
lead = []
for t in months:
    alive = in_orbit_at(t)
    lead.append((alive.type == "Payload").sum() > (alive.type == "Debris").sum())
lead = pd.Series(lead, index=months)
flips = lead[lead != lead.shift()]
print("Months where 'payloads outnumber debris' changes:\n", flips.to_string())
report["in_orbit"] = {"years": [float(x) for x in alive_df.index],
                      **{k: alive_df[k].astype(int).tolist() for k in TYPE_NAMES.values()}}

# 4. Breakups: the launches that produced the most cataloged debris ------------
deb = df[df.type == "Debris"]
def event_name(names):
    s = names.str.replace(r"\s*\[.*?\]", "", regex=True).str.replace(r"\s+DEB.*$", "", regex=True)
    return s.mode().iat[0]
ev = (deb.groupby("launch_id")
         .agg(debris=("norad_cat_id", "size"), in_orbit=("in_orbit", "sum"),
              name=("object_name", event_name), owner=("owner_name", "first"),
              launched=("launch_date", "min"), appeared=("appear", "median"),
              incl=("inclination", "median"), alt=("mean_alt", "median"))
         .sort_values("debris", ascending=False))
ev["share_in_orbit"] = (ev.in_orbit / ev.debris * 100).round(1)
top_ev = ev.head(10)
print("\n[4] Launches with the most cataloged debris\n", top_ev)
print(f"Debris pieces: {len(deb):,}; from the top 10 launches: {top_ev.debris.sum():,} "
      f"({top_ev.debris.sum() / len(deb) * 100:.1f}%); launches with any debris: {len(ev):,}")
print(f"Top 10 still in orbit: {top_ev.in_orbit.sum():,} of {len(deb[deb.in_orbit]):,} debris in orbit")
report["breakups"] = [{"launch_id": i, "name": r["name"], "owner": r.owner,
                       "launched": r.launched.strftime("%Y-%m-%d"), "appeared": r.appeared.strftime("%Y-%m-%d"),
                       "debris": int(r.debris),
                       "in_orbit": int(r.in_orbit)} for i, r in top_ev.iterrows()]

# 5. Who launches payloads: share by owner and decade ---------------------------
pay = df[df.type == "Payload"].copy()
pay["decade"] = pay.year // 10 * 10
big = ["United States", "Russia (USSR)", "China"]
pay["owner_group"] = pay.owner_name.where(pay.owner_name.isin(big), "Everyone else")
own = pd.crosstab(pay.decade, pay.owner_group)[big + ["Everyone else"]]
own_pct = (own.div(own.sum(axis=1), axis=0) * 100).round(1)
print("\n[5] Payloads by decade and owner\n", own, "\n", own_pct)
nown = pay.groupby("decade").owner.nunique()
print("Distinct owners launching payloads, by decade\n", nown.to_string())
pay_nosl = pay[~pay.starlink]
own_nosl = pd.crosstab(pay_nosl.decade, pay_nosl.owner_group)[big + ["Everyone else"]]
print("Same without Starlink\n", (own_nosl.div(own_nosl.sum(axis=1), axis=0) * 100).round(1))
report["owners"] = {"decades": own.index.tolist(), "groups": own.columns.tolist(),
                    "counts": own.values.tolist(), "owners_per_decade": nown.tolist()}

# 6. Altitude of objects in Earth orbit ------------------------------------------
eo = df[df.in_earth_orbit & df.mean_alt.notna()].copy()
bins = list(range(100, 2001, 25))
leo = eo[eo.mean_alt < 2000]
hist = (pd.crosstab(pd.cut(leo.mean_alt, bins, right=False), leo.type)
          .reindex(index=pd.IntervalIndex.from_breaks(bins, closed="left"),
                   columns=list(TYPE_NAMES.values()), fill_value=0))
print("\n[6] Objects in Earth orbit by mean altitude, 25 km bands (top 12 bands)\n",
      hist.assign(total=hist.sum(axis=1)).sort_values("total", ascending=False).head(12))
print(f"Below 2,000 km: {len(leo):,} of {len(eo):,} ({len(leo) / len(eo) * 100:.1f}%)")
geo = eo[eo.regime == "Geosynchronous and above"]
print("Regime counts in Earth orbit\n", eo.regime.value_counts().to_string())
print(f"Between 500 and 600 km: {((eo.mean_alt >= 500) & (eo.mean_alt < 600)).sum():,}, "
      f"of which Starlink {(eo.starlink & (eo.mean_alt >= 500) & (eo.mean_alt < 600)).sum():,}")
print(f"Between 700 and 900 km: {((eo.mean_alt >= 700) & (eo.mean_alt < 900)).sum():,}, "
      f"of which debris {((eo.type == 'Debris') & (eo.mean_alt >= 700) & (eo.mean_alt < 900)).sum():,}")
report["altitude"] = {"bins": bins[:-1], **{k: hist[k].tolist() for k in TYPE_NAMES.values()}}
report["regimes"] = eo.regime.value_counts().to_dict()

# 7. Inclination -------------------------------------------------------------------
eo_inc = eo[eo.inclination.notna()]
inc_hist = pd.cut(eo_inc.inclination, range(0, 181, 1), right=False).value_counts().sort_index()
print("\n[7] Inclination, top 12 one-degree bands for objects in Earth orbit\n",
      inc_hist.sort_values(ascending=False).head(12).to_string())
sso = eo_inc[(eo_inc.inclination >= 96) & (eo_inc.inclination < 100) & (eo_inc.mean_alt < 2000)]
print(f"96-100 degrees below 2,000 km (sun-synchronous band): {len(sso):,} "
      f"({len(sso) / len(eo_inc) * 100:.1f}%); Starlink among them {sso.starlink.sum():,}")
for lo, hi in [(0, 1), (43, 44), (51, 52), (53, 54), (69, 71), (86, 88), (97, 98)]:
    band = eo_inc[(eo_inc.inclination >= lo) & (eo_inc.inclination < hi)]
    print(f"  {lo}-{hi} deg: {len(band):,}  starlink {band.starlink.sum():,}  debris {(band.type == 'Debris').sum():,}")
report["inclination"] = {"counts": inc_hist.astype(int).tolist()}

# 8. Age of what is still up, by altitude -------------------------------------------
# Age is counted from the date the object entered the catalog (see section 3),
# so a debris piece is as old as the breakup that made it.
eo_age = eo[eo.mean_alt < 2000].copy()
eo_age["band"] = pd.cut(eo_age.mean_alt, [0, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000], right=False)
eo_age["age"] = (SNAP - eo_age.appear).dt.days / 365.25
eo_age["appear_year"] = eo_age.appear.dt.year
age = eo_age.groupby("band", observed=True).agg(
    objects=("norad_cat_id", "size"), median_year=("appear", lambda s: int(s.median().year)),
    median_age=("age", "median"),
    before_2000=("appear_year", lambda s: round((s < 2000).mean() * 100, 1)))
print("\n[8] Objects below 2,000 km by altitude band (age from catalog entry)\n", age)
report["age"] = {"bands": [f"{int(b.left)}-{int(b.right)}" for b in age.index],
                 "median_age": age.median_age.round(1).tolist(), "objects": age.objects.tolist(),
                 "median_year": age.median_year.tolist(), "pre2000": age.before_2000.tolist()}
print("Band 450-500 km:", ((eo.mean_alt >= 450) & (eo.mean_alt < 500)).sum(),
      "Starlink", (eo.starlink & (eo.mean_alt >= 450) & (eo.mean_alt < 500)).sum(),
      "payloads", ((eo.type == "Payload") & (eo.mean_alt >= 450) & (eo.mean_alt < 500)).sum())
print("Debris in orbit, peak on 1 January:", alive_df.Debris.idxmax(), alive_df.Debris.max())
print("Full in-orbit table\n", alive_df.to_string())
old = eo.sort_values("launch_date").head(5)[["object_name", "launch_date", "perigee", "apogee"]]
print("Oldest objects still in orbit\n", old)

# 9. Reentries per year -----------------------------------------------------------------
dec = df[df.decayed].copy()
dec["year"] = dec.decay_date.dt.year
ree = pd.crosstab(dec.year, dec.type).reindex(columns=list(TYPE_NAMES.values()), fill_value=0)
ree["Starlink"] = dec[dec.starlink].groupby("year").size().reindex(ree.index, fill_value=0)
print("\n[9] Reentries by year (last 10)\n", ree.tail(10))
life = dec.groupby("type").years_in_orbit.median()
print("Median years in orbit before reentry, by type\n", life.to_string())
print("Median years in orbit, reentered Starlink:", dec[dec.starlink].years_in_orbit.median())
report["reentries"] = {"years": ree.index.tolist(), **{k: ree[k].tolist() for k in list(TYPE_NAMES.values()) + ["Starlink"]}}

# 10. Beyond Earth orbit -----------------------------------------------------------------
beyond = df[df.regime == "Beyond Earth orbit"].copy()
CENTER = {"SU": "The Sun", "MO": "The Moon", "MA": "Mars", "VE": "Venus", "EM": "Earth-Moon barycenter",
          "EL1": "Earth L1", "EL2": "Earth L2", "SS": "Leaving the solar system", "JU": "Jupiter",
          "AS": "An asteroid", "SA": "Saturn", "ME": "Mercury", "CO": "A comet"}
beyond["center"] = beyond.orbit_center.map(CENTER).fillna(beyond.orbit_center)
print("\n[10] Objects whose orbit is not centered on Earth\n", beyond.center.value_counts().to_string())
print(pd.crosstab(beyond.center, beyond.orbit_type))
print("By owner\n", beyond.owner_name.value_counts().head(6).to_string())
print("Leaving the solar system:", beyond[beyond.orbit_center == "SS"].object_name.tolist())
report["beyond"] = beyond.center.value_counts().to_dict()
report["beyond_owner"] = beyond.owner_name.value_counts().head(6).to_dict()

# 11. Launch sites by decade (dashboard and closing notes) ---------------------------------
df["decade"] = df.launch_year // 10 * 10
ls = pd.crosstab(df[df.type == "Payload"].site_name, df[df.type == "Payload"].decade)
print("\n[11] Payloads by launch site and decade (top 10 sites)\n",
      ls.loc[ls.sum(axis=1).sort_values(ascending=False).index[:10]])

# ---------------------------------------------------------------------------
# Every number the two pages quote, in page order
# ---------------------------------------------------------------------------
print("\n" + "=" * 78 + "\nNUMBERS QUOTED IN THE PAGES\n" + "=" * 78)
def q(label, value):
    if isinstance(value, float):
        value = f"{value:,.1f}"
    elif isinstance(value, (int, np.integer)):
        value = f"{int(value):,}"
    print(f"  {label}: {value}")
def pct(a, b):
    return f"{a:,} of {b:,} ({a / b * 100:.1f}%)"

allpay = df[df.type == "Payload"]
eo_all = df[df.in_earth_orbit]
act = df[df.status == "Active"]
print("\nHero and headline numbers")
q("Objects cataloged", len(df))
for t in TYPE_NAMES.values():
    q(f"  {t}", int((df.type == t).sum()))
q("Reentered", pct(int(df.decayed.sum()), len(df)))
q("In Earth orbit", len(eo_all))
q("Active payloads", len(act))
q("Starlink among active payloads", pct(int(act.starlink.sum()), len(act)))
q("Debris in Earth orbit", int((eo_all.type == "Debris").sum()))
q("Payloads in Earth orbit", int((eo_all.type == "Payload").sum()))
q("Rocket bodies in Earth orbit", int((eo_all.type == "Rocket body").sum()))
q("Launch dates", f"{df.launch_date.min().date()} to {df.launch_date.max().date()}")
q("Last reentry date", str(df.decay_date.max().date()))
q("Payloads released from the space station", int(from_station.sum()))

print("\nGrowth")
q("Payloads 1957-2019", int(pay_year.loc[:2019].sum()))
for y in [2019, 2020, 2025, 2026]:
    q(f"Payloads in {y}", int(pay_year.loc[y]))
q("Largest year", f"{pay_year.idxmax()} ({pay_year.max():,})")
q("Payloads 2020 or later", pct(int(pay_year.loc[2020:].sum()), int(pay_year.sum())))

print("\nStarlink")
slp = df[df.starlink]
q("Starlink satellites launched", len(slp))
q("  first launch", str(slp.launch_date.min().date()))
q("  active", int((slp.status == "Active").sum()))
q("  reentered", int(slp.decayed.sum()))
q("  launched in 2019", int(slp[slp.year == 2019].shape[0]))
p19 = allpay[allpay.year >= 2019]
q("Share of payloads launched 2019 on that are Starlink", pct(int(p19.starlink.sum()), len(p19)))
for c in ["US", "PRC", "UK", "CIS"]:
    q(f"Active payloads owned by {c}", int((act.owner == c).sum()))
ow = act[act.owner == "UK"]
q("  UK active that are OneWeb", int(ow.object_name.str.startswith("ONEWEB", na=False).sum()))

print("\nPayloads and debris in Earth orbit (first day of each month, rule appear <= t < decay)")
mon = pd.date_range("1958-01-01", SNAPSHOT, freq="MS")
mrows = []
for t in mon:
    a = in_orbit_at(t)
    mrows.append(((a.type == "Payload").sum(), (a.type == "Debris").sum()))
mdf = pd.DataFrame(mrows, index=mon, columns=["payloads", "debris"])
q("Debris peak (monthly)", f"{mdf.debris.idxmax().date()} ({mdf.debris.max():,})")
q("Debris on 1 January 2022", int(mdf.loc["2022-01-01", "debris"]))
q("Payloads on 1 January 2020", int(mdf.loc["2020-01-01", "payloads"]))
q("Payloads on 1 January 2015", int(mdf.loc["2015-01-01", "payloads"]))
ahead = mdf.payloads > mdf.debris
ch = ahead[ahead != ahead.shift()]
print("  Months in which the lead changes (True = payloads ahead):")
for t, v in ch.items():
    r = mdf.loc[t]
    print(f"    {t.date()}  payloads ahead: {v}  ({r.payloads:,} payloads, {r.debris:,} debris)")
for t in pd.date_range("2024-03-01", "2025-01-01", freq="MS"):
    r = mdf.loc[t]
    print(f"    {t.date()}  payloads {r.payloads:,}  debris {r.debris:,}  lead {r.payloads - r.debris:+,}")
report["in_orbit_monthly"] = {"months": [t.strftime("%Y-%m-%d") for t in mon[mon >= "2015-01-01"]],
                              "payloads": mdf.payloads[mdf.index >= "2015-01-01"].astype(int).tolist(),
                              "debris": mdf.debris[mdf.index >= "2015-01-01"].astype(int).tolist()}

print("\nBreakups")
q("Debris pieces", len(deb))
q("Launches with any debris", len(ev))
q("From the ten largest sources", pct(int(top_ev.debris.sum()), len(deb)))
for i, r in top_ev.iterrows():
    print(f"    {i}  {r['name']:<14} launched {r.launched.date()}  pieces {r.debris:,}  "
          f"in orbit {r.in_orbit:,} ({r.share_in_orbit}%)  median mean altitude of those up {deb[(deb.launch_id == i) & deb.in_orbit].mean_alt.median():.0f} km")
q("Iridium 33 and Cosmos 2251 pieces together",
  int(ev.loc["1993-036", "debris"] + ev.loc["1997-051", "debris"]))

print("\nOwners (payloads by decade)")
for dcd in own.index:
    tot = int(own.loc[dcd].sum())
    print(f"    {dcd}s  total {tot:,}  " + "  ".join(f"{g} {own.loc[dcd, g]:,} ({own_pct.loc[dcd, g]}%)" for g in own.columns))
p20 = pay[(pay.decade == 2020) & (pay.owner == "US")]
q("U.S. payloads of the 2020s that are Starlink", f"{int(p20.starlink.sum()):,} of {len(p20):,}")
nosl = (own_nosl.div(own_nosl.sum(axis=1), axis=0) * 100).round(1).loc[2020]
q("2020s without Starlink", ", ".join(f"{g} {v}%" for g, v in nosl.items()))
q("Distinct owners by decade", ", ".join(f"{d}s {n}" for d, n in nown.items()))
q("1960s: U.S. and USSR payloads", f"{int(own.loc[1960, ['United States', 'Russia (USSR)']].sum()):,} of {int(own.loc[1960].sum()):,}")
q("1970s: USSR payloads", f"{int(own.loc[1970, 'Russia (USSR)']):,} of {int(own.loc[1970].sum()):,}")

print("\nAltitude")
q("In Earth orbit with a cataloged perigee and apogee", len(eo))
q("  of those, with an inclination too (drawn on the globes)", int(eo.inclination.notna().sum()))
q("  missing an inclination", ", ".join(f"{n} ({r})" for n, r in eo[eo.inclination.isna()][["object_name", "regime"]].values))
q("Below 2,000 km mean altitude", pct(len(leo), len(eo)))
b450 = eo[(eo.mean_alt >= 450) & (eo.mean_alt < 500)]
q("450 to 500 km", f"{len(b450):,}, Starlink {int(b450.starlink.sum()):,}")
b700 = eo[(eo.mean_alt >= 700) & (eo.mean_alt < 900)]
q("700 to 900 km", f"{len(b700):,}, debris {int((b700.type == 'Debris').sum()):,}")
hp = eo[eo.perigee >= 35000]
q("Perigee of 35,000 km or more", len(hp))
nc = hp[hp.eccentricity < 0.25]
q("  near-circular (eccentricity below 0.25)", len(nc))
q("  mean altitude within 500 km of 35,786 km", int(((nc.mean_alt - 35786).abs() < 500).sum()))

print("\nInclination (objects in Earth orbit with a known inclination)")
bands = {}
for lo, hi in [(43, 44), (53, 54), (97, 99)]:
    bd = eo_inc[(eo_inc.inclination >= lo) & (eo_inc.inclination < hi)]
    bands[(lo, hi)] = len(bd)
    q(f"{lo} to {hi} degrees", f"{len(bd):,}, Starlink {int(bd.starlink.sum()):,}")
q("The three bands together", pct(sum(bands.values()), len(eo_inc)))

print("\nAge (below 2,000 km, years since entering the catalog)")
print(age.to_string())
for lo, hi in [(400, 500), (700, 900), (1000, 1200), (1500, 2000)]:
    bd = eo_age[(eo_age.mean_alt >= lo) & (eo_age.mean_alt < hi)]
    q(f"Median age {lo} to {hi} km", f"{bd.age.median():.1f} years ({len(bd):,} objects)")
b12 = eo_age[(eo_age.mean_alt >= 1000) & (eo_age.mean_alt < 1200)]
mega = {"OneWeb": "ONEWEB", "Qianfan": "QIANFAN", "Hulianwang": "HULIANWANG"}
tot_mega = 0
for k, pre in mega.items():
    m = b12[b12.object_name.str.startswith(pre, na=False)]
    tot_mega += len(m)
    q(f"  1,000 to 1,200 km, {k}", f"{len(m):,} (entered 2025 or 2026: {int((m.appear.dt.year >= 2025).sum()):,})")
q("  1,000 to 1,200 km, the three together", pct(tot_mega, len(b12)))
q("  1,000 to 1,200 km, entered 2025 or 2026", pct(int((b12.appear.dt.year >= 2025).sum()), len(b12)))
rest = b12[~b12.object_name.str.startswith(tuple(mega.values()), na=False)]
q("  1,000 to 1,200 km without the three", f"median {rest.age.median():.1f} years ({len(rest):,} objects)")
print("  Most common names in 1,000 to 1,200 km:",
      b12.object_name.str.extract(r"^([A-Z]+)")[0].value_counts().head(6).to_dict())
van = df[df.object_name == "VANGUARD 1"].iloc[0]
q("Vanguard 1", f"launched {van.launch_date.date()}, orbit {van.perigee:,.0f} by {van.apogee:,.0f} km")
q("Oldest object still in orbit", eo.sort_values("launch_date").iloc[0].object_name)

print("\nReentries")
for y in [2020, 2025]:
    q(f"Payloads reentered in {y}", int(ree.loc[y, "Payload"]))
q("Starlink reentered in 2025", int(ree.loc[2025, "Starlink"]))
d25 = dec[(dec.year == 2025) & dec.starlink]
q("  median years in orbit of those", float(d25.years_in_orbit.median()))
q("Median years in orbit, every reentered Starlink", float(dec[dec.starlink].years_in_orbit.median()))
q("All objects reentered in 2025", int(ree.loc[2025, list(TYPE_NAMES.values())].sum()))
q("Debris reentries peak", f"{ree.Debris.idxmax()} ({ree.Debris.max():,})")
rbd = dec[dec.type == "Rocket body"]
q("Median days from launch to reentry, rocket bodies", int((rbd.decay_date - rbd.launch_date).dt.days.median()))
globe = df[df.regime.isin(["Low Earth orbit", "Medium Earth orbit", "Geosynchronous and above", "Highly elliptical", "Docked"])
           & df.perigee.notna() & df.apogee.notna() & df.inclination.notna()]
q("Objects on the globe that reentered from 1 January 2024 to the snapshot",
  int((globe.decay_date >= "2024-01-01").sum()))

print("\nBeyond Earth orbit")
q("Objects", len(beyond))
for c, sub in beyond.groupby("center"):
    print(f"    {c}: {len(sub):,}  " + ", ".join(f"{k} {v}" for k, v in sub.type.value_counts().items())
          + "  |  " + ", ".join(f"{k} {v}" for k, v in sub.orbit_type.value_counts().items()))
q("Owned by the United States", int((beyond.owner == "US").sum()))
q("Owned by Russia (USSR)", int((beyond.owner == "CIS").sum()))

# ---------------------------------------------------------------------------
# The slim file both pages load
# ---------------------------------------------------------------------------
TYPE_CODE = {"Payload": "P", "Rocket body": "R", "Debris": "D", "Unknown": "U"}
REGIME_CODE = {"Low Earth orbit": "L", "Medium Earth orbit": "M", "Geosynchronous and above": "G",
               "Highly elliptical": "H", "Beyond Earth orbit": "B", "Docked": "K", "No orbit data": "N"}
site = pd.DataFrame({
    "norad": df.norad_cat_id,
    "name": df.object_name.fillna(""),
    "intl_id": df.object_id,
    "type": df.type.map(TYPE_CODE),
    "owner": df.owner,
    "site": df.launch_site,
    "launch": df.launch_date.dt.strftime("%Y-%m-%d"),
    "appear": df.appear.dt.strftime("%Y-%m-%d"),
    "decay": df.decay_date.dt.strftime("%Y-%m-%d").fillna(""),
    "status": df.status.map({"Active": "A", "Inactive": "I", "No status (debris, stages)": "N", "Reentered": "R"}),
    "regime": df.regime.map(REGIME_CODE),
    "center": df.orbit_center,
    "perigee": df.perigee.round(0).astype("Int64"),
    "apogee": df.apogee.round(0).astype("Int64"),
    "inclination": df.inclination.round(2),
    "period": df.period.round(2),
    "starlink": df.starlink.astype(int),
})
site.to_csv(DATA / "satcat_site.csv", index=False, lineterminator="\n")
report["breakup_ids"] = [e["launch_id"] for e in report["breakups"]]
report["owner_names"] = {c: n for c, n in df[["owner", "owner_name"]].drop_duplicates().values}
report["site_names"] = {c: n for c, n in df[["launch_site", "site_name"]].drop_duplicates().values}
# For the objects sent to the Moon, CelesTrak's orbit type says whether each one struck
# the surface (IMP), landed (LAN) or went into lunar orbit (ORB); the report's Moon cards use it.
moon = df[df.orbit_center == "MO"]
report["moon_kind"] = {str(k): v for k, v in moon[["norad_cat_id", "orbit_type"]].values}
print("\nMOON OBJECTS BY ORBIT TYPE", moon.orbit_type.value_counts().to_dict())
(DATA / "report_numbers.json").write_text(json.dumps(report, indent=1), newline="\n")
print(f"\nWrote data/satcat_site.csv ({len(site):,} rows, {site.shape[1]} columns)")

# ---------------------------------------------------------------------------
# Land mask for the globe: 1024 x 512 equirectangular, white = land
# ---------------------------------------------------------------------------
W, H = 1024, 512
img = Image.new("L", (W, H), 0)
draw = ImageDraw.Draw(img)
land = json.loads((DATA / "ne_50m_land.geojson").read_text(encoding="utf-8"))
def ring_px(ring):
    return [((lon + 180) / 360 * W, (90 - lat) / 180 * H) for lon, lat in ring]
for feat in land["features"]:
    g = feat["geometry"]
    polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
    for poly in polys:
        draw.polygon(ring_px(poly[0]), fill=255)
        for hole in poly[1:]:
            draw.polygon(ring_px(hole), fill=0)
img.save(DATA / "land_mask.png", optimize=True)
print(f"Wrote data/land_mask.png ({(DATA / 'land_mask.png').stat().st_size:,} bytes)")

# ---------------------------------------------------------------------------
# Copy the data into the two pages
# ---------------------------------------------------------------------------
def b64gz(b):
    return base64.b64encode(gzip.compress(b, 9, mtime=0)).decode("ascii")

blobs = {
    "SATCAT": b64gz((DATA / "satcat_site.csv").read_bytes()),
    "REPORT": b64gz((DATA / "report_numbers.json").read_bytes()),
    "LAND": base64.b64encode((DATA / "land_mask.png").read_bytes()).decode("ascii"),
}
for page in ["index.html", "dashboard.html"]:
    p = ROOT / page
    if not p.exists():
        continue
    html = p.read_text(encoding="utf-8")
    for key, blob in blobs.items():
        html, n = re.subn(rf'(<script type="text/plain" id="data-{key.lower()}">)[^<]*(</script>)',
                          lambda m: m.group(1) + blob + m.group(2), html)
        if n:
            print(f"{page}: {key} {len(blob):,} characters")
    p.write_text(html, encoding="utf-8", newline="\n")
