# Everything in Orbit

A two-page website about the public satellite catalog: every object that has
been tracked in orbit around Earth, from Sputnik in October 1957 to September
2026. The report page (`index.html`) is a scrolling set of findings built
around a WebGL globe that draws each of the 34,500 objects in Earth orbit on
its cataloged orbit. The dashboard page (`dashboard.html`) lets the reader
filter all 70,745 objects and recomputes every number, chart and table row in
the browser.

By Alejandro Pacheco, designed and built with Claude Opus 5.5.

Once the repository is public with GitHub Pages serving the `main` branch,
the site is at https://apach008.github.io/space/.

## Files

| Path | What it is |
|---|---|
| `index.html` | The report page: title, headline numbers, ten findings with a chart each, the scroll-driven timeline from 1957 to 2026, and the notes on the data. The page carries its own copy of the data (see below), so it opens directly from disk. |
| `dashboard.html` | The dashboard page: filters for launch year, owner, launch site, object type, orbit and status; five summary numbers; a measure switch and a breakdown switch; a yearly chart, a bar chart, a decade heatmap, an altitude and inclination scatter plot, a globe of the filtered objects; and a sortable table of the numbers behind the view with a CSV download. The current view is kept in the page address. |
| `fetch_data.py` | Downloads the raw files into `data/`. Running it again replaces the snapshot with the current catalog. |
| `prepare_data.py` | Labels the catalog, prints every table the report quotes, writes the three built files below, and copies them into both HTML pages between the `data-satcat`, `data-report` and `data-land` script tags. |
| `data/satcat.csv` | The raw satellite catalog (SATCAT) as downloaded from CelesTrak on September 23, 2026: 70,745 rows and 17 columns, one row per cataloged object. |
| `data/owner_codes.csv` | CelesTrak's owner codes and what they stand for. |
| `data/launch_site_codes.csv` | CelesTrak's launch site codes and what they stand for. |
| `data/ne_50m_land.geojson` | Natural Earth 1:50m land outlines, used only to draw the continents on the globes. |
| `data/satcat_site.csv` | Built by `prepare_data.py`: the 17 columns both pages use, one row per object, including the orbit regime and the estimated date each object entered the catalog. |
| `data/report_numbers.json` | Built by `prepare_data.py`: the yearly, decade and altitude tables behind the report charts. |
| `data/land_mask.png` | Built by `prepare_data.py`: a 1024 by 512 land and sea mask the globe shaders read. |
| `.gitattributes`, `.gitignore` | Keep line endings consistent and keep scratch files out of the repository. |

## Where the data comes from

The satellite catalog is published by CelesTrak at
https://celestrak.org/pub/satcat.csv and is built from the U.S. Space Force
catalog released on Space-Track.org. The owner and launch site code tables
come from https://celestrak.org/satcat/sources.php and
https://celestrak.org/satcat/launchsites.php. The land outlines are Natural
Earth's public domain 1:50m land file from
https://github.com/nvkelso/natural-earth-vector.

## Viewing the site

Open `index.html` in a browser. No server is needed, because the data is
stored inside each page as a compressed, base64-encoded copy of
`data/satcat_site.csv`, which the page unpacks with the browser's built-in
`DecompressionStream`. The pages load their libraries and fonts over the
internet: three.js 0.186.0, GSAP 3.15.0 (with ScrollTrigger and SplitText),
Lenis 1.3.26 and D3 7.9.0 from jsDelivr, and the Anybody and Newsreader
fonts from Google Fonts. Both pages are designed for desktop screens.

## Rebuilding the data

    python fetch_data.py      # optional: replaces the snapshot with today's catalog
    python prepare_data.py    # rebuilds data/ and the data inside both pages

The scripts need pandas, lxml and Pillow. After a new snapshot, the numbers
written in the report text have to be updated by hand from the tables
`prepare_data.py` prints.

## Definitions

The notes at the end of the report page give every definition used. In short:
an object is in Earth orbit when it has no decay date and its orbit is
centered on Earth; an active payload has operational status +, P, B, S or X;
a Starlink satellite is one whose name starts with STARLINK; mean altitude is
the average of apogee and perigee; debris is dated by the estimated day it
entered the catalog rather than by its parent's launch date; and the 416
payloads released from the International Space Station are counted in the
year they entered the catalog instead of the station's 1998 launch year. The
globes draw each object on an ellipse with its cataloged apogee, perigee and
inclination, with the orbit's orientation and the object's position along it
assigned at random, since the catalog does not record them.
