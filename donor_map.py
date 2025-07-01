"""
donor_map.py  –  visualises Quincy donor statistics

🔹 Colour  = median score (light -> high)  
🔹 Size    = number of contributors on that street
"""

###############################################################################
# Imports
###############################################################################
import ssl, base64, numpy as np, pandas as pd, matplotlib.pyplot as plt
from io import BytesIO
from pathlib import Path
from time import sleep
from tqdm.auto import tqdm
import folium, branca.colormap as cm
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError

###############################################################################
# CONFIG
###############################################################################
EXCEL_FILE   = Path("data/donor_scores.xlsx")
OUTPUT_HTML  = Path("donor_scores_map_preview.html")
ADDRESS_COL  = "contributor_street_1"
SCORE_COL    = "donor_score"

SAMPLE_ROWS  = 500      # None → full dataset
GEOCODE_TIMEOUT_SECS = 10
RETRIES, BACKOFFS     = 3, [2, 5, 10]

###############################################################################
# LOAD
###############################################################################
df = pd.read_excel(EXCEL_FILE, nrows=SAMPLE_ROWS or None)
df["street"] = (df[ADDRESS_COL]
                .astype(str)
                .str.replace(r"^\d+\s+", "", regex=True, n=1)
                .str.strip())

###############################################################################
# GEOCODE (w/ progress)
###############################################################################
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
geolocator   = Nominatim(user_agent="quincy_donor_mapper", timeout=GEOCODE_TIMEOUT_SECS,
                         ssl_context=ctx)
rate_limited = RateLimiter(geolocator.geocode, min_delay_seconds=1, swallow_exceptions=True)

def gcode(street):
    for a in range(RETRIES):
        try:
            loc = rate_limited(f"{street}, Quincy, MA")
            if loc:
                return loc.latitude, loc.longitude
        except (GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError):
            pass
        if a < RETRIES-1: sleep(BACKOFFS[a])
    return np.nan, np.nan

ustreets = df["street"].drop_duplicates()
coords   = [gcode(s) for s in tqdm(ustreets, desc="Geocoding streets")]
df = df.join(pd.DataFrame(coords, index=ustreets, columns=["lat","lon"]), on="street")
df = df.dropna(subset=["lat","lon"])

###############################################################################
# AGGREGATE STATS
###############################################################################
agg = (df.groupby(["street","lat","lon"])
         [SCORE_COL]
         .agg(count="size",
              median="median",
              mean="mean",
              scores=list)
         .reset_index())

###############################################################################
# COLOUR + SIZE SCALES
###############################################################################
colormap = cm.linear.YlOrRd_09.scale(agg["median"].min(), agg["median"].max())
def colour(m): return colormap(m)
def radius(n): return 4 + 1.5*n   # √n feels good on a map

###############################################################################
# BUILD MAP
###############################################################################
city = geolocator.geocode("Quincy, MA") or (42.2529, -71.0023)
m = folium.Map(location=[getattr(city,"latitude",city[0]),
                         getattr(city,"longitude",city[1])],
               zoom_start=13)

for _, r in tqdm(agg.iterrows(), total=len(agg), desc="Adding markers"):
    scores = np.array(r["scores"], dtype=float)
    fig, ax = plt.subplots()
    ax.hist(scores, bins="auto"); ax.set_xlabel("Score"); ax.set_ylabel("Freq")
    ax.set_title(f"{r['street']}"); fig.tight_layout()
    buf = BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    img64 = base64.b64encode(buf.getvalue()).decode()

    popup_html = (
        f"<h4>{r['street']}</h4>"
        f"<b>Count:</b> {r['count']}<br>"
        f"<b>Median:</b> {r['median']:.2f}<br>"
        f"<b>Mean:</b> {r['mean']:.2f}<br><br>"
        f"<img src='data:image/png;base64,{img64}' width='300' height='200'>"
    )

    folium.CircleMarker(
        location=(r["lat"], r["lon"]),
        radius=radius(r["count"]),
        color="black", weight=0.8,
        fill=True, fill_color=colour(r["median"]), fill_opacity=0.85,
        popup=folium.Popup(folium.IFrame(popup_html, width=320, height=420),
                           max_width=500)
    ).add_to(m)

# add legend
colormap.caption = "Median donor score"
colormap.add_to(m)

###############################################################################
# SAVE
###############################################################################
m.save(OUTPUT_HTML)
print(f"✓ Map saved to {OUTPUT_HTML.resolve()}  "
      f"({len(df)} rows → {len(agg)} streets)")
print("Swap SAMPLE_ROWS=None for the full run when you’re happy.")
