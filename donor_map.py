"""
donor_map_blended.py  –  Quincy donor scores with
 • on-disk geocode cache
 • per-street statistics
 • spatial *blending* of nearby streets into one larger marker
"""

###############################################################################
# Imports & config
###############################################################################
import ssl, base64, pickle, numpy as np, pandas as pd, matplotlib.pyplot as plt
from io import BytesIO; from pathlib import Path; from time import sleep
from tqdm.auto import tqdm
import folium, branca.colormap as cm
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError

EXCEL_FILE   = Path("data/donor_scores.xlsx")
OUTPUT_HTML  = Path("donor_scores_map_preview.html")
ADDRESS_COL  = "contributor_street_1"
SCORE_COL    = "donor_score"

SAMPLE_ROWS     = 1000        # None → full run
BLEND_PRECISION = 2          # 3 dec ≈ 111 m; 4 dec ≈ 11 m; None → no blending

CACHE_FILE = Path("street_cache.pkl")
GEOCODE_TIMEOUT_SECS = 10
RETRIES, BACKOFFS    = 3, [2, 5, 10]

###############################################################################
# Helper – tiny persistent cache ------------------------------------------------
###############################################################################
def load_cache() -> dict:          # street -> (lat,lon)
    if CACHE_FILE.exists():
        try:
            with CACHE_FILE.open("rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {}

def save_cache(cache: dict):
    with CACHE_FILE.open("wb") as f:
        pickle.dump(cache, f)

cache = load_cache()

###############################################################################
# 1. Read + clean data
###############################################################################
df = pd.read_excel(EXCEL_FILE, nrows=SAMPLE_ROWS or None)

df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="coerce")
df = df[np.isfinite(df[SCORE_COL])]                  # drop NaN / inf scores

df["street"] = (df[ADDRESS_COL]
                .astype(str)
                .str.replace(r"^\d+\s+", "", regex=True, n=1)
                .str.strip())

###############################################################################
# 2. Geocode (cached + courteous retry)
###############################################################################
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
geo  = Nominatim(user_agent="quincy_donor_mapper", timeout=GEOCODE_TIMEOUT_SECS,
                 ssl_context=ctx)
rate = RateLimiter(geo.geocode, min_delay_seconds=1, swallow_exceptions=True)

def gcode(street: str):
    if street in cache:
        return cache[street]

    for a in range(RETRIES):
        try:
            loc = rate(f"{street}, Quincy, MA")
            if loc:
                cache[street] = (loc.latitude, loc.longitude)
                return cache[street]
        except (GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError):
            pass
        if a < RETRIES-1: sleep(BACKOFFS[a])

    cache[street] = (np.nan, np.nan)
    return cache[street]

u_streets = df["street"].drop_duplicates()
coords    = [gcode(s) for s in tqdm(u_streets, desc="Geocoding streets")]
df = df.join(pd.DataFrame(coords, index=u_streets, columns=["lat","lon"]), on="street")
df = df.dropna(subset=["lat","lon"])

###############################################################################
# 3. Aggregate per *street* first
###############################################################################
street_stats = (df.groupby(["street","lat","lon"])
                  [SCORE_COL]
                  .agg(count="size",
                       median="median",
                       mean="mean",
                       scores=list)
                  .reset_index())

###############################################################################
# 4. BLEND spatially close streets  (grid-snap trick)
###############################################################################
if BLEND_PRECISION is not None:
    street_stats["grid_lat"] = street_stats["lat"].round(BLEND_PRECISION)
    street_stats["grid_lon"] = street_stats["lon"].round(BLEND_PRECISION)

    blended = (street_stats
                 .groupby(["grid_lat","grid_lon"], as_index=False)
                 .agg(lat   = ("lat",   "mean"),
                      lon   = ("lon",   "mean"),
                      count = ("count", "sum"),
                      median_scores = ("scores", lambda L: np.median(np.hstack(L))),
                      mean_scores   = ("scores", lambda L: np.mean(np.hstack(L))),
                      all_scores    = ("scores", lambda L: np.hstack(L).tolist()),
                      streets       = ("street", lambda L: list(L))))
    blended.rename(columns={"median_scores":"median",
                            "mean_scores":"mean"}, inplace=True)
else:
    blended = street_stats.copy()
    blended["streets"] = blended["street"]

###############################################################################
# 5. Visual scales
###############################################################################
vmin, vmax = blended["median"].min(), blended["median"].max()
if vmin == vmax: vmin -= 1; vmax += 1
cmap   = cm.linear.YlOrRd_09.scale(vmin, vmax)
radius = lambda n: 4 + 1.8*np.sqrt(n)

###############################################################################
# 6. Map
###############################################################################
city = geo.geocode("Quincy, MA") or (42.2529, -71.0023)
m = folium.Map(location=[getattr(city,"latitude",city[0]),
                         getattr(city,"longitude",city[1])],
               zoom_start=13)

for _, r in tqdm(blended.iterrows(), total=len(blended), desc="Adding markers"):
    scores = np.asarray(r["all_scores"], dtype=float)

    # inline histogram
    fig, ax = plt.subplots()
    ax.hist(scores, bins="auto")
    ax.set_title(", ".join(sorted(set(r["streets"]))[:3]) + ("…" if len(r["streets"])>3 else ""))
    ax.set_xlabel("Donor Score"); ax.set_ylabel("Freq"); fig.tight_layout()
    buf = BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    img64 = base64.b64encode(buf.getvalue()).decode()

    html = (
        f"<h4>{len(r['streets'])} nearby street(s)</h4>"
        f"<b>Total contributors:</b> {r['count']}<br>"
        f"<b>Median score:</b> {r['median']:.2f}<br>"
        f"<b>Mean score:</b> {r['mean']:.2f}<br>"
        f"<small>{', '.join(sorted(set(r['streets']))[:8])}"
        f"{' …' if len(r['streets'])>8 else ''}</small><br><br>"
        f"<img src='data:image/png;base64,{img64}' width='300' height='200'>"
    )

    folium.CircleMarker(
        location=(r["lat"], r["lon"]),
        radius=radius(r["count"]),
        color="black", weight=0.8,
        fill=True, fill_color=cmap(r["median"]), fill_opacity=0.85,
        popup=folium.Popup(folium.IFrame(html, width=320, height=420),
                           max_width=500)
    ).add_to(m)

cmap.caption = "Median donor score"; cmap.add_to(m)

###############################################################################
# 7. Save map & cache
###############################################################################
m.save(OUTPUT_HTML)
save_cache(cache)

print(f"✓ Map with blended markers saved to {OUTPUT_HTML.resolve()}  "
      f"({len(df)} original rows → {len(blended)} blended markers ; "
      f"cache size = {len(cache)} streets)")
print("Adjust BLEND_PRECISION or set it to None to see the effect, "
      "then set SAMPLE_ROWS=None for a full-dataset run.")
