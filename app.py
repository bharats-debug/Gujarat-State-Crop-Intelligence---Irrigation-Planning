"""
India State Crop Intelligence & Irrigation Planning Platform
=============================================================
"""

from __future__ import annotations
import datetime as dt

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(
    page_title="India Crop Intelligence & Irrigation Planning Platform",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLOR_PRIMARY   = "#2E7D32"
COLOR_SECONDARY = "#EF6C00"
COLOR_CRITICAL  = "#C62828"
COLOR_BG        = "var(--background-color)"
COLOR_SURFACE   = "var(--secondary-background-color)"
COLOR_TEXT      = "var(--text-color)"
COLOR_MUTED     = "var(--text-color)"
COLOR_BORDER    = "rgba(128,128,128,0.25)"

STATUS_COLORS = {"Optimal": COLOR_PRIMARY, "Moderate": COLOR_SECONDARY, "Critical": COLOR_CRITICAL}

CROP_COLORS = {
    "Wheat": "#1565C0", "Cumin": "#6A1B9A", "Mustard": "#F9A825",
    "Castor": "#EF6C00", "Cotton": "#00897B", "Rice": "#558B2F",
    "Sugarcane": "#AD1457", "Soybean": "#6D4C41", "Fallow": "#9E9E9E",
}

STAGE_ORDER         = ["Sowing", "Early Vegetative", "Vegetative", "Flowering", "Maturity"]
CROPS_MAIN          = ["Cumin", "Mustard", "Castor", "Cotton", "Wheat", "Rice", "Sugarcane", "Soybean"]
CROP_FILTER_OPTIONS = ["All Crops"] + CROPS_MAIN + ["Fallow"]
KHARIF_CROPS        = ["Rice", "Cotton", "Soybean", "Castor"]
RABI_CROPS          = ["Wheat", "Cumin", "Mustard"]

st.markdown(f"""<style>
    .block-container {{ padding-top:1.6rem; padding-bottom:2.5rem; max-width:1400px; }}
    div[data-testid="stMetric"] {{
        background-color:{COLOR_SURFACE}; border:1px solid {COLOR_BORDER};
        border-radius:10px; padding:1.1rem 1.3rem 0.9rem 1.3rem;
    }}
    div[data-testid="stMetricLabel"] {{ opacity:0.75; font-weight:600; }}
    button[data-baseweb="tab"] {{ font-size:1.0rem; font-weight:600; opacity:0.7; }}
    button[data-baseweb="tab"][aria-selected="true"] {{ opacity:1; color:{COLOR_PRIMARY}; border-bottom-color:{COLOR_PRIMARY}!important; }}
    div[data-baseweb="tab-highlight"] {{ background-color:{COLOR_PRIMARY}!important; }}
    .control-card {{ background-color:{COLOR_SURFACE}; border:1px solid {COLOR_BORDER}; border-radius:12px; padding:1.4rem 1.4rem 0.6rem 1.4rem; }}
    .legend-chip {{ display:inline-flex; align-items:center; gap:0.4rem; font-size:0.85rem; opacity:0.85; margin-right:1.1rem; }}
    .legend-dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
    .app-header-eyebrow {{ color:{COLOR_PRIMARY}; font-weight:700; font-size:0.85rem; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:-0.4rem; }}
    .stButton button {{ background-color:{COLOR_PRIMARY}; color:white; border:none; border-radius:8px; font-weight:600; }}
    .stButton button:hover {{ background-color:#245F27; color:white; }}
    .detail-card {{ background-color:{COLOR_SURFACE}; border:1px solid {COLOR_BORDER}; border-left:5px solid {COLOR_PRIMARY}; border-radius:8px; padding:1rem 1.2rem; margin-top:0.8rem; }}
    .detail-card-title {{ color:{COLOR_PRIMARY}; font-weight:700; font-size:1rem; margin-bottom:0.4rem; }}
    .meta-banner {{ background-color:#0F2027; background-image:linear-gradient(90deg,#14302B 0%,#173A2E 100%); border:1px solid #1F4D3C; border-radius:10px; padding:0.75rem 1.2rem; margin-bottom:0.4rem; }}
    .meta-banner span {{ color:#D7F3E3; font-size:0.82rem; font-family:"Source Code Pro","Consolas",monospace; }}
    .meta-banner .dot-live {{ color:#66FF99; font-weight:700; }}
    .conf-high {{ color:#43A047; font-weight:700; }}
    .conf-medium {{ color:#FB8C00; font-weight:700; }}
    .conf-low {{ color:#E53935; font-weight:700; }}
    .geo-status {{ font-size:0.80rem; opacity:0.85; padding:0.3rem 0; margin-bottom:0.2rem; }}
    div[data-testid="stFormSubmitButton"] button {{
        width: 100%;
        height: 2.7rem;
        padding: 0 !important;
        font-size: 1.05rem;
        border-radius: 8px;
        background-color: {COLOR_PRIMARY};
        color: white;
        border: none;
    }}
    div[data-testid="stFormSubmitButton"] button:hover {{
        background-color: #245F27;
        color: white;
    }}
    div[data-testid="stForm"] {{
        border: none;
        padding: 0;
    }}
</style>""", unsafe_allow_html=True)

ZONES = {
    "Western Himalayan Zone (J&K / HP / Uttarakhand)":
        {"center":(32.50,76.00),"seed":11},
    "Trans-Gangetic Plain (Punjab / Haryana / W-UP)":
        {"center":(30.60,75.90),"seed":22},
    "Upper Gangetic Plain (Central UP / Uttarakhand Terai)":
        {"center":(27.10,80.20),"seed":33},
    "Middle Gangetic Plain (Bihar / E-UP)":
        {"center":(25.60,85.10),"seed":44},
    "Lower Gangetic Plain (West Bengal)":
        {"center":(23.30,88.40),"seed":55},
    "Eastern Plateau & Hills (Jharkhand / Chhattisgarh / Odisha)":
        {"center":(22.00,84.50),"seed":66},
    "Central Plateau & Hills (MP / SE Rajasthan)":
        {"center":(23.80,77.50),"seed":77},
    "Western Dry Zone (Rajasthan Desert)":
        {"center":(27.00,72.00),"seed":88},
    "Gujarat Plains & Hills (Narmada / Mahi Command)":
        {"center":(22.80,72.80),"seed":99},
    "Deccan Plateau (Maharashtra / N-Karnataka / Telangana)":
        {"center":(18.00,76.50),"seed":111},
    "Eastern Coastal Plains (Andhra / Odisha Coast)":
        {"center":(16.50,80.60),"seed":122},
    "Western Coastal Plains (Konkan / Malabar)":
        {"center":(15.50,73.80),"seed":133},
    "Southern Plateau & Hills (S-Karnataka / Tamil Nadu Upland)":
        {"center":(12.50,77.20),"seed":144},
    "Northeast Hills (Assam Valley / NE States)":
        {"center":(26.10,92.00),"seed":155},
    "Island Territories (A&N / Lakshadweep)":
        {"center":(11.70,92.70),"seed":166},
}

ZONE_SHAPEFILE_PATH     = "zone_data/Agroclimatic_regions.shp"
STATE_SHAPEFILE_PATH    = "state_data/India_State_Boundary.shp"
DISTRICT_SHAPEFILE_PATH = "district_data/India_District_Boundary.shp"

@st.cache_data(show_spinner=False)
def load_local_datasets():
    import geopandas as gpd

    def _safe_load(path: str, label: str) -> "gpd.GeoDataFrame":
        try:
            gdf = gpd.read_file(path)
            if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            elif gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            return gdf
        except Exception as e:
            st.session_state.setdefault("_local_dataset_errors", []).append(
                f"{label}: could not load '{path}' ({type(e).__name__}: {e})"
            )
            return gpd.GeoDataFrame()

    st.session_state["_local_dataset_errors"] = []
    acz       = _safe_load(ZONE_SHAPEFILE_PATH,     "Agro-climatic zones")
    states    = _safe_load(STATE_SHAPEFILE_PATH,    "States")
    districts = _safe_load(DISTRICT_SHAPEFILE_PATH, "Districts")
    return acz, states, districts


ACZ_GDF, STATES_GDF, DISTRICTS_GDF = load_local_datasets()

ZONE_NAME_TO_REGIONCODE = {
    "Western Himalayan Zone (J&K / HP / Uttarakhand)":              1,
    "Trans-Gangetic Plain (Punjab / Haryana / W-UP)":                2,
    "Upper Gangetic Plain (Central UP / Uttarakhand Terai)":         3,
    "Middle Gangetic Plain (Bihar / E-UP)":                          4,
    "Lower Gangetic Plain (West Bengal)":                            5,
    "Eastern Plateau & Hills (Jharkhand / Chhattisgarh / Odisha)":   6,
    "Central Plateau & Hills (MP / SE Rajasthan)":                   7,
    "Western Dry Zone (Rajasthan Desert)":                           8,
    "Gujarat Plains & Hills (Narmada / Mahi Command)":               9,
    "Deccan Plateau (Maharashtra / N-Karnataka / Telangana)":       10,
    "Eastern Coastal Plains (Andhra / Odisha Coast)":               11,
    "Western Coastal Plains (Konkan / Malabar)":                    12,
    "Southern Plateau & Hills (S-Karnataka / Tamil Nadu Upland)":   13,
    "Northeast Hills (Assam Valley / NE States)":                   14,
    "Island Territories (A&N / Lakshadweep)":                       15,
}

NOMINATIM_UA = "IndiaAgriDashboard_BharatiyaAntarikshHackathon_v3"


def compute_real_area_km2(geom) -> float | None:
    if geom is None:
        return None
    try:
        import geopandas as gpd
        gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs="EPSG:4326")
        gdf_proj = gdf.to_crs("EPSG:6933")
        return float(gdf_proj.geometry.area.iloc[0]) / 1_000_000.0
    except Exception:
        return None


def _local_boundary_lookup(query: str) -> dict | None:
    import geopandas as gpd
    from shapely.geometry import mapping

    clean_query = query.strip().lower()

    try:
        if not STATES_GDF.empty and "STATE" in STATES_GDF.columns:
            match = STATES_GDF[STATES_GDF["STATE"].astype(str).str.lower() == clean_query]
            if not match.empty:
                row      = match.iloc[0]
                geom     = row.geometry
                area_km2 = compute_real_area_km2(geom)
                minx, miny, maxx, maxy = geom.bounds
                return {
                    "geojson":      mapping(geom),
                    "bbox":         [miny, maxy, minx, maxx],
                    "display_name": f"State: {row['STATE']}",
                    "centroid":     (geom.centroid.y, geom.centroid.x),
                    "_geom":        geom,
                    "source":       "local-shapefile-state",
                    "area_km2":     area_km2,
                }
    except Exception:
        pass

    try:
        if not DISTRICTS_GDF.empty and "District" in DISTRICTS_GDF.columns:
            match = DISTRICTS_GDF[DISTRICTS_GDF["District"].astype(str).str.lower() == clean_query]
            if not match.empty:
                row      = match.iloc[0]
                geom     = row.geometry
                state    = row["STATE"] if "STATE" in DISTRICTS_GDF.columns else ""
                area_km2 = compute_real_area_km2(geom)
                minx, miny, maxx, maxy = geom.bounds
                return {
                    "geojson":      mapping(geom),
                    "bbox":         [miny, maxy, minx, maxx],
                    "display_name": f"District: {row['District']} ({state})",
                    "centroid":     (geom.centroid.y, geom.centroid.x),
                    "_geom":        geom,
                    "source":       "local-shapefile-district",
                    "area_km2":     area_km2,
                }
    except Exception:
        pass

    return None


@st.cache_data(show_spinner=False, ttl=3600)
def find_boundary(query: str) -> dict | None:
    local_hit = _local_boundary_lookup(query)
    if local_hit is not None:
        return local_hit

    try:
        import osmnx as ox
        import geopandas as gpd
        from shapely.geometry import mapping, Point as SPoint
    except ImportError as e:
        st.session_state._boundary_errors = [
            f"Missing library: {e}. Run: pip install osmnx shapely geopandas"
        ]
        return None

    ox.settings.user_agent = NOMINATIM_UA
    errors  = []
    lat_ref = None

    for q in [query, f"{query}, India"]:
        try:
            gdf = ox.geocode_to_gdf(q)
            if gdf is not None and not gdf.empty:
                geom = gdf.iloc[0]["geometry"]
                if geom.geom_type in ["Polygon", "MultiPolygon"]:
                    name    = str(gdf.iloc[0].get("display_name", q))
                    lat_ref = float(gdf.iloc[0].get("lat", geom.centroid.y))
                    lon_ref = float(gdf.iloc[0].get("lon", geom.centroid.x))
                    minx, miny, maxx, maxy = geom.bounds
                    return {
                        "geojson":      mapping(geom),
                        "bbox":         [miny, maxy, minx, maxx],
                        "display_name": name[:80],
                        "centroid":     (geom.centroid.y, geom.centroid.x),
                        "_geom":        geom,
                        "source":       "osmnx-step1",
                        "area_km2":     compute_real_area_km2(geom),
                    }
                else:
                    lat_ref = float(gdf.iloc[0].get("lat", geom.y))
                    lon_ref = float(gdf.iloc[0].get("lon", geom.x))
        except Exception as e:
            errors.append(f"Step1 OSMnx failed for '{q}': {type(e).__name__}: {e}")

    for q in [query, f"{query}, India", f"{query} district, India"]:
        try:
            resp = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q":               q,
                    "format":          "json",
                    "polygon_geojson": 1,
                    "limit":           1,
                    "countrycodes":    "in",
                    "addressdetails":  1,
                },
                headers={"User-Agent": NOMINATIM_UA},
                timeout=15,
            )

            if resp.status_code == 429:
                errors.append("Step2: Nominatim rate-limiting (429) — wait 1 min and retry")
                continue
            if resp.status_code == 403:
                errors.append("Step2: Nominatim blocked (403) — user-agent rejected")
                continue
            if resp.status_code != 200:
                errors.append(f"Step2: Nominatim HTTP {resp.status_code}")
                continue

            data = resp.json()
            if not data:
                errors.append(f"Step2: No Nominatim result for '{q}'")
                continue

            res         = data[0]
            place_name  = res.get("display_name", q)
            lat_ref     = float(res["lat"])
            lon_ref     = float(res["lon"])
            geojson_raw = res.get("geojson")

            if geojson_raw and geojson_raw["type"] in ["Polygon", "MultiPolygon"]:
                gdf_step2 = gpd.GeoDataFrame.from_features([{
                    "type":       "Feature",
                    "geometry":   geojson_raw,
                    "properties": {"display_name": place_name},
                }])
                geom = gdf_step2.iloc[0]["geometry"]
                minx, miny, maxx, maxy = geom.bounds
                return {
                    "geojson":      mapping(geom),
                    "bbox":         [miny, maxy, minx, maxx],
                    "display_name": place_name[:80],
                    "centroid":     (lat_ref, lon_ref),
                    "_geom":        geom,
                    "source":       "nominatim-polygon",
                    "area_km2":     compute_real_area_km2(geom),
                }
            else:
                errors.append(
                    f"Step2: Nominatim returned a point for '{q}' "
                    f"(type={geojson_raw.get('type') if geojson_raw else 'None'}) — trying Step 3"
                )

        except Exception as e:
            errors.append(f"Step2 Nominatim error for '{q}': {type(e).__name__}: {e}")

    if lat_ref is not None:
        try:
            tags = {"boundary": "administrative"}
            features = ox.features_from_point(
                (lat_ref, lon_ref), tags=tags, dist=5000
            )

            if features is not None and not features.empty:
                point_geom = SPoint(lon_ref, lat_ref)

                containing = features[
                    features.geometry.apply(
                        lambda g: g.geom_type in ["Polygon","MultiPolygon"]
                                  and g.contains(point_geom)
                    )
                ]

                candidates = containing if not containing.empty else features
                candidates = candidates[
                    candidates.geometry.apply(
                        lambda g: g.geom_type in ["Polygon","MultiPolygon"]
                    )
                ]

                if not candidates.empty:
                    candidates = candidates.copy()
                    candidates["_area"] = candidates.geometry.area
                    candidates = candidates.sort_values("_area")
                    row  = candidates.iloc[0]
                    geom = row["geometry"]
                    name = str(row.get("name", query))
                    minx, miny, maxx, maxy = geom.bounds
                    return {
                        "geojson":      mapping(geom),
                        "bbox":         [miny, maxy, minx, maxx],
                        "display_name": name[:80],
                        "centroid":     (lat_ref, lon_ref),
                        "_geom":        geom,
                        "source":       "osmnx-features-step3",
                        "area_km2":     compute_real_area_km2(geom),
                    }
                else:
                    errors.append("Step3: No polygon features found near point")
        except Exception as e:
            errors.append(f"Step3 features_from_point error: {type(e).__name__}: {e}")
    else:
        errors.append("Step3: Skipped — no lat/lon reference from Steps 1 or 2")

    st.session_state._boundary_errors = errors
    return None


@st.cache_data(show_spinner=False)
def get_zone_boundary(zone_name: str) -> dict | None:
    from shapely.geometry import mapping

    if ACZ_GDF.empty or "regioncode" not in ACZ_GDF.columns:
        return None

    target_code = ZONE_NAME_TO_REGIONCODE.get(zone_name)
    if target_code is None:
        return None

    try:
        codes_clean = ACZ_GDF["regioncode"].astype(str).str.strip().str.lower()
        exact = ACZ_GDF[codes_clean == str(target_code).strip().lower()]
        if exact.empty:
            return None
        row  = exact.iloc[0]
        geom = row.geometry

        area_km2 = None
        if "area_ha" in ACZ_GDF.columns:
            try:
                area_km2 = float(row["area_ha"]) / 100.0
            except (TypeError, ValueError):
                area_km2 = None
        if area_km2 is None:
            area_km2 = compute_real_area_km2(geom)

        minx, miny, maxx, maxy = geom.bounds
        attrs = {
            "regioncode": row.get("regioncode", "N/A"),
            "regionname": row.get("regionname", zone_name),
            "state":      row.get("state", "—"),
            "soil":       row.get("soil", "—"),
            "majorcrops": row.get("majorcrops", "—"),
            "avgann_rf":  row.get("avgann_rf", "—"),
            "avgtmp_jan": row.get("avgtmp_jan", "—"),
            "avgtmp_jul": row.get("avgtmp_jul", "—"),
            "remarks":    row.get("remarks", "—"),
            "area_ha":    row.get("area_ha", "—"),
        }
        return {
            "geojson":      mapping(geom),
            "bbox":         [miny, maxy, minx, maxx],
            "display_name": str(row.get("regionname", zone_name)),
            "centroid":     (geom.centroid.y, geom.centroid.x),
            "_geom":        geom,
            "source":       "local-shapefile-zone",
            "attributes":   attrs,
            "area_km2":     area_km2,
        }
    except Exception as e:
        st.session_state.setdefault("_local_dataset_errors", []).append(
            f"Zone shapefile lookup failed for '{zone_name}': {type(e).__name__}: {e}"
        )
        return None


def geocode(query: str) -> dict | None:
    try:
        from geopy.geocoders import Nominatim
    except ImportError:
        return None

    geolocator = Nominatim(
        user_agent="IndiaAgriDashboard/3.0 (bharatiya-antariksh-hackathon)",
        timeout=10,
    )

    def _attempt(q, country_codes=None):
        try:
            kwargs = {"exactly_one": True, "addressdetails": True}
            if country_codes:
                kwargs["country_codes"] = country_codes
            loc = geolocator.geocode(q, **kwargs)
            if loc is None:
                return None
            bbox = loc.raw.get("boundingbox", [])
            return {
                "lat": loc.latitude, "lon": loc.longitude,
                "display_name": loc.address,
                "bbox": [float(x) for x in bbox] if len(bbox)==4 else None,
            }
        except Exception:
            return None

    for q, cc in [(query,"in"), (f"{query}, India", None), (query, None)]:
        r = _attempt(q, cc)
        if r:
            return r
    return None


def zoom_from_bbox(bbox):
    if bbox is None: return 11
    span = max(abs(bbox[1]-bbox[0]), abs(bbox[3]-bbox[2]))
    if span > 10: return 5
    if span > 4:  return 7
    if span > 1.5:return 9
    if span > 0.5:return 11
    return 13


def filter_by_boundary(df: pd.DataFrame, boundary: dict | None) -> pd.DataFrame:
    if boundary is None or "_geom" not in boundary:
        return df
    if len(df) == 0:
        return df

    try:
        import geopandas as gpd

        gdf_clusters = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(df["lon"], df["lat"]),
            crs="EPSG:4326",
        )

        geom = boundary["_geom"]
        gdf_boundary = gpd.GeoDataFrame(
            [{"display_name": boundary.get("display_name", "boundary")}],
            geometry=[geom],
            crs="EPSG:4326",
        )

        if gdf_boundary.crs is None:
            gdf_boundary.set_crs("EPSG:4326", inplace=True)
        else:
            gdf_boundary = gdf_boundary.to_crs("EPSG:4326")

        joined = gpd.sjoin(gdf_clusters, gdf_boundary, predicate="intersects", how="inner")

        drop_cols = [c for c in joined.columns if c in
                     ["geometry", "index_right", "display_name_right",
                      "display_name_left"]]
        result = pd.DataFrame(joined.drop(columns=drop_cols, errors="ignore"))
        return result

    except ImportError:
        try:
            bbox = boundary.get("bbox")
            if bbox:
                s, n, w, e = bbox
                mask = (df["lat"]>=s)&(df["lat"]<=n)&(df["lon"]>=w)&(df["lon"]<=e)
                return df[mask]
        except Exception:
            pass
        return df

    except Exception as e:
        errs = st.session_state.get("_filter_errors", [])
        errs.append(f"filter_by_boundary error: {type(e).__name__}: {e}")
        st.session_state._filter_errors = errs
        return df


def filter_by_viewport(df: pd.DataFrame, bounds: dict | None) -> pd.DataFrame:
    if bounds is None:
        return df
    try:
        s = bounds["_southWest"]["lat"]; n = bounds["_northEast"]["lat"]
        w = bounds["_southWest"]["lng"]; e = bounds["_northEast"]["lng"]
        mask = (df["lat"]>=s)&(df["lat"]<=n)&(df["lon"]>=w)&(df["lon"]<=e)
        return df[mask]
    except Exception:
        return df


def boundary_to_folium(boundary: dict | None, label: str = "Boundary Area", show: bool = True) -> folium.FeatureGroup:
    fg = folium.FeatureGroup(name=f"🗺️ {label}", show=show)
    if boundary is None:
        return fg
    try:
        folium.GeoJson(
            boundary["geojson"],
            style_function=lambda _: {
                "fillColor": "#22c55e", "color": "#15803d",
                "weight": 2.5, "fillOpacity": 0.05,
            },
            tooltip=folium.Tooltip(boundary["display_name"][:80], sticky=False),
        ).add_to(fg)
    except Exception:
        pass
    return fg


ZONE_BBOX = {
    "Western Himalayan Zone (J&K / HP / Uttarakhand)":       [30.0,36.0,74.0,80.0],
    "Trans-Gangetic Plain (Punjab / Haryana / W-UP)":         [27.5,32.5,74.0,80.5],
    "Upper Gangetic Plain (Central UP / Uttarakhand Terai)":  [25.5,30.0,77.5,83.5],
    "Middle Gangetic Plain (Bihar / E-UP)":                   [24.0,27.5,82.5,88.5],
    "Lower Gangetic Plain (West Bengal)":                     [21.5,27.0,85.5,89.5],
    "Eastern Plateau & Hills (Jharkhand / Chhattisgarh / Odisha)":[17.5,24.5,80.5,87.5],
    "Central Plateau & Hills (MP / SE Rajasthan)":            [21.0,26.5,74.0,82.5],
    "Western Dry Zone (Rajasthan Desert)":                    [24.0,30.5,69.5,75.5],
    "Gujarat Plains & Hills (Narmada / Mahi Command)":        [20.0,24.7,68.5,74.5],
    "Deccan Plateau (Maharashtra / N-Karnataka / Telangana)": [14.5,21.5,73.5,81.5],
    "Eastern Coastal Plains (Andhra / Odisha Coast)":         [13.5,20.5,79.5,87.0],
    "Western Coastal Plains (Konkan / Malabar)":              [8.5,20.0,73.0,77.5],
    "Southern Plateau & Hills (S-Karnataka / Tamil Nadu Upland)":[9.5,15.5,75.5,80.5],
    "Northeast Hills (Assam Valley / NE States)":             [22.5,28.5,89.5,97.5],
    "Island Territories (A&N / Lakshadweep)":                 [6.5,14.0,92.0,94.0],
}

@st.cache_data(show_spinner=False)
def build_all_india_clusters(season: str) -> pd.DataFrame:
    frames = []
    n_per_zone = 40

    if "Kharif" in season:
        pool  = KHARIF_CROPS + ["Fallow"]
        probs = [0.22,0.22,0.18,0.18,0.20]
    else:
        pool  = RABI_CROPS + KHARIF_CROPS[:1] + ["Fallow"]
        probs = [0.25,0.25,0.20,0.15,0.15]

    for zone_name, cfg in ZONES.items():
        rng  = np.random.default_rng(seed=cfg["seed"] + (0 if "Rabi" in season else 1000))
        bbox = ZONE_BBOX[zone_name]
        s,n,w,e = bbox

        tag  = "".join(wrd[0] for wrd in zone_name.split()[:3]).upper()
        ids  = [f"{tag}-{i:03d}" for i in range(1, n_per_zone+1)]

        lats = rng.uniform(s + 0.05*(n-s), n - 0.05*(n-s), n_per_zone)
        lons = rng.uniform(w + 0.05*(e-w), e - 0.05*(e-w), n_per_zone)

        crops  = rng.choice(pool, size=n_per_zone, p=probs).astype(object)
        stages = rng.choice(STAGE_ORDER, size=n_per_zone, p=[0.12,0.18,0.30,0.22,0.18])
        stages = np.where(crops=="Fallow","Sowing",stages).astype(object)
        status = rng.choice(["Optimal","Moderate","Critical"], size=n_per_zone, p=[0.45,0.30,0.25])
        status = np.where(crops=="Fallow","Optimal",status).astype(object)

        deficit = np.select(
            [status=="Optimal",status=="Moderate",status=="Critical"],
            [rng.integers(0,10,n_per_zone),rng.integers(12,26,n_per_zone),rng.integers(28,48,n_per_zone)],
        )
        deficit = np.where(crops=="Fallow",0,deficit)

        confidence = np.where(
            status=="Optimal",  rng.integers(78,96,n_per_zone),
            np.where(status=="Moderate", rng.integers(62,82,n_per_zone),
                                         rng.integers(55,78,n_per_zone))
        )

        frames.append(pd.DataFrame({
            "cluster_id":           ids,
            "crop":                 crops,
            "growth_stage":         stages,
            "status":               status,
            "deficit_mm":           deficit.astype(int),
            "model_confidence_pct": confidence.astype(int),
            "lat":                  lats,
            "lon":                  lons,
            "zone":                 zone_name,
        }))

    combined = pd.concat(frames, ignore_index=True)

    try:
        import geopandas as gpd
        gdf = gpd.GeoDataFrame(
            combined,
            geometry=gpd.points_from_xy(combined["lon"], combined["lat"]),
            crs="EPSG:4326",
        )
        return gdf
    except ImportError:
        return combined


def stage_progress_bar(stage, width=8):
    idx=STAGE_ORDER.index(stage)
    filled=max(1,min(width,round((idx+1)/len(STAGE_ORDER)*width)))
    return "█"*filled+"░"*(width-filled)


def confidence_html(pct):
    cls="conf-high" if pct>=80 else ("conf-medium" if pct>=60 else "conf-low")
    return f'<span class="{cls}">{pct}%</span>'


@st.cache_data(show_spinner=False)
def build_kc_profile():
    rng=np.random.default_rng(seed=7); days=np.arange(0,121,4)
    kc_th=np.piecewise(days.astype(float),
        [days<=20,(days>20)&(days<=55),(days>55)&(days<=85),days>85],
        [lambda d:0.35+0*d,lambda d:0.35+(d-20)*(1.15-0.35)/35,
         lambda d:1.15+0*d,lambda d:1.15-(d-85)*(1.15-0.45)/35])
    kc_rs=np.clip(kc_th+rng.normal(0,0.045,days.shape)-0.04,0.05,None)
    return pd.DataFrame({"day":days,"Theoretical Kc (FAO-56)":kc_th,"Remote-Sensing Derived Kc":kc_rs})

@st.cache_data(show_spinner=False)
def build_ndvi_ndwi_series(pre=13):
    rng=np.random.default_rng(seed=9); db=np.arange(-90,1,8); n=len(db)
    cd=-10; ds=cd-pre
    bn=0.66+0.05*np.sin(np.linspace(0,1.6,n)); bw=0.36+0.04*np.sin(np.linspace(0,1.3,n))
    fr=np.where(db>=ds,np.clip((db-ds)/max(1,-ds),0,1),0.0)
    return pd.DataFrame({"days_ago":db,"NDVI":bn-0.24*fr+rng.normal(0,0.012,n),
                          "NDWI":bw-0.17*fr+rng.normal(0,0.010,n)}),ds,cd

@st.cache_data(show_spinner=False)
def build_baseline():
    rng=np.random.default_rng(seed=11); x=np.arange(0,121,8)
    bl=0.42+0.10*np.sin(np.linspace(0.2,2.6,len(x)))+rng.normal(0,0.006,len(x))
    cur=bl-0.05+0.07*np.sin(np.linspace(0,1.8,len(x)))+rng.normal(0,0.012,len(x))
    cur=np.where((x>70)&(x<100),cur-0.07,cur)
    return pd.DataFrame({"day":x,"Current Season Root-Zone Moisture":np.clip(cur,0.05,None),
                          "5-Year Historical Baseline Average":np.clip(bl,0.05,None)})

KC_DF=build_kc_profile()
NDVI_DF,DECLINE_START,CRASH_DAY=build_ndvi_ndwi_series()
BASELINE_DF=build_baseline()


def build_advisories(clusters):
    narratives={
        "Critical":"Severe water deficit at the {stage} stage. Shortfall of {deficit} mm. Irrigation action recommended within 48 hours.",
        "Moderate":"Moderate stress at the {stage} stage. Deficit of {deficit} mm. Monitor; schedule irrigation if deficit exceeds 25 mm.",
        "Optimal":"Crop water balance normal at the {stage} stage. Deficit {deficit} mm. No action required.",
    }
    lmap={"Critical":"critical","Moderate":"moderate","Optimal":"optimal"}
    records=[]
    for _,row in clusters.iterrows():
        if row["crop"]=="Fallow": continue
        records.append({"level":lmap[row["status"]],"cluster":row["cluster_id"],
            "crop":row["crop"],"deficit_mm":row["deficit_mm"],
            "confidence":row["model_confidence_pct"],
            "message":narratives[row["status"]].format(
                stage=row["growth_stage"].lower(),deficit=row["deficit_mm"])})
    records.sort(key=lambda r:({"critical":0,"moderate":1,"optimal":2}[r["level"]],-r["deficit_mm"]))
    return records


defaults={
    "map_center":(22.5,78.9),"map_zoom":5,
    "geo_status":"","viewport_bounds":None,"clicked_cluster":None,
    "active_zone":list(ZONES.keys())[8],
    "search_boundary":None,"zone_boundary":None,
}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k]=v

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛰️ Platform Navigation")

    st.markdown("#### 🔍 Search Location")
    with st.form("location_search_form", clear_on_submit=False):
        search_col, button_col = st.columns([4, 1], gap="small")
        with search_col:
            location_query = st.text_input(
                "District / City / State",
                value="Gujarat",
                label_visibility="collapsed",
                help="Searches for an exact administrative boundary polygon.",
            )
        with button_col:
            search_submitted = st.form_submit_button("🔍", use_container_width=True)

    if search_submitted:
        q = location_query.strip()
        st.session_state._boundary_errors = []
        with st.spinner(f"Searching '{q}'…"):
            boundary = find_boundary(q)

        if boundary:
            st.session_state.search_boundary = boundary
            st.session_state.map_center      = boundary["centroid"]
            st.session_state.map_zoom        = zoom_from_bbox(boundary["bbox"])
            # CHANGED: removed source label from status message
            st.session_state.geo_status = f"✅ {boundary['display_name'][:55]}"
        else:
            with st.spinner("Boundary not found — trying to center map via Nominatim…"):
                geo = geocode(q)
            if geo:
                st.session_state.map_center      = (geo["lat"], geo["lon"])
                st.session_state.map_zoom        = zoom_from_bbox(geo.get("bbox"))
                st.session_state.geo_status      = (
                    f"⚠️ Map centered on '{q}' but NO boundary polygon found. "
                    "Cluster filter not active. See debug info below."
                )
                st.session_state.search_boundary = None
            else:
                st.session_state.geo_status      = f"❌ Could not locate '{q}'. Check spelling or try a broader name."
                st.session_state.search_boundary = None

    if st.session_state.geo_status:
        color = "green" if "✅" in st.session_state.geo_status else (
                "orange" if "⚠️" in st.session_state.geo_status else "red")
        st.markdown(
            f'<div class="geo-status" style="color:{color}">{st.session_state.geo_status}</div>',
            unsafe_allow_html=True,
        )

    errors = st.session_state.get("_boundary_errors", [])
    if errors:
        label = "⚠️ Boundary search debug log" if st.session_state.search_boundary else "🔍 Why did it fail?"
        with st.expander(label, expanded=(st.session_state.search_boundary is None)):
            st.caption(f"Steps tried: {len(errors)} attempts logged")
            for e in errors:
                st.caption(f"• {e}")
            if st.session_state.search_boundary is None:
                st.caption(
                    "**Quick fixes:**  \n"
                    "• `pip install osmnx shapely geopandas`  \n"
                    "• Try: `Ahmedabad, Gujarat` instead of just `Ahmedabad`  \n"
                    "• For zones: use **Load Zone Boundary** button below"
                )

    if st.session_state.search_boundary is not None:
        all_cl  = st.session_state.get("_all_clusters", pd.DataFrame())
        n_in    = len(filter_by_boundary(all_cl, st.session_state.search_boundary))
        n_total = len(all_cl)
        pct     = round(100 * n_in / n_total) if n_total > 0 else 0
        if n_in == 0:
            st.warning(
                f"🔷 **Boundary active** · 0/{n_total} clusters inside  \n"
                "No synthetic clusters happen to fall in this exact area — "
                "the boundary itself loaded correctly.",
                icon="🗺️",
            )
        else:
            st.success(
                f"🔷 **Boundary active** · {n_in}/{n_total} clusters ({pct}%) inside",
                icon="🗺️",
            )
        filter_errs = st.session_state.get("_filter_errors", [])
        if filter_errs:
            with st.expander("⚠️ Filter warnings", expanded=False):
                for fe in filter_errs:
                    st.caption(fe)
                st.caption("Install geopandas for accurate spatial filtering: pip install geopandas")
        if st.button("✖ Clear boundary filter"):
            st.session_state.search_boundary  = None
            st.session_state.geo_status       = ""
            st.session_state._filter_errors   = []
            st.rerun()

    st.markdown("---")

    st.markdown("#### 🗺️ Agro-Climatic Zone")
    selected_zone = st.selectbox(
        "Active Analysis Zone",
        list(ZONES.keys()),
        index=list(ZONES.keys()).index(st.session_state.active_zone),
    )
    if selected_zone != st.session_state.active_zone:
        st.session_state.active_zone    = selected_zone
        st.session_state.zone_boundary  = None

    if st.button("🗺️ Load Zone Boundary"):
        zb = get_zone_boundary(selected_zone)
        st.session_state.zone_boundary = zb
        if zb:
            st.session_state.search_boundary = zb
            st.session_state.map_center      = zb["centroid"]
            st.session_state.map_zoom        = zoom_from_bbox(zb["bbox"])
            st.session_state.geo_status = f"✅ Zone boundary: {zb['display_name'][:45]}"
        else:
            st.warning(
                f"Zone boundary not found for **{selected_zone}**. Either the zone "
                "shapefile isn't loaded, or its regioncode isn't mapped/confirmed yet."
            )

    zb_current = st.session_state.get("zone_boundary")
    if zb_current and zb_current.get("attributes"):
        attrs = zb_current["attributes"]
        area_km2 = zb_current.get("area_km2")
        area_str = f"{area_km2:,.0f} km²" if area_km2 is not None else "—"
        with st.expander("📊 Zone Agro-Climatology Profile", expanded=False):
            st.dataframe(pd.DataFrame({
                "Attribute": ["Zone Code","Zone Name","State(s)","Soil","Major Crops",
                              "Avg. Annual Rainfall","Avg. Jan Temp","Avg. Jul Temp",
                              "Area","Remarks"],
                "Value": [attrs["regioncode"], attrs["regionname"], attrs["state"],
                          attrs["soil"], attrs["majorcrops"], attrs["avgann_rf"],
                          attrs["avgtmp_jan"], attrs["avgtmp_jul"], area_str,
                          attrs["remarks"]],
            }), hide_index=True, use_container_width=True)

    # REMOVED: "Inspect zone shapefile data" expander (Image 2)

    local_errs = st.session_state.get("_local_dataset_errors", [])
    if local_errs:
        with st.expander("⚠️ Local shapefile status", expanded=False):
            for e in local_errs:
                st.caption(f"• {e}")

    st.markdown("---")
    season = st.selectbox("🗓️ Season", ["Rabi (Winter)","Kharif (Monsoon)"], index=0)

    st.markdown("---")
    st.caption(f"🛰️ Sentinel-1 (C-band) + NISAR (L-band)  \n⏱️ 8-Day Composite  \n📅 {dt.date.today():%d %b %Y}")


# ─────────────────────────────────────────────────────────────────────────────
# BUILD DATA
# ─────────────────────────────────────────────────────────────────────────────
ALL_CLUSTERS = build_all_india_clusters(season)
st.session_state["_all_clusters"] = ALL_CLUSTERS

ACTIVE_CLUSTERS = ALL_CLUSTERS.copy()
active_boundary = st.session_state.search_boundary or st.session_state.zone_boundary

BOUNDARY_CLUSTERS = filter_by_boundary(ACTIVE_CLUSTERS, active_boundary)
ADVISORIES        = build_advisories(BOUNDARY_CLUSTERS)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="app-header-eyebrow">ISRO-ALIGNED SATELLITE INTELLIGENCE · ALL-INDIA DEPLOYMENT · 600 FIELD CLUSTERS</div>',
            unsafe_allow_html=True)
st.title("🛰️ India Crop Intelligence & Irrigation Planning Platform")
st.markdown("""
    <div class="meta-banner"><span>
    <span class="dot-live">●</span> Pipeline: ACTIVE &nbsp;|&nbsp;
    Sensors: Sentinel-1 (C-band SAR) + NISAR (L-band SAR) &nbsp;|&nbsp;
    Cadence: 8-Day Best-Pixel Composite &nbsp;|&nbsp;
    Boundaries: Local Shapefile Lookup + Live OSM/Nominatim Fallback
    </span></div>""", unsafe_allow_html=True)

boundary_label = active_boundary["display_name"][:60] if active_boundary else "All India (no boundary filter)"
st.caption(f"Active boundary: **{boundary_label}** · Zone: **{selected_zone}** · Season: **{season}**")
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────
VIEWPORT_CLUSTERS = filter_by_viewport(BOUNDARY_CLUSTERS, st.session_state.viewport_bounds)

n_total       = len(VIEWPORT_CLUSTERS)
n_critical    = int((VIEWPORT_CLUSTERS["status"]=="Critical").sum())
n_boundary    = len(BOUNDARY_CLUSTERS)
dominant_crop = (
    VIEWPORT_CLUSTERS[VIEWPORT_CLUSTERS["crop"]!="Fallow"]["crop"].value_counts().idxmax()
    if len(VIEWPORT_CLUSTERS[VIEWPORT_CLUSTERS["crop"]!="Fallow"])>0 else "—"
)

c1,c2,c3,c4,c5 = st.columns(5)
with c1: st.metric("🌍 Boundary Clusters", f"{n_boundary}")
with c2: st.metric("🔭 Visible (Viewport)", f"{n_total}")
with c3: st.metric("🚨 Critical Alerts", f"{n_critical}", delta="+3 vs last cycle", delta_color="inverse")
with c4: st.metric("🌱 Dominant Crop", dominant_crop)
with c5:
    area_km2 = active_boundary.get("area_km2") if active_boundary else None
    st.metric("📏 Selected Area", f"{area_km2:,.0f} km²" if area_km2 is not None else "—")

# REMOVED: the long caption below the KPI cards (Image 3)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_map, tab_trends, tab_advisory = st.tabs([
    "🛰️ Geospatial Control Center",
    "📈 Multi-Temporal Analytics",
    "📋 Irrigation Advisory",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    ctrl_col, map_col = st.columns([1,2], gap="large")

    with ctrl_col:
        st.markdown('<div class="control-card">', unsafe_allow_html=True)
        st.subheader("🌾 Monitoring Controls")

        layer_view = st.radio("🗺️ Layer",
            ["🌱 Crop Classification","💧 Moisture Stress"], index=0)
        crop_class = st.selectbox("🌱 Crop Filter", CROP_FILTER_OPTIONS, index=0)

        st.markdown("&nbsp;", unsafe_allow_html=True)
        if layer_view.startswith("🌱"):
            st.markdown("**Crop Legend**")
            legend_html="".join(f'<span class="legend-chip"><span class="legend-dot" style="background:{c}"></span>{k}</span>'
                                 for k,c in CROP_COLORS.items())
        else:
            st.markdown("**Stress Legend**")
            legend_html="".join(f'<span class="legend-chip"><span class="legend-dot" style="background:{c}"></span>{s}</span>'
                                 for s,c in STATUS_COLORS.items())
        st.markdown(f"<div>{legend_html}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        view_clusters = BOUNDARY_CLUSTERS.copy()
        if crop_class!="All Crops":
            view_clusters = view_clusters[view_clusters["crop"]==crop_class]
        st.markdown("&nbsp;")

        if len(BOUNDARY_CLUSTERS) == 0 and active_boundary is not None:
            st.info(
                f"📍 No field clusters fall inside **{active_boundary['display_name'][:40]}**. "
                "The boundary and filter are working correctly — this selection "
                "just doesn't contain any of the synthetic monitoring clusters.",
                icon="ℹ️",
            )
        else:
            st.caption(f"Showing **{len(view_clusters)}** clusters · {crop_class} · {season}")

        if st.session_state.clicked_cluster is not None:
            cid = st.session_state.clicked_cluster
            row = ACTIVE_CLUSTERS[ACTIVE_CLUSTERS["cluster_id"]==cid]
            if len(row)>0:
                r=row.iloc[0]
                st.markdown(f"""
                <div class="detail-card">
                    <div class="detail-card-title">📌 {r['cluster_id']} — {r['crop']}</div>
                    <b>Zone:</b> {r['zone'][:40]}<br>
                    <b>Stage:</b> {r['growth_stage']} [{stage_progress_bar(r['growth_stage'])}]<br>
                    <b>Status:</b> {r['status']}<br>
                    <b>Deficit:</b> {r['deficit_mm']} mm<br>
                    <b>Model Confidence:</b> {confidence_html(int(r['model_confidence_pct']))}<br>
                </div>""", unsafe_allow_html=True)
            if st.button("✖ Clear"):
                st.session_state.clicked_cluster=None; st.rerun()

    with map_col:
        map_lat,map_lon = st.session_state.map_center
        map_zoom        = st.session_state.map_zoom

        m = folium.Map(
            location=[map_lat, map_lon], zoom_start=map_zoom,
            tiles=None,
            control_scale=True,
        )
        folium.TileLayer(
            tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
            attr='Map data: &copy; OpenStreetMap contributors, SRTM | Map style: &copy; OpenTopoMap (CC-BY-SA)',
            name="Topographic",
            control=False,
        ).add_to(m)

        if active_boundary:
            label = "Search Boundary" if st.session_state.search_boundary else "Zone Boundary"
            boundary_to_folium(active_boundary, label, show=False).add_to(m)

        # CHANGED: show=True so clusters are visible by default
        cluster_fg = folium.FeatureGroup(name="🌾 Field Clusters", show=True)
        crop_mode  = layer_view.startswith("🌱")
        half_box   = 0.018

        for _,row in view_clusters.iterrows():
            color = CROP_COLORS.get(row["crop"],"#9E9E9E") if crop_mode else STATUS_COLORS[row["status"]]
            tip   = (f"<b>{row['cluster_id']}</b> · {row['crop']} · {row['zone'][:30]}<br>"
                     f"Stage:{row['growth_stage']} | Status:{row['status']} | Deficit:{row['deficit_mm']}mm<br>"
                     f"Conf:{row['model_confidence_pct']}%")
            folium.Rectangle(
                bounds=[[row["lat"]-half_box,row["lon"]-half_box],[row["lat"]+half_box,row["lon"]+half_box]],
                color=color, weight=1.5, fill=True, fill_color=color, fill_opacity=0.30,
                tooltip=tip,
            ).add_to(cluster_fg)
            folium.CircleMarker(
                location=[row["lat"],row["lon"]], radius=4,
                color=color, fill=True, fill_color=color, fill_opacity=0.9,
                tooltip=tip,
            ).add_to(cluster_fg)

        cluster_fg.add_to(m)

        if active_boundary:
            folium.Marker(
                location=list(active_boundary["centroid"]),
                popup=f"Search Center: {active_boundary['display_name'][:60]}",
                tooltip="📍 Search Center",
                icon=folium.Icon(color="red"),
            ).add_to(m)

        folium.LayerControl(collapsed=True).add_to(m)

        map_output = st_folium(m, use_container_width=True, height=560,
                               returned_objects=["bounds","last_object_clicked"])

        if map_output and map_output.get("bounds"):
            st.session_state.viewport_bounds = map_output["bounds"]

        clicked = map_output.get("last_object_clicked") if map_output else None
        if clicked and isinstance(clicked,dict):
            clat=clicked.get("lat"); clon=clicked.get("lng")
            if clat is not None:
                dists=np.hypot(ACTIVE_CLUSTERS["lat"].values-clat, ACTIVE_CLUSTERS["lon"].values-clon)
                nid=ACTIVE_CLUSTERS.iloc[int(np.argmin(dists))]["cluster_id"]
                if st.session_state.clicked_cluster!=nid:
                    st.session_state.clicked_cluster=nid; st.rerun()

        st.caption(
            "💡 **Boundary filter**: use sidebar search to filter clusters to any state/district/village.  \n"
            "💡 **Basemap**: the topographic layer renders rivers, canals, and terrain contours natively.  \n"
            "💡 **Layers**: use the layers icon (top-right of the map) to toggle Field Clusters or Boundary outline."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_trends:
    st.subheader("📈 Multi-Temporal Satellite Analytics")
    st.caption(f"8-day composite indices and baseline comparison · Zone: **{selected_zone}** · {season}")

    c1,c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("##### 🛰️ NDVI / NDWI — Last 90 Days")
        fig=go.Figure()
        fig.add_vrect(x0=DECLINE_START,x1=CRASH_DAY,fillcolor=COLOR_SECONDARY,opacity=0.12,line_width=0,
                      annotation_text="Pre-Visual Window",annotation_position="top left",
                      annotation=dict(font_size=10,font_color=COLOR_MUTED))
        fig.add_vline(x=CRASH_DAY,line_width=1.5,line_dash="dot",line_color=COLOR_CRITICAL,
                      annotation_text="Canopy Crash",annotation_position="top right",
                      annotation=dict(font_size=10,font_color=COLOR_CRITICAL))
        for col,color,name in [("NDVI",COLOR_PRIMARY,"NDVI"),("NDWI","#1565C0","NDWI")]:
            fig.add_trace(go.Scatter(x=NDVI_DF["days_ago"],y=NDVI_DF[col],
                mode="lines+markers",name=name,line=dict(color=color,width=3),marker=dict(size=5)))
        fig.update_layout(height=420,margin=dict(l=10,r=10,t=40,b=10),
            plot_bgcolor=COLOR_SURFACE,paper_bgcolor=COLOR_SURFACE,
            xaxis_title="Days Relative to Today",yaxis_title="Index",
            legend=dict(orientation="h",yanchor="bottom",y=1.02),font=dict(color=COLOR_TEXT),
            xaxis=dict(gridcolor=COLOR_BORDER),yaxis=dict(gridcolor=COLOR_BORDER))
        st.plotly_chart(fig,use_container_width=True)

    with c2:
        st.markdown("##### 📊 Root-Zone Moisture vs 5-Year Baseline")
        fig2=go.Figure()
        fig2.add_trace(go.Scatter(x=BASELINE_DF["day"],y=BASELINE_DF["5-Year Historical Baseline Average"],
            mode="lines",name="5-Year Baseline",line=dict(color=COLOR_MUTED,width=2,dash="dash")))
        fig2.add_trace(go.Scatter(x=BASELINE_DF["day"],y=BASELINE_DF["Current Season Root-Zone Moisture"],
            mode="lines+markers",name="Current Season",line=dict(color=COLOR_PRIMARY,width=3),marker=dict(size=5)))
        fig2.update_layout(height=420,margin=dict(l=10,r=10,t=40,b=10),
            plot_bgcolor=COLOR_SURFACE,paper_bgcolor=COLOR_SURFACE,
            xaxis_title="Days Since Sowing",yaxis_title="Root-Zone Moisture Index",
            legend=dict(orientation="h",yanchor="bottom",y=1.02),font=dict(color=COLOR_TEXT),
            xaxis=dict(gridcolor=COLOR_BORDER),yaxis=dict(gridcolor=COLOR_BORDER))
        st.plotly_chart(fig2,use_container_width=True)

    st.markdown("##### 📐 Crop Coefficient (Kc) — Theoretical vs Remote-Sensing")
    fig3=go.Figure()
    for col,color,dash,name in [
        ("Theoretical Kc (FAO-56)",COLOR_MUTED,"dash","Theoretical Kc (FAO-56)"),
        ("Remote-Sensing Derived Kc",COLOR_PRIMARY,"solid","Remote-Sensing Derived Kc"),
    ]:
        fig3.add_trace(go.Scatter(x=KC_DF["day"],y=KC_DF[col],mode="lines",name=name,
            line=dict(color=color,width=2.5,dash=dash)))
    fig3.update_layout(height=300,margin=dict(l=10,r=10,t=30,b=10),
        plot_bgcolor=COLOR_SURFACE,paper_bgcolor=COLOR_SURFACE,
        xaxis_title="Days Since Sowing",yaxis_title="Kc",
        legend=dict(orientation="h",yanchor="bottom",y=1.02),font=dict(color=COLOR_TEXT),
        xaxis=dict(gridcolor=COLOR_BORDER),yaxis=dict(gridcolor=COLOR_BORDER))
    st.plotly_chart(fig3,use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — IRRIGATION ADVISORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_advisory:
    st.subheader("📋 Field Irrigation Advisory")
    st.caption(
        f"Advisories for **{boundary_label}** · Zone: **{selected_zone}** · {season}  \n"
        "Scoped to the currently selected state/district/zone/search result."
    )

    region_ids   = set(BOUNDARY_CLUSTERS["cluster_id"])
    region_advs  = [a for a in ADVISORIES if a["cluster"] in region_ids]
    crit_adv     = [a for a in region_advs if a["level"]=="critical"]
    mod_adv      = [a for a in region_advs if a["level"]=="moderate"]
    opt_adv      = [a for a in region_advs if a["level"]=="optimal"]

    s1, s2, s3 = st.columns(3)
    with s1: st.metric("🔴 Critical", len(crit_adv))
    with s2: st.metric("🟠 Moderate", len(mod_adv))
    with s3: st.metric("🟢 Optimal", len(opt_adv))
    st.markdown("---")

    if not region_advs:
        st.info(
            f"No irrigation advisories for **{boundary_label}** — either no field "
            "clusters fall inside this selection, or none have active crops this season.",
            icon="ℹ️",
        )
    else:
        col_crit, col_mod, col_opt = st.columns(3, gap="medium")

        with col_crit:
            st.markdown("##### 🔴 Critical")
            if crit_adv:
                for a in crit_adv:
                    st.error(
                        f"**{a['cluster']} · {a['crop']}** (conf {a['confidence']}%)  \n{a['message']}",
                        icon="🚨",
                    )
            else:
                st.caption("No critical advisories.")

        with col_mod:
            st.markdown("##### 🟠 Moderate")
            if mod_adv:
                for a in mod_adv:
                    st.warning(
                        f"**{a['cluster']} · {a['crop']}** (conf {a['confidence']}%)  \n{a['message']}",
                        icon="⚠️",
                    )
            else:
                st.caption("No moderate advisories.")

        with col_opt:
            st.markdown("##### 🟢 Optimal")
            if opt_adv:
                for a in opt_adv:
                    st.success(
                        f"**{a['cluster']} · {a['crop']}** (conf {a['confidence']}%)  \n{a['message']}",
                        icon="✅",
                    )
            else:
                st.caption("No optimal-status fields.")

    st.markdown("---")
    st.caption(f"Snapshot: {dt.datetime.now():%d %b %Y, %H:%M} IST · "
               f"Boundary: {boundary_label} · Next revisit: 8 days · Model: AquaSense v1")