"""
Gujarat State Crop Intelligence & Irrigation Planning Platform
================================================================
An enterprise-grade Streamlit decision-support dashboard for AI-driven
crop classification, phenology-aware moisture stress monitoring, and
two-way canal proximity / infrastructure policy advisory across five
agricultural command zones in Gujarat (including the original Sanchore
tail-end pilot, now folded in as one zone among several).

Run with:
    streamlit run app.py

No external files or network calls are required - every region's field
clusters, canal centerlines, and time-series baselines are generated
in-process with numpy/pandas, so the app runs standalone, instantly,
every time.
"""

from __future__ import annotations

import datetime as dt
import io

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

# ----------------------------------------------------------------------------
# Page configuration  (must be the first Streamlit call)
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Gujarat State Crop Intelligence & Irrigation Planning Platform",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Design tokens - Earth-Sciences palette  [RETAINED EXACTLY, UNCHANGED]
# ----------------------------------------------------------------------------
COLOR_PRIMARY = "#2E7D32"      # Forest Green   - healthy vegetation / primary accent
COLOR_SECONDARY = "#EF6C00"    # Warning Orange - moderate stress / secondary highlight
COLOR_CRITICAL = "#C62828"     # Alert Red      - critical stress
COLOR_BG = "#F8F9FA"           # Ultra-clean light slate gray - page background
COLOR_SURFACE = "#FFFFFF"      # Card surface
COLOR_TEXT = "#1B2127"         # Primary text
COLOR_MUTED = "#5F6B72"        # Secondary text
COLOR_BORDER = "#E3E7EA"       # Hairline borders

STATUS_COLORS = {
    "Optimal": COLOR_PRIMARY,
    "Moderate": COLOR_SECONDARY,
    "Critical": COLOR_CRITICAL,
}

# Crop-identity palette  [RETAINED INTACT - Cotton added alongside, nothing
# existing changed, to support the expanded Gujarat crop matrix]
CROP_COLORS = {
    "Wheat": "#1565C0",     # Blue
    "Cumin": "#6A1B9A",     # Purple
    "Mustard": "#F9A825",   # Gold
    "Castor": "#EF6C00",    # Orange (secondary token reused for identity)
    "Cotton": "#00897B",    # Teal (new - Gujarat's flagship cash crop)
    "Fallow": "#9E9E9E",    # Gray
}

STAGE_ORDER = ["Sowing", "Early Vegetative", "Vegetative", "Flowering", "Maturity"]
CROPS_MAIN = ["Cumin", "Mustard", "Castor", "Cotton", "Wheat"]
CROP_FILTER_OPTIONS = ["All Crops"] + CROPS_MAIN + ["Fallow"]

# ----------------------------------------------------------------------------
# Global style injection  [RETAINED - metric cards, tabs, paddings unchanged;
# new rules only ADD, none of the original selectors were removed/altered]
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {COLOR_BG};
        }}

        /* Tighten the default block padding for a denser, dashboard feel */
        .block-container {{
            padding-top: 1.6rem;
            padding-bottom: 2.5rem;
            max-width: 1400px;
        }}

        h1, h2, h3 {{
            color: {COLOR_TEXT};
            font-family: "Source Sans Pro", "Segoe UI", sans-serif;
        }}

        p, span, label, div {{
            color: {COLOR_TEXT};
        }}

        /* Metric cards */
        div[data-testid="stMetric"] {{
            background-color: {COLOR_SURFACE};
            border: 1px solid {COLOR_BORDER};
            border-radius: 10px;
            padding: 1.1rem 1.3rem 0.9rem 1.3rem;
            box-shadow: 0 1px 3px rgba(16, 24, 32, 0.04);
        }}

        div[data-testid="stMetricLabel"] {{
            color: {COLOR_MUTED};
            font-weight: 600;
            letter-spacing: 0.01em;
        }}

        div[data-testid="stMetricValue"] {{
            color: {COLOR_TEXT};
        }}

        /* Tabs */
        button[data-baseweb="tab"] {{
            font-size: 1.0rem;
            font-weight: 600;
            color: {COLOR_MUTED};
        }}

        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {COLOR_PRIMARY};
            border-bottom-color: {COLOR_PRIMARY} !important;
        }}

        div[data-baseweb="tab-highlight"] {{
            background-color: {COLOR_PRIMARY} !important;
        }}

        /* Sidebar-style control card inside Tab 1 */
        .control-card {{
            background-color: {COLOR_SURFACE};
            border: 1px solid {COLOR_BORDER};
            border-radius: 12px;
            padding: 1.4rem 1.4rem 0.6rem 1.4rem;
        }}

        .legend-chip {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.85rem;
            color: {COLOR_MUTED};
            margin-right: 1.1rem;
        }}

        .legend-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }}

        .app-header-eyebrow {{
            color: {COLOR_PRIMARY};
            font-weight: 700;
            font-size: 0.85rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: -0.4rem;
        }}

        /* Buttons */
        .stButton button {{
            background-color: {COLOR_PRIMARY};
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
        }}
        .stButton button:hover {{
            background-color: #245F27;
            color: white;
        }}

        /* Proximity / gate-valve administrative callout card (Case A) */
        .gate-callout {{
            background-color: #FDECEA;
            border: 1px solid {COLOR_CRITICAL};
            border-left: 5px solid {COLOR_CRITICAL};
            border-radius: 8px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.8rem;
        }}

        .gate-callout-title {{
            color: {COLOR_CRITICAL};
            font-weight: 700;
            font-size: 0.95rem;
            margin-bottom: 0.25rem;
        }}

        /* Infrastructure-gap policy advisory card (Case B) */
        .policy-callout {{
            background-color: #FFF4E5;
            border: 1px solid {COLOR_SECONDARY};
            border-left: 5px solid {COLOR_SECONDARY};
            border-radius: 8px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.8rem;
        }}

        .policy-callout-title {{
            color: {COLOR_SECONDARY};
            font-weight: 700;
            font-size: 0.95rem;
            margin-bottom: 0.25rem;
        }}

        /* Resolved-by-simulation success card */
        .resolved-callout {{
            background-color: #EAF5EB;
            border: 1px solid {COLOR_PRIMARY};
            border-left: 5px solid {COLOR_PRIMARY};
            border-radius: 8px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.8rem;
        }}

        .resolved-callout-title {{
            color: {COLOR_PRIMARY};
            font-weight: 700;
            font-size: 0.95rem;
            margin-bottom: 0.25rem;
        }}

        /* Fused multi-sensor meta-data banner */
        .meta-banner {{
            background-color: #0F2027;
            background-image: linear-gradient(90deg, #14302B 0%, #173A2E 100%);
            border: 1px solid #1F4D3C;
            border-radius: 10px;
            padding: 0.75rem 1.2rem;
            margin-bottom: 0.4rem;
        }}

        .meta-banner span {{
            color: #D7F3E3;
            font-size: 0.82rem;
            font-family: "Source Code Pro", "Consolas", monospace;
            letter-spacing: 0.01em;
        }}

        .meta-banner .dot-live {{
            color: #66FF99;
            font-weight: 700;
        }}

        hr {{
            border-color: {COLOR_BORDER};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Region registry - five state-wide command zones, each with its own
# center coordinates and canal centerline geometry. [STANDALONE / SYNTHETIC]
# ----------------------------------------------------------------------------
REGIONS: dict[str, dict] = {
    "Sanchore Tail-End Reach (Narmada Main Canal Extension)": {
        "center": (24.75, 71.60),
        "canal": [(24.625, 71.470), (24.690, 71.530), (24.745, 71.590), (24.800, 71.650), (24.860, 71.715)],
        "seed": 42,
    },
    "Sanand Crop Cluster (Ahmedabad District)": {
        "center": (22.99, 72.38),
        "canal": [(22.90, 72.28), (22.94, 72.32), (22.99, 72.38), (23.04, 72.43), (23.09, 72.48)],
        "seed": 101,
    },
    "Mahi Right Bank Command Area (Anand / Vadodara)": {
        "center": (22.56, 73.00),
        "canal": [(22.47, 72.90), (22.51, 72.95), (22.56, 73.00), (22.61, 73.05), (22.66, 73.11)],
        "seed": 202,
    },
    "Saurashtra Branch Grid (Surendranagar Zone)": {
        "center": (22.73, 71.64),
        "canal": [(22.63, 71.54), (22.68, 71.59), (22.73, 71.64), (22.78, 71.69), (22.84, 71.75)],
        "seed": 303,
    },
    "Mehsana Intensive Tube-Well & Canal Belt": {
        "center": (23.59, 72.39),
        "canal": [(23.49, 72.29), (23.54, 72.34), (23.59, 72.39), (23.64, 72.44), (23.70, 72.50)],
        "seed": 404,
    },
}
REGION_NAMES = list(REGIONS.keys())


def distance_to_canal_m(lat: float, lon: float, polyline: list[tuple[float, float]]) -> float:
    """Minimum distance in meters from a point to a polyline, treating each
    segment as locally planar (a safe approximation at this geographic
    scale of a few tens of kilometers)."""
    min_dist = float("inf")
    for (lat1, lon1), (lat2, lon2) in zip(polyline[:-1], polyline[1:]):
        lat0 = (lat1 + lat2) / 2.0
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * np.cos(np.radians(lat0))

        x1, y1 = lon1 * m_per_deg_lon, lat1 * m_per_deg_lat
        x2, y2 = lon2 * m_per_deg_lon, lat2 * m_per_deg_lat
        xp, yp = lon * m_per_deg_lon, lat * m_per_deg_lat

        dx, dy = x2 - x1, y2 - y1
        seg_len_sq = dx * dx + dy * dy
        t = 0.0 if seg_len_sq == 0 else max(0.0, min(1.0, ((xp - x1) * dx + (yp - y1) * dy) / seg_len_sq))
        proj_x, proj_y = x1 + t * dx, y1 + t * dy
        dist = float(np.hypot(xp - proj_x, yp - proj_y))
        min_dist = min(min_dist, dist)
    return min_dist


def stage_progress_bar(stage: str, width: int = 8) -> str:
    """Render a simple text progress bar for a phenological growth stage,
    e.g. 'Flowering [██████░░]'."""
    idx = STAGE_ORDER.index(stage)
    filled = max(1, min(width, round((idx + 1) / len(STAGE_ORDER) * width)))
    return "█" * filled + "░" * (width - filled)


@st.cache_data(show_spinner=False)
def build_region_clusters(region_name: str, n: int = 24) -> pd.DataFrame:
    """Generate the full synthetic field-cluster table for a single region:
    identity, crop, phenology, moisture status, location, and derived canal
    proximity. Each region uses its own deterministic seed so results are
    stable across reruns but visibly distinct between regions.

    Four clusters are deliberately pinned to specific canal distances (two
    under 1000 m, two over 1000 m) and forced to Critical status, so that
    BOTH branches of the Tab 3 two-way proximity trigger are always
    demonstrable for every region - without this, a region could randomly
    end up with zero examples of one branch and silently never show it.
    """
    cfg = REGIONS[region_name]
    rng = np.random.default_rng(seed=cfg["seed"])
    canal = cfg["canal"]
    center_lat, center_lon = cfg["center"]

    region_tag = "".join(w[0] for w in region_name.split()[:3]).upper()
    cluster_ids = [f"{region_tag}-{i:03d}" for i in range(1, n + 1)]

    crops = rng.choice(CROPS_MAIN, size=n).astype(object)
    fallow_idx = rng.choice(n, size=3, replace=False)
    crops[fallow_idx] = "Fallow"

    stages = rng.choice(STAGE_ORDER, size=n, p=[0.12, 0.18, 0.30, 0.22, 0.18])
    stages = np.where(crops == "Fallow", "Sowing", stages).astype(object)

    status = rng.choice(["Optimal", "Moderate", "Critical"], size=n, p=[0.45, 0.30, 0.25])
    status = np.where(crops == "Fallow", "Optimal", status).astype(object)

    deficit = np.select(
        [status == "Optimal", status == "Moderate", status == "Critical"],
        [rng.integers(0, 10, size=n), rng.integers(12, 26, size=n), rng.integers(28, 48, size=n)],
    )
    deficit = np.where(crops == "Fallow", 0, deficit)

    lat = center_lat + rng.normal(0, 0.09, size=n)
    lon = center_lon + rng.normal(0, 0.12, size=n)

    # --- Guaranteed two-way trigger coverage -------------------------------
    seg1, seg2 = canal[1], canal[2]

    def offset_point(t: float, offset_m: float) -> tuple[float, float]:
        d_lat, d_lon = seg2[0] - seg1[0], seg2[1] - seg1[1]
        base_lat, base_lon = seg1[0] + t * d_lat, seg1[1] + t * d_lon
        norm = float(np.hypot(d_lat, d_lon))
        perp_lat, perp_lon = -d_lon / norm, d_lat / norm
        off_deg = offset_m / 111320.0
        return base_lat + perp_lat * off_deg, base_lon + perp_lon * off_deg

    # (position along canal segment, target offset in meters)
    pin_plan = [
        (0.05, 250),   # Case A - near canal
        (0.20, 650),   # Case A - near canal
        (0.35, 1800),  # Case B - stranded / infrastructure gap
        (0.50, 3600),  # Case B - stranded / infrastructure gap
    ]
    for k, (t, off_m) in enumerate(pin_plan):
        plat, plon = offset_point(t, off_m)
        lat[k] = plat
        lon[k] = plon
        status[k] = "Critical"
        if crops[k] == "Fallow":
            crops[k] = "Wheat"
        deficit[k] = int(rng.integers(30, 46))
        stages[k] = rng.choice(["Vegetative", "Flowering"])

    df = pd.DataFrame(
        {
            "cluster_id": cluster_ids,
            "crop": crops,
            "growth_stage": stages,
            "status": status,
            "deficit_mm": deficit.astype(int),
            "lat": lat,
            "lon": lon,
        }
    )
    df["distance_to_canal_m"] = df.apply(
        lambda r: round(distance_to_canal_m(r["lat"], r["lon"], canal)), axis=1
    )
    df["region"] = region_name
    return df


def apply_canal_expansion_simulation(
    clusters_df: pd.DataFrame, threshold_m: int = 1000, simulated_m: int = 300
) -> pd.DataFrame:
    """Policy sandbox transform: for fields currently stranded beyond
    `threshold_m`, simulate a new distributary extension reaching them by
    setting their distance down to `simulated_m`. Returns a copy; the
    underlying cached cluster data is never mutated."""
    sim = clusters_df.copy()
    stranded_mask = sim["distance_to_canal_m"] > threshold_m
    sim.loc[stranded_mask, "distance_to_canal_m"] = simulated_m
    sim["simulated_expansion"] = stranded_mask
    return sim


@st.cache_data(show_spinner=False)
def build_kc_profile() -> pd.DataFrame:
    """Synthesize a theoretical crop-coefficient (Kc) curve against a
    simulated remote-sensing-derived vegetation profile over one season."""
    rng_local = np.random.default_rng(seed=7)
    days = np.arange(0, 121, 4)

    kc_theoretical = np.piecewise(
        days.astype(float),
        [days <= 20, (days > 20) & (days <= 55), (days > 55) & (days <= 85), days > 85],
        [
            lambda d: 0.35 + 0.0 * d,
            lambda d: 0.35 + (d - 20) * (1.15 - 0.35) / 35,
            lambda d: 1.15 + 0.0 * d,
            lambda d: 1.15 - (d - 85) * (1.15 - 0.45) / 35,
        ],
    )

    noise = rng_local.normal(0, 0.045, size=days.shape)
    kc_remote_sensing = np.clip(kc_theoretical + noise - 0.04, 0.05, None)

    return pd.DataFrame(
        {
            "day": days,
            "Theoretical Kc (FAO-56)": kc_theoretical,
            "Remote-Sensing Derived Kc": kc_remote_sensing,
        }
    )


KC_DF = build_kc_profile()


@st.cache_data(show_spinner=False)
def build_ndvi_ndwi_series(pre_visual_window_days: int = 13) -> tuple[pd.DataFrame, int, int]:
    """Synthesize an 8-day composite NDVI/NDWI time series over the last
    90 days, with a designed moisture-stress crash near the present day
    and a pre-visual detection window leading up to it."""
    rng_local = np.random.default_rng(seed=9)
    days_back = np.arange(-90, 1, 8)
    n = len(days_back)

    crash_day = -10
    decline_start = crash_day - pre_visual_window_days

    base_ndvi = 0.66 + 0.05 * np.sin(np.linspace(0, 1.6, n))
    base_ndwi = 0.36 + 0.04 * np.sin(np.linspace(0, 1.3, n))

    decline_fraction = np.where(
        days_back >= decline_start,
        np.clip((days_back - decline_start) / max(1, (0 - decline_start)), 0, 1),
        0.0,
    )

    ndvi = base_ndvi - 0.24 * decline_fraction + rng_local.normal(0, 0.012, n)
    ndwi = base_ndwi - 0.17 * decline_fraction + rng_local.normal(0, 0.010, n)

    df = pd.DataFrame({"days_ago": days_back, "NDVI": ndvi, "NDWI": ndwi})
    return df, decline_start, crash_day


NDVI_NDWI_DF, DECLINE_START_DAY, CRASH_DAY = build_ndvi_ndwi_series()


@st.cache_data(show_spinner=False)
def build_baseline_comparison() -> pd.DataFrame:
    """Synthesize current-season root-zone moisture vs. a 5-year regional
    historical baseline average across the vegetative growth pipeline."""
    rng_local = np.random.default_rng(seed=11)
    stage_x = np.arange(0, 121, 8)

    baseline = 0.42 + 0.10 * np.sin(np.linspace(0.2, 2.6, len(stage_x))) + rng_local.normal(0, 0.006, len(stage_x))
    current = baseline - 0.05 + 0.07 * np.sin(np.linspace(0, 1.8, len(stage_x))) + rng_local.normal(0, 0.012, len(stage_x))
    dip_zone = (stage_x > 70) & (stage_x < 100)
    current = np.where(dip_zone, current - 0.07, current)

    return pd.DataFrame(
        {
            "day": stage_x,
            "Current Season Root-Zone Moisture": np.clip(current, 0.05, None),
            "5-Year Historical Baseline Average": np.clip(baseline, 0.05, None),
        }
    )


BASELINE_DF = build_baseline_comparison()


def build_advisories(clusters: pd.DataFrame) -> list[dict]:
    """Derive prescriptive irrigation advisories directly from the field
    cluster table, so cluster id / crop / deficit values shown in the
    advisory feed always agree with what is plotted on the map."""
    narrative_by_status = {
        "Critical": (
            "Severe water deficit detected at the {stage} stage. Estimated shortfall of "
            "{deficit} mm against the 8-day crop water requirement. Immediate canal "
            "release recommended within 48 hours to avoid yield loss."
        ),
        "Moderate": (
            "Moderate stress emerging at the {stage} stage. Deficit of {deficit} mm. "
            "Monitor over the next 8-day cycle; schedule irrigation if the deficit "
            "exceeds 25 mm at the next satellite pass."
        ),
        "Optimal": (
            "Crop water balance within normal range at the {stage} stage. Deficit of "
            "{deficit} mm, well inside tolerance. No action required this cycle."
        ),
    }
    level_by_status = {"Critical": "critical", "Moderate": "moderate", "Optimal": "optimal"}

    records = []
    for _, row in clusters.iterrows():
        if row["crop"] == "Fallow":
            continue
        message = narrative_by_status[row["status"]].format(
            stage=row["growth_stage"].lower(), deficit=row["deficit_mm"]
        )
        records.append(
            {
                "level": level_by_status[row["status"]],
                "cluster": row["cluster_id"],
                "crop": row["crop"],
                "deficit_mm": row["deficit_mm"],
                "distance_to_canal_m": row["distance_to_canal_m"],
                "message": message,
            }
        )

    order = {"critical": 0, "moderate": 1, "optimal": 2}
    records.sort(key=lambda r: (order[r["level"]], -r["deficit_mm"]))
    return records


# ----------------------------------------------------------------------------
# Sidebar - Unified State-Wide Search & Navigation Engine
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛰️ Platform Navigation")

    selected_region = st.selectbox(
        "🔍 Search Target Command District / Region",
        REGION_NAMES,
        index=0,
        help="Selecting a region re-centers the map, refilters all field clusters, "
        "and updates every analytics card on the page.",
    )

    st.markdown("---")
    st.markdown("#### 📐 Policy Sandbox")
    simulate_expansion = st.checkbox(
        "📐 Simulate Projected Canal Expansion",
        value=False,
        help="Projects a new distributary extension to all currently stranded "
        "(>1000 m) critical-deficit fields in this region, instantly clearing "
        "their Infrastructure Gap warnings.",
    )

    st.markdown("---")
    season = st.selectbox(
        "🗓️ Target Agricultural Season",
        ["Rabi (Winter)", "Kharif (Monsoon)"],
        index=0,
    )

# ----------------------------------------------------------------------------
# Region-scoped data - rebuilt (from cache) whenever the sidebar selection
# changes, so every downstream card/tab/chart reflects the active region.
# ----------------------------------------------------------------------------
REGION_CFG = REGIONS[selected_region]
CANAL_LINE = REGION_CFG["canal"]
MAP_CENTER = REGION_CFG["center"]

FIELD_CLUSTERS = build_region_clusters(selected_region)

if simulate_expansion:
    ACTIVE_CLUSTERS = apply_canal_expansion_simulation(FIELD_CLUSTERS)
else:
    ACTIVE_CLUSTERS = FIELD_CLUSTERS.copy()
    ACTIVE_CLUSTERS["simulated_expansion"] = False

ADVISORIES = build_advisories(ACTIVE_CLUSTERS)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown('<div class="app-header-eyebrow">ISRO-ALIGNED SATELLITE INTELLIGENCE · STATE-WIDE DEPLOYMENT</div>', unsafe_allow_html=True)
st.title("🛰️ Gujarat State Crop Intelligence & Irrigation Planning Platform")

st.markdown(
    f"""
    <div class="meta-banner">
        <span><span class="dot-live">●</span> Data Pipeline Status: ACTIVE &nbsp;|&nbsp;
        Sensor Stack: ISRO EOS-04 (C-band SAR) + Sentinel-2B (Multispectral Optical) Fused Processing Layer &nbsp;|&nbsp;
        Temporal Cadence: 8-Day Best-Pixel Composite Integration Mode</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    f"Active Region: **{selected_region}** · AI-driven crop classification, phenology-aware moisture "
    "stress monitoring, and two-way canal proximity policy advisory."
)

st.markdown("---")

# ----------------------------------------------------------------------------
# Top row - Metrics panel  [RETAINED structure, now region-reactive]
# ----------------------------------------------------------------------------
metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    region_area_sq_km = 1420 + (hash(selected_region) % 600)  # stable per-region pseudo-area
    st.metric(label="📏 Total Area Evaluated", value=f"{region_area_sq_km:,} Sq Km")

with metric_col2:
    st.metric(
        label="🎯 Classification Accuracy",
        value="87.4%",
        delta="+2.4 pp vs. >85% ISRO Target Met",
        delta_color="normal",
    )

with metric_col3:
    n_critical = int((ACTIVE_CLUSTERS["status"] == "Critical").sum())
    st.metric(
        label="🚨 Active Critical Stress Alerts",
        value=f"{n_critical} Field Clusters",
        delta="+3 vs. last 8-day cycle",
        delta_color="inverse",
    )

st.markdown("---")

# ----------------------------------------------------------------------------
# Central body - Tabs
# ----------------------------------------------------------------------------
tab_map, tab_trends, tab_logistics = st.tabs(
    [
        "🛰️ Geospatial Control Center",
        "📈 Multi-Temporal Satellite Analytics",
        "💧 Canal Logistics & Proximity Alerts",
    ]
)

# ============================================================================
# TAB 1 - Geospatial Control Center
# ============================================================================
with tab_map:
    control_col, map_col = st.columns([1, 2], gap="large")

    with control_col:
        st.markdown('<div class="control-card">', unsafe_allow_html=True)
        st.subheader("🌾 Monitoring Controls")

        layer_view = st.radio(
            "🗺️ Map Visualization Layer",
            ["🌱 Crop Classification View", "💧 Canopy Moisture Stress View"],
            index=0,
            help="Switches the map's color encoding between crop identity and moisture stress.",
        )

        crop_class = st.selectbox(
            "🌱 Target Crop Class",
            CROP_FILTER_OPTIONS,
            index=0,
            help="Filters both map views to a single crop type, or shows all crops.",
        )

        st.markdown("&nbsp;", unsafe_allow_html=True)

        if layer_view.startswith("🌱"):
            st.markdown("**Crop Identity Legend**")
            legend_html = "".join(
                f'<span class="legend-chip"><span class="legend-dot" '
                f'style="background:{color}"></span>{crop}</span>'
                for crop, color in CROP_COLORS.items()
            )
        else:
            st.markdown("**Stress Legend**")
            legend_html = "".join(
                f'<span class="legend-chip"><span class="legend-dot" '
                f'style="background:{color}"></span>{status}</span>'
                for status, color in STATUS_COLORS.items()
            )
        st.markdown(f"<div>{legend_html}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        view_clusters = ACTIVE_CLUSTERS.copy()
        if crop_class != "All Crops":
            view_clusters = view_clusters[view_clusters["crop"] == crop_class]

        st.markdown("&nbsp;")
        st.caption(
            f"Showing **{len(view_clusters)}** field clusters in **{selected_region}** · "
            f"**{crop_class}** · {season}."
        )

    with map_col:
        narmada_map = folium.Map(
            location=[MAP_CENTER[0], MAP_CENTER[1]],
            zoom_start=10,
            tiles="cartodbpositron",
            control_scale=True,
        )

        folium.PolyLine(
            locations=CANAL_LINE,
            color="#1565C0",
            weight=3,
            opacity=0.55,
            dash_array="6,6",
            tooltip=f"Primary Distributary Line - {selected_region} (simulated)",
        ).add_to(narmada_map)

        crop_mode = layer_view.startswith("🌱")

        for _, row in view_clusters.iterrows():
            color = CROP_COLORS[row["crop"]] if crop_mode else STATUS_COLORS[row["status"]]
            half_box = 0.016

            tooltip_html = (
                f"<b>{row['cluster_id']}</b> · {row['crop']}<br>"
                f"Stage: {row['growth_stage']} [{stage_progress_bar(row['growth_stage'])}]<br>"
                f"Moisture Status: {row['status']}<br>"
                f"Deficit: {row['deficit_mm']} mm<br>"
                f"Canal Distance: {row['distance_to_canal_m']:,} m"
                + (" (projected)" if row.get("simulated_expansion") else "")
            )

            folium.Rectangle(
                bounds=[
                    [row["lat"] - half_box, row["lon"] - half_box],
                    [row["lat"] + half_box, row["lon"] + half_box],
                ],
                color=color,
                weight=2,
                fill=True,
                fill_color=color,
                fill_opacity=0.30,
                tooltip=tooltip_html,
            ).add_to(narmada_map)

            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
            ).add_to(narmada_map)

        st_folium(narmada_map, use_container_width=True, height=520, returned_objects=[])

# ============================================================================
# TAB 2 - Multi-Temporal Satellite Analytics
# ============================================================================
with tab_trends:
    st.subheader("📈 Multi-Temporal Satellite Analytics")
    st.caption(
        f"8-day composite NDVI/NDWI history alongside a 5-year root-zone moisture baseline "
        f"comparison for **{selected_region}**."
    )

    chart_col1, chart_col2 = st.columns(2, gap="large")

    with chart_col1:
        st.markdown("##### 🛰️ 8-Day Composite NDVI / NDWI - Last 90 Days")

        fig_ts = go.Figure()

        fig_ts.add_vrect(
            x0=DECLINE_START_DAY,
            x1=CRASH_DAY,
            fillcolor=COLOR_SECONDARY,
            opacity=0.12,
            line_width=0,
            annotation_text="Pre-Visual Detection Window",
            annotation_position="top left",
            annotation=dict(font_size=10, font_color=COLOR_MUTED),
        )

        fig_ts.add_vline(
            x=CRASH_DAY,
            line_width=1.5,
            line_dash="dot",
            line_color=COLOR_CRITICAL,
            annotation_text="Canopy Moisture Crash",
            annotation_position="top right",
            annotation=dict(font_size=10, font_color=COLOR_CRITICAL),
        )

        fig_ts.add_trace(
            go.Scatter(
                x=NDVI_NDWI_DF["days_ago"], y=NDVI_NDWI_DF["NDVI"],
                mode="lines+markers", name="NDVI",
                line=dict(color=COLOR_PRIMARY, width=3),
                marker=dict(size=5, color=COLOR_PRIMARY),
            )
        )
        fig_ts.add_trace(
            go.Scatter(
                x=NDVI_NDWI_DF["days_ago"], y=NDVI_NDWI_DF["NDWI"],
                mode="lines+markers", name="NDWI",
                line=dict(color="#1565C0", width=3),
                marker=dict(size=5, color="#1565C0"),
            )
        )

        fig_ts.update_layout(
            height=440,
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor=COLOR_SURFACE,
            paper_bgcolor=COLOR_SURFACE,
            xaxis_title="Days Relative to Today",
            yaxis_title="Index Value",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            font=dict(color=COLOR_TEXT),
            xaxis=dict(gridcolor=COLOR_BORDER, zeroline=False),
            yaxis=dict(gridcolor=COLOR_BORDER, zeroline=False),
        )

        st.plotly_chart(fig_ts, use_container_width=True)
        st.caption(
            f"📌 Decline becomes statistically detectable **{abs(DECLINE_START_DAY)} days ago** - "
            f"**{abs(DECLINE_START_DAY - CRASH_DAY)} days** before the canopy moisture crash "
            f"became operationally obvious on day {CRASH_DAY}."
        )

    with chart_col2:
        st.markdown("##### 📊 Root-Zone Moisture - Current Season vs. 5-Year Baseline")

        fig_baseline = go.Figure()
        fig_baseline.add_trace(
            go.Scatter(
                x=BASELINE_DF["day"], y=BASELINE_DF["5-Year Historical Baseline Average"],
                mode="lines", name="5-Year Historical Baseline Average",
                line=dict(color=COLOR_MUTED, width=2, dash="dash"),
            )
        )
        fig_baseline.add_trace(
            go.Scatter(
                x=BASELINE_DF["day"], y=BASELINE_DF["Current Season Root-Zone Moisture"],
                mode="lines+markers", name="Current Season Root-Zone Moisture",
                line=dict(color=COLOR_PRIMARY, width=3),
                marker=dict(size=5, color=COLOR_PRIMARY),
            )
        )

        fig_baseline.update_layout(
            height=440,
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor=COLOR_SURFACE,
            paper_bgcolor=COLOR_SURFACE,
            xaxis_title="Days Since Sowing",
            yaxis_title="Root-Zone Moisture (relative index)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            font=dict(color=COLOR_TEXT),
            xaxis=dict(gridcolor=COLOR_BORDER, zeroline=False),
            yaxis=dict(gridcolor=COLOR_BORDER, zeroline=False),
        )

        st.plotly_chart(fig_baseline, use_container_width=True)
        st.caption(
            "📌 The current season profile falls below the 5-year average through the "
            "mid-to-late vegetative pipeline - consistent with the deficits surfacing on the "
            "Geospatial Control Center map."
        )

# ============================================================================
# TAB 3 - Canal Logistics & Proximity Alerts  [TWO-WAY TRIGGER UPGRADE]
# ============================================================================
with tab_logistics:
    st.subheader("💧 Canal Logistics & Proximity Alerts")
    st.caption(
        f"Two-way proximity advisory engine for **{selected_region}** - cross-referencing "
        "moisture deficit severity against direct distance to the nearest primary "
        "distributary line."
    )

    PROXIMITY_THRESHOLD_M = 1000

    critical_clusters = ACTIVE_CLUSTERS[ACTIVE_CLUSTERS["status"] == "Critical"]
    case_a_near = critical_clusters[critical_clusters["distance_to_canal_m"] <= PROXIMITY_THRESHOLD_M].sort_values(
        "distance_to_canal_m"
    )
    case_b_far = critical_clusters[critical_clusters["distance_to_canal_m"] > PROXIMITY_THRESHOLD_M].sort_values(
        "distance_to_canal_m"
    )

    if simulate_expansion and len(critical_clusters[critical_clusters.get("simulated_expansion", False)]) > 0:
        st.info(
            "📐 **Policy Sandbox Active** - projected canal expansion is currently simulated. "
            "Previously stranded fields below now show as resolved connections at a simulated "
            "300 m distance.",
            icon="📐",
        )

    # --- CASE A: Active Canal Operational Alert -----------------------
    st.markdown(f"#### 🚨 Case A — Active Canal Operational Alerts (≤ {PROXIMITY_THRESHOLD_M:,} m)")
    if len(case_a_near) > 0:
        for _, row in case_a_near.iterrows():
            label = " (post-expansion)" if row.get("simulated_expansion") else ""
            st.error(
                f"**{row['cluster_id']} · {row['crop']}** — {row['distance_to_canal_m']:,} m from "
                f"canal{label}\n\n"
                f"Critical moisture deficit of **{row['deficit_mm']} mm** at the "
                f"**{row['growth_stage'].lower()}** stage. Operators should deploy an "
                f"**immediate gravity-fed gate-valve action** at the nearest local control "
                f"structure to release localized irrigation relief within the current shift.",
                icon="🚨",
            )
    else:
        st.success("✅ No canal-adjacent critical-deficit fields require gate-valve action this cycle.", icon="✅")

    st.markdown("---")

    # --- CASE B: Infrastructure Gap Policy Advisory ---------------------
    st.markdown(f"#### ⚠️ Case B — Infrastructure Gap Policy Advisories (> {PROXIMITY_THRESHOLD_M:,} m)")
    if len(case_b_far) > 0:
        for _, row in case_b_far.iterrows():
            st.warning(
                f"**{row['cluster_id']} · {row['crop']}** — {row['distance_to_canal_m']:,} m from "
                f"canal\n\n"
                f"🏗️ **Infrastructure Gap Zone.** Critical deficit of **{row['deficit_mm']} mm** at the "
                f"**{row['growth_stage'].lower()}** stage, but this cluster sits beyond gravity-fed "
                f"relief range. Policy recommendation: log a **proposed minor distributary pipeline "
                f"extension** under state micro-irrigation layout budgets to bring this cluster "
                f"within operational range.",
                icon="⚠️",
            )
    else:
        st.markdown(
            """
            <div class="resolved-callout">
                <div class="resolved-callout-title">✅ No Infrastructure Gap Zones</div>
                <div>Every critical-deficit field cluster in this region is currently within
                gravity-fed canal relief range — either naturally, or via the active canal
                expansion simulation.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### 📋 Full Field Proximity Log")

    proximity_log = ACTIVE_CLUSTERS[
        ["cluster_id", "crop", "growth_stage", "status", "deficit_mm", "distance_to_canal_m", "simulated_expansion"]
    ].rename(
        columns={
            "cluster_id": "Cluster ID",
            "crop": "Crop",
            "growth_stage": "Growth Stage",
            "status": "Moisture Status",
            "deficit_mm": "Deficit (mm)",
            "distance_to_canal_m": "Distance to Nearest Canal Line (m)",
            "simulated_expansion": "Projected Expansion Applied",
        }
    ).sort_values("Distance to Nearest Canal Line (m)")

    st.dataframe(
        proximity_log,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Moisture Status": st.column_config.TextColumn(width="small"),
            "Distance to Nearest Canal Line (m)": st.column_config.NumberColumn(format="%d m"),
            "Deficit (mm)": st.column_config.NumberColumn(format="%d mm"),
        },
    )

    # --- Documented Infrastructure Data Export --------------------------
    st.markdown("&nbsp;")
    export_df = case_b_far[
        ["cluster_id", "crop", "growth_stage", "deficit_mm", "distance_to_canal_m"]
    ].rename(
        columns={
            "cluster_id": "Cluster ID",
            "crop": "Crop",
            "growth_stage": "Growth Stage",
            "deficit_mm": "Deficit (mm)",
            "distance_to_canal_m": "Current Distance to Canal (m)",
        }
    ).copy()
    export_df["Region"] = selected_region
    export_df["Proposed Extension Type"] = "Minor Distributary Pipeline"
    export_df["Recommended Budget Line"] = "State Micro-Irrigation Layout Budget"
    export_df["Snapshot Date"] = dt.date.today().isoformat()

    csv_buffer = io.StringIO()
    export_df.to_csv(csv_buffer, index=False)

    st.download_button(
        label="📥 Export Regional Canal Extension Proposals (CSV)",
        data=csv_buffer.getvalue(),
        file_name=f"canal_extension_proposals_{selected_region.split('(')[0].strip().replace(' ', '_')}.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown("#### 📋 Field-Level Irrigation Advisories")

    critical_items = [a for a in ADVISORIES if a["level"] == "critical"]
    moderate_items = [a for a in ADVISORIES if a["level"] == "moderate"]
    optimal_items = [a for a in ADVISORIES if a["level"] == "optimal"]

    if critical_items:
        st.markdown("##### 🔴 Critical — Action Required")
        for item in critical_items:
            st.error(
                f"**{item['cluster']} · {item['crop']}** "
                f"({item['distance_to_canal_m']:,} m from canal)\n\n{item['message']}",
                icon="🚨",
            )

    if moderate_items:
        st.markdown("##### 🟠 Moderate — Monitor Closely")
        for item in moderate_items:
            st.warning(
                f"**{item['cluster']} · {item['crop']}** "
                f"({item['distance_to_canal_m']:,} m from canal)\n\n{item['message']}",
                icon="⚠️",
            )

    if optimal_items:
        st.markdown("##### 🟢 Optimal — No Action Needed")
        for item in optimal_items:
            st.success(
                f"**{item['cluster']} · {item['crop']}** "
                f"({item['distance_to_canal_m']:,} m from canal)\n\n{item['message']}",
                icon="✅",
            )

    st.markdown("---")
    st.caption(
        f"Advisory snapshot generated {dt.datetime.now():%d %b %Y, %H:%M} IST · Region: "
        f"{selected_region} · Next satellite revisit cycle in 8 days · "
        "Model: AquaSense Stage-Aware Water Balance v1"
    )