import os
import math
import numpy as np
import streamlit as st
import pydeck as pdk

# -------------------------
# Geometry helpers
# -------------------------

EARTH_RADIUS_M = 6371000.0

def wrap_deg(x: float) -> float:
    x = x % 360.0
    return x + 360.0 if x < 0 else x

def destination_point(lat_deg: float, lon_deg: float, bearing_deg: float, distance_m: float):
    """
    Great-circle destination point given start lat/lon, bearing, distance.
    Returns (lat, lon) in degrees.
    """
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    brng = math.radians(bearing_deg)
    dr = distance_m / EARTH_RADIUS_M

    lat2 = math.asin(math.sin(lat1) * math.cos(dr) + math.cos(lat1) * math.sin(dr) * math.cos(brng))
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(dr) * math.cos(lat1),
        math.cos(dr) - math.sin(lat1) * math.sin(lat2)
    )

    return math.degrees(lat2), math.degrees(lon2)

def circle_polygon(lat: float, lon: float, radius_m: float, n: int = 120):
    """Approximate a circle as a polygon (list of [lon, lat])."""
    pts = []
    for b in np.linspace(0, 360, n, endpoint=False):
        lat2, lon2 = destination_point(lat, lon, float(b), radius_m)
        pts.append([lon2, lat2])
    pts.append(pts[0])  # close ring
    return pts

def wedge_polygon(lat: float, lon: float, center_bearing_deg: float, half_angle_deg: float, radius_m: float, n: int = 40):
    """
    Sector / wedge polygon:
    - Start at device point
    - Go along arc from (center-half_angle) to (center+half_angle)
    - Back to device point
    Output: list of [lon, lat]
    """
    start = center_bearing_deg - half_angle_deg
    end = center_bearing_deg + half_angle_deg

    arc = []
    for b in np.linspace(start, end, n):
        lat2, lon2 = destination_point(lat, lon, float(b), radius_m)
        arc.append([lon2, lat2])

    return [[lon, lat]] + arc + [[lon, lat]]  # close at device


# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Drone Direction Map MVP", layout="wide")

token = os.getenv("MAPBOX_TOKEN")
if not token:
    st.error("MAPBOX_TOKEN environment variable is not set. Set it and rerun.")
    st.stop()

st.title("Drone Detection Direction on Map (Mapbox + Python)")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Device location")
    lat = st.number_input("Latitude", value=31.7780, format="%.6f")   # Jerusalem-ish default
    lon = st.number_input("Longitude", value=35.2350, format="%.6f")

with col2:
    st.subheader("Device heading / azimuth")
    device_azimuth = st.slider("Device azimuth (deg, 0=N)", 0.0, 360.0, 0.0, 0.1)

with col3:
    st.subheader("Detection")
    bearing_mode = st.selectbox("Bearing mode", ["Relative to device forward", "Absolute (true north)"])
    detected_bearing = st.slider("Detected bearing (deg)", 0.0, 360.0, 45.0, 0.1)
    confidence = st.slider("Confidence", 0.0, 1.0, 0.8, 0.01)

# Compute absolute bearing
if bearing_mode == "Relative to device forward":
    bearing_abs = wrap_deg(device_azimuth + detected_bearing)
else:
    bearing_abs = wrap_deg(detected_bearing)

# Overlay configuration
st.sidebar.header("Overlay settings")
ring_radius_m = st.sidebar.slider("Ring radius (meters)", 50, 3000, 800, 50)
wedge_radius_m = st.sidebar.slider("Wedge radius (meters)", 50, 5000, 1200, 50)
half_angle = st.sidebar.slider("Wedge half-angle (deg)", 1.0, 30.0, 8.0, 0.5)
auto_zoom = st.sidebar.checkbox("Auto zoom to device", value=True)
pitch = st.sidebar.slider("Map pitch", 0, 70, 45, 1)

# Create geometries
ring = circle_polygon(lat, lon, ring_radius_m)
wedge = wedge_polygon(lat, lon, bearing_abs, half_angle, wedge_radius_m)

# Data for layers
device_point = [{"position": [lon, lat], "label": "Device"}]
ring_poly = [{"polygon": ring}]
wedge_poly = [{
    "polygon": wedge,
    "confidence": confidence,
    "bearing_abs": bearing_abs
}]

# Color logic (simple): higher confidence => more opaque
# pydeck wants RGBA [0..255]; we keep it simple and readable.
alpha = int(60 + 160 * confidence)  # 60..220
wedge_fill = [255, 0, 0, alpha]     # red with alpha

layers = [
    # Wedge sector
    pdk.Layer(
        "PolygonLayer",
        data=wedge_poly,
        get_polygon="polygon",
        get_fill_color=wedge_fill,
        get_line_color=[255, 0, 0, 255],
        line_width_min_pixels=2,
        pickable=True,
    ),
    # 360 ring
    pdk.Layer(
        "PolygonLayer",
        data=ring_poly,
        get_polygon="polygon",
        get_fill_color=[0, 0, 0, 0],
        get_line_color=[0, 255, 255, 180],
        line_width_min_pixels=2,
        pickable=False,
    ),
    # Device marker
    pdk.Layer(
        "ScatterplotLayer",
        data=device_point,
        get_position="position",
        get_radius=10,
        radius_min_pixels=6,
        get_fill_color=[0, 255, 0, 220],
        pickable=True,
    ),
]

tooltip = {
    "html": "<b>Bearing:</b> {bearing_abs}°<br/><b>Confidence:</b> {confidence}",
    "style": {"backgroundColor": "rgba(0,0,0,0.7)", "color": "white"},
}

view_state = pdk.ViewState(
    latitude=lat,
    longitude=lon,
    zoom=16 if auto_zoom else 12,
    pitch=pitch,
    bearing=0,  # keep map north-up; overlays show direction
)

deck = pdk.Deck(
    map_style="mapbox://styles/mapbox/satellite-streets-v12",
    initial_view_state=view_state,
    layers=layers,
    tooltip=tooltip,
)

st.pydeck_chart(deck, width=True)

st.markdown(
    f"""
**Computed absolute bearing:** `{bearing_abs:.1f}°`  
(0° = North, clockwise positive)
"""
)
