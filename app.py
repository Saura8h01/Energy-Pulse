import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Energy Pulse | Unified Building Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "energy_pulse_model.pkl"

@st.cache_resource
def load_model():
    # Loads the existing trained model. This application does not retrain it.
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("energy_pulse_model.pkl is missing. Keep it beside app.py.")
    st.stop()

try:
    bundle = load_model()
    model = bundle["model"]
    model_name = bundle.get("model_name", type(model).__name__)
    features = list(bundle["features"])
    defaults = bundle["defaults"]
    minimums = bundle["minimums"]
    maximums = bundle["maximums"]
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()

# Simulated 15-floor values for the visual demonstration only.
# They are not real building sensor data and are not model input values.
floor_data = []
for floor_number in range(1, 16):
    temperature = round(
        19 + 0.55 * floor_number + 2.0 * np.sin(floor_number * 0.8), 1
    )
    humidity = round(48 + 12 * np.sin(floor_number * 0.5), 1)
    if temperature < 20:
        status, color = "Cold", "#55C9FF"
    elif temperature < 26:
        status, color = "Normal", "#45E6B0"
    else:
        status, color = "Hot", "#FFB52E"
    floor_data.append({
        "floor": floor_number,
        "temperature": temperature,
        "humidity": humidity,
        "status": status,
        "color": color,
    })

# Human-readable names for the original UCI residential-room sensor features.
# The original feature keys remain unchanged when values are sent to the model.
FEATURE_LABELS = {
    "lights": "Living-area lights",
    "T1": "Kitchen temperature (T1)",
    "RH_1": "Kitchen humidity (RH_1)",
    "T2": "Living-room temperature (T2)",
    "RH_2": "Living-room humidity (RH_2)",
    "T3": "Laundry-room temperature (T3)",
    "RH_3": "Laundry-room humidity (RH_3)",
    "T4": "Office temperature (T4)",
    "RH_4": "Office humidity (RH_4)",
    "T5": "Bathroom temperature (T5)",
    "RH_5": "Bathroom humidity (RH_5)",
    "T6": "Outside-north temperature (T6)",
    "RH_6": "Outside-north humidity (RH_6)",
    "T7": "Ironing-room temperature (T7)",
    "RH_7": "Ironing-room humidity (RH_7)",
    "T8": "Teenager-room temperature (T8)",
    "RH_8": "Teenager-room humidity (RH_8)",
    "T9": "Parents'-room temperature (T9)",
    "RH_9": "Parents'-room humidity (RH_9)",
    "T_out": "Outdoor temperature",
    "Press_mm_hg": "Atmospheric pressure",
    "RH_out": "Outdoor humidity",
    "Windspeed": "Wind speed",
    "Visibility": "Visibility",
    "Tdewpoint": "Dew point",
}

FEATURE_GROUPS = [
    ("Lighting", ["lights"]),
    ("Kitchen", ["T1", "RH_1"]),
    ("Living room", ["T2", "RH_2"]),
    ("Laundry room", ["T3", "RH_3"]),
    ("Office", ["T4", "RH_4"]),
    ("Bathroom", ["T5", "RH_5"]),
    ("Outdoor north side", ["T6", "RH_6"]),
    ("Ironing room", ["T7", "RH_7"]),
    ("Teenager room", ["T8", "RH_8"]),
    ("Parents' room", ["T9", "RH_9"]),
    ("Outdoor weather", ["T_out", "RH_out", "Press_mm_hg", "Windspeed", "Visibility", "Tdewpoint"]),
]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --ep-bg: #071522;
  --ep-panel: #102233;
  --ep-panel-2: #142b3d;
  --ep-border: #29485b;
  --ep-text: #f4f8fb;
  --ep-muted: #c0d0dc;
  --ep-accent: #a3e635;
  --ep-accent-dark: #17240a;
}
html, body, [class*="css"], .stApp,
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"], input, textarea, button {
  font-family: 'Manrope', sans-serif !important;
}
h1, h2, h3, h4,
.ep-hero h1,
[data-testid="stMetricValue"] {
  font-family: 'Space Grotesk', 'Manrope', sans-serif !important;
  letter-spacing: -0.035em;
}
h1 { font-weight: 700 !important; }
h2, h3, h4 { font-weight: 600 !important; }
p, li, label, .stMarkdown, [data-testid="stCaptionContainer"] {
  font-weight: 500;
  line-height: 1.65;
}
[data-testid="stMetricLabel"] {
  font-family: 'Manrope', sans-serif !important;
  font-weight: 600 !important;
  letter-spacing: 0.01em;
}
.stApp {
  background: radial-gradient(ellipse at 70% 0%, #17364b 0%, transparent 42%), var(--ep-bg);
  color: var(--ep-text);
}
header[data-testid="stHeader"] { background: rgba(7,21,34,.88); }
[data-testid="stSidebar"] { display: none; }
h1, h2, h3, h4, p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
  color: var(--ep-text);
}
/* Keep Streamlit notices readable on their own backgrounds. */
[data-testid="stAlert"] {
  background: #132a3b !important;
  border: 1px solid #36566b !important;
}
[data-testid="stAlert"] p,
[data-testid="stAlert"] div,
[data-testid="stAlert"] span,
[data-testid="stAlert"] svg {
  color: #f4f8fb !important;
}
/* Expander headers must stay dark with bright text (Streamlit defaults can be white). */
div[data-testid="stExpander"] details,
div[data-testid="stExpander"] details > summary {
  background: #132a3b !important;
  color: #f4f8fb !important;
  border-color: #36566b !important;
}
div[data-testid="stExpander"] details > summary p,
div[data-testid="stExpander"] details > summary span,
div[data-testid="stExpander"] details > summary svg {
  color: #f4f8fb !important;
  fill: #f4f8fb !important;
}
div[data-testid="stExpander"] details > summary:hover {
  background: #1a394e !important;
}
div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
  background: #102436 !important;
  color: #f4f8fb !important;
}
/* Restore dark text inside intentionally light input controls. */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] input,
[data-testid="stDateInput"] input,
[data-testid="stTimeInput"] input {
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
}
h1, h2, h3 { font-weight: 700 !important; letter-spacing: .1px; }
.ep-hero {
  background: linear-gradient(110deg, #102b3d, #0c1c2b);
  border: 1px solid var(--ep-border);
  border-radius: 18px;
  padding: 22px 26px;
  margin: 2px 0 16px 0;
}
.ep-kicker { color: var(--ep-accent); font-size: 12px; font-weight: 700; letter-spacing: 3px; }
.ep-sub { color: var(--ep-muted); font-size: 16px; }
.ep-section {
  background: linear-gradient(145deg, #102436, #0b1b2a);
  border: 1px solid var(--ep-border);
  border-radius: 16px;
  padding: 17px 19px;
  margin: 12px 0;
}
div[data-testid="stMetric"] {
  background: #102638; border: 1px solid var(--ep-border);
  border-radius: 13px; padding: 14px;
}
[data-testid="stMetricLabel"] { color: var(--ep-muted) !important; }
[data-testid="stMetricValue"] { color: var(--ep-accent) !important; }
.stButton button, .stFormSubmitButton button {
  background: var(--ep-accent) !important;
  color: var(--ep-accent-dark) !important;
  border: 1px solid #c5f66b !important;
  border-radius: 9px !important;
  font-weight: 800 !important;
}
.stButton button:hover, .stFormSubmitButton button:hover {
  background: #b8f34f !important;
  color: #111d08 !important;
  border-color: #d4ff8c !important;
}
div[data-testid="stForm"] {
  background: #102436; border: 1px solid var(--ep-border);
  border-radius: 14px; padding: 18px;
}
/* Inputs are intentionally white with dark text for contrast, matching the supplied palette reference. */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stDateInput"] input,
[data-testid="stTimeInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
  background: #f5f8fc !important;
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
  border-color: #cbd5e1 !important;
}
[data-testid="stNumberInput"] button {
  background: #e7edf4 !important;
  color: #111827 !important;
}
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label { color: var(--ep-text) !important; }
[data-baseweb="popover"], [role="listbox"] {
  background: #f5f8fc !important; color: #111827 !important;
}
[data-baseweb="popover"] * { color: #111827 !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--ep-border); }
hr { border-color: var(--ep-border); }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
  background: #132a3b; color: #f4f8fb; border-radius: 8px 8px 0 0;
}
.stTabs [aria-selected="true"] {
  background: var(--ep-accent) !important; color: #17240a !important;
}
</style>
""", unsafe_allow_html=True)

def _lookup(container, key, fallback):
    try:
        value = container[key]
        if pd.isna(value):
            return fallback
        return float(value)
    except Exception:
        return fallback

def feature_bounds(feature):
    default = _lookup(defaults, feature, 0.0)
    low = _lookup(minimums, feature, default - abs(default) - 1.0)
    high = _lookup(maximums, feature, default + abs(default) + 1.0)
    if low > high:
        low, high = high, low
    if low == high:
        low -= 1.0
        high += 1.0
    default = min(max(default, low), high)
    return low, high, default

def render_feature_inputs():
    """Render all model inputs in grouped room/zone sections, retaining exact model keys."""
    values = {}
    assigned = set()
    group_map = []
    for group_name, group_features in FEATURE_GROUPS:
        available = [name for name in group_features if name in features]
        if available:
            group_map.append((group_name, available))
            assigned.update(available)
    remaining = [name for name in features if name not in assigned]
    if remaining:
        group_map.append(("Additional model inputs", remaining))

    for group_name, group_features in group_map:
        with st.expander(group_name, expanded=(group_name in ("Kitchen", "Living room", "Lighting"))):
            cols = st.columns(2)
            for index, feature in enumerate(group_features):
                low, high, default = feature_bounds(feature)
                step = max((high - low) / 100.0, 0.01)
                with cols[index % 2]:
                    values[feature] = st.number_input(
                        FEATURE_LABELS.get(feature, feature),
                        min_value=float(low),
                        max_value=float(high),
                        value=float(default),
                        step=float(step),
                        key=f"input_{feature}",
                        help=f"Original model input: {feature}. Saved training range: {low:.3f} to {high:.3f}",
                    )
    # Keep exactly the original feature order expected by the trained estimator.
    return {feature: values[feature] for feature in features}

def render_prediction():
    st.markdown("### Energy prediction")
    st.caption(
        "Adjust the room and weather inputs below, then estimate appliance energy "
        "for the same observation. This is not a future-time forecast."
    )
    with st.form("energy_prediction_form"):
        input_values = render_feature_inputs()
        submitted = st.form_submit_button("Run energy prediction", use_container_width=True)
    if submitted:
        try:
            input_frame = pd.DataFrame([[input_values[name] for name in features]], columns=features)
            prediction = float(np.asarray(model.predict(input_frame)).reshape(-1)[0])
            st.session_state["last_energy_prediction"] = prediction
            st.session_state["last_energy_inputs"] = input_values
            st.success("Prediction completed using your existing trained model.")
            st.metric("Predicted appliance energy", f"{prediction:,.2f} Wh")
            st.caption("Target unit: Wh per 10-minute observation, consistent with the dataset.")
        except Exception as error:
            st.error(f"Prediction failed: {error}")

def render_analytics():
    st.markdown("### Model analytics")
    a, b = st.columns(2)
    a.metric("Loaded model", str(model_name))
    b.metric("Model input features", len(features))

    saved_metrics = bundle.get("metrics") if isinstance(bundle, dict) else None
    if isinstance(saved_metrics, dict) and saved_metrics:
        st.markdown("#### Saved evaluation metrics")
        metric_values = [(str(k), v) for k, v in saved_metrics.items()
                         if isinstance(v, (int, float, np.number))]
        if metric_values:
            metric_cols = st.columns(min(4, len(metric_values)))
            for i, (name, value) in enumerate(metric_values):
                metric_cols[i % len(metric_cols)].metric(name, f"{float(value):.4f}")
        else:
            st.write(saved_metrics)
    else:
        st.info(
            "The saved model bundle does not contain evaluation metrics. "
            "This page will not invent accuracy, MAE, RMSE or R² values."
        )

    importances = getattr(model, "feature_importances_", None)
    coefficients = getattr(model, "coef_", None)
    if importances is not None and len(importances) == len(features):
        plot_df = pd.DataFrame({"Feature": features, "Importance": importances})
        plot_df = plot_df.sort_values("Importance", ascending=True)
        fig = px.bar(
            plot_df, x="Importance", y="Feature", orientation="h",
            title="Feature importance from the loaded model",
            color="Importance", color_continuous_scale=["#3b82f6", "#a3e635"],
        )
        fig.update_layout(
            paper_bgcolor="#071522", plot_bgcolor="#102436",
            font_color="#f4f8fb", height=max(430, len(features) * 24),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    elif coefficients is not None and len(np.asarray(coefficients).reshape(-1)) == len(features):
        plot_df = pd.DataFrame({
            "Feature": features,
            "Coefficient": np.asarray(coefficients).reshape(-1),
        }).sort_values("Coefficient", ascending=True)
        fig = px.bar(
            plot_df, x="Coefficient", y="Feature", orientation="h",
            title="Linear model coefficients",
            color="Coefficient", color_continuous_scale=["#3b82f6", "#a3e635"],
        )
        fig.update_layout(
            paper_bgcolor="#071522", plot_bgcolor="#102436",
            font_color="#f4f8fb", height=max(430, len(features) * 24),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("The loaded estimator does not expose feature importance or coefficients.")

def render_floor_inspector():
    st.markdown("### Floor inspector")
    st.caption("This is a separate simulated building display. The source dataset contains room sensors, not 15 floors.")
    selected = st.select_slider("Select demonstration floor", options=list(range(1, 16)), value=1)
    row = floor_data[selected - 1]
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Floor {selected} temperature", f"{row['temperature']} °C")
    c2.metric("Demonstration humidity", f"{row['humidity']}%")
    c3.metric("Temperature status", row["status"])



def show_building_viewer():

    html = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
* { box-sizing: border-box; }

body {
    margin: 0;
    overflow: hidden;
    background: #061321;
    font-family: 'Manrope', Arial, sans-serif;
}

#scene {
    position: absolute;
    inset: 0;
}

#topbar {
    position: absolute;
    z-index: 5;
    top: 15px;
    left: 18px;
    right: 18px;
    display: flex;
    justify-content: space-between;
    pointer-events: none;
}

#title {
    color: #69e3ff;
    font-size: 13px;
    font-weight: bold;
    letter-spacing: 3px;
}

#subtitle {
    color: #a4c7dc;
    font-size: 10px;
    margin-top: 6px;
}

#online {
    color: #70ffc4;
    background: rgba(12, 70, 58, .85);
    border: 1px solid #37a987;
    padding: 7px 12px;
    border-radius: 20px;
    font-size: 10px;
    height: 28px;
}

#floorPanel {
    position: absolute;
    z-index: 5;
    right: 17px;
    bottom: 17px;
    width: 205px;
    background: rgba(5, 24, 40, .94);
    border: 1px solid #3283a5;
    border-radius: 12px;
    padding: 15px;
    color: white;
    pointer-events: none;
    box-shadow: 0 0 25px rgba(0, 180, 255, .12);
}

#floorName {
    color: #72e2ff;
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 1px;
}

#temperature {
    font-size: 31px;
    font-weight: bold;
    margin: 10px 0;
}

#humidity, #floorStatus {
    color: #b7d0df;
    font-size: 11px;
    margin-top: 6px;
}

#reset {
    position: absolute;
    right: 17px;
    top: 64px;
    z-index: 6;
    background: #10344d;
    color: #8ceaff;
    border: 1px solid #3286a8;
    border-radius: 7px;
    padding: 9px 13px;
    cursor: pointer;
}

#reset:hover {
    background: #155475;
}

#legend {
    position: absolute;
    left: 17px;
    bottom: 17px;
    z-index: 5;
    background: rgba(5, 24, 40, .9);
    border: 1px solid #26546d;
    border-radius: 10px;
    padding: 12px;
    color: #d7eafa;
    font-size: 10px;
    line-height: 2;
    pointer-events: none;
}

.dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 7px;
}

canvas { display: block; }
</style>
</head>

<body>

<div id="scene"></div>

<div id="topbar">
    <div>
        <div id="title">ENERGY PULSE / DIGITAL TWIN</div>
        <div id="subtitle">15-LEVEL HOLOGRAPHIC BUILDING INTELLIGENCE</div>
    </div>
    <div id="online">● SYSTEM ONLINE</div>
</div>

<button id="reset">↺ Reset Camera</button>

<div id="floorPanel">
    <div id="floorName">BUILDING OVERVIEW</div>
    <div id="temperature" style="color:#68dcff">15 FLOORS</div>
    <div id="humidity">Select a floor to inspect</div>
    <div id="floorStatus">Interactive building simulation</div>
</div>

<div id="legend">
    <div><span class="dot" style="background:#ff9b37"></span>Heating energy flow</div>
    <div><span class="dot" style="background:#52caff"></span>Cooling energy flow</div>
    <div><span class="dot" style="background:#51e9ba"></span>Normal temperature</div>
    <div style="color:#829eb3">Click a floor to inspect</div>
</div>

<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js"
  }
}
</script>

<script type="module">

import * as THREE from "three";
import { OrbitControls } from
"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js";

const FLOOR_DATA = __FLOOR_DATA__;

const container = document.getElementById("scene");

const scene = new THREE.Scene();
scene.background = new THREE.Color("#061321");
scene.fog = new THREE.FogExp2("#061321", 0.009);

const camera = new THREE.PerspectiveCamera(
    44,
    window.innerWidth / window.innerHeight,
    0.1,
    180
);

camera.position.set(29, 22, 34);

const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: false
});

renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.25;

container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
// Let the page scroll normally when the user uses the mouse wheel over the viewer.
// Floor selection and drag-to-rotate remain enabled.
controls.enableZoom = false;
controls.target.set(0, 9, 0);
controls.minDistance = 9;
controls.maxDistance = 50;
controls.maxPolarAngle = Math.PI / 2;

// --------------------------------------------------
// LIGHTING
// --------------------------------------------------

scene.add(new THREE.AmbientLight("#8bc9ed", 1.3));

const mainLight = new THREE.PointLight("#45cfff", 180, 70);
mainLight.position.set(8, 19, 10);
scene.add(mainLight);

const warmLight = new THREE.PointLight("#ff823b", 95, 28);
warmLight.position.set(0, 9, 0);
scene.add(warmLight);

const coolLight = new THREE.PointLight("#168fff", 100, 35);
coolLight.position.set(-7, 12, 0);
scene.add(coolLight);

// --------------------------------------------------
// GROUND AND BLUEPRINT GRID
// --------------------------------------------------

const grid = new THREE.GridHelper(
    42, 42, "#236d96", "#12364e"
);

grid.position.y = -0.5;
scene.add(grid);

const buildingWidth = 9;
const buildingDepth = 7;
const floorHeight = 1.12;
const floorSpacing = 1.35;
const floorCount = 15;
const totalHeight = floorCount * floorSpacing;

const floorObjects = [];
const floorHighlights = [];
const animatedParticles = [];

const blueMaterial = new THREE.MeshBasicMaterial({
    color: "#42bfff",
    transparent: true,
    opacity: 0.09,
    side: THREE.DoubleSide
});

// --------------------------------------------------
// MATERIAL HELPERS
// --------------------------------------------------

function glassMaterial(color, opacity = 0.16) {
    return new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity,
        side: THREE.DoubleSide,
        depthWrite: false
    });
}

function addBox(
    width, height, depth,
    x, y, z,
    material,
    parent = scene
) {
    const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(width, height, depth),
        material
    );

    mesh.position.set(x, y, z);
    parent.add(mesh);

    return mesh;
}

function addEdges(mesh, color = "#48caff") {

    const edgeGeometry = new THREE.EdgesGeometry(
        mesh.geometry
    );

    const edges = new THREE.LineSegments(
        edgeGeometry,
        new THREE.LineBasicMaterial({
            color,
            transparent: true,
            opacity: 0.7
        })
    );

    edges.position.copy(mesh.position);
    edges.rotation.copy(mesh.rotation);
    scene.add(edges);

    return edges;
}

// --------------------------------------------------
// ROOM DETAILS
// --------------------------------------------------

function createRoom(floor, x, z, y) {

    const room = new THREE.Group();
    scene.add(room);

    const roomColor = floor % 3 === 0
        ? "#3c91c8"
        : "#286f9f";

    // Transparent room walls
    const wallMaterial = glassMaterial(roomColor, 0.13);

    addBox(2.05, 0.92, 0.07, x, y, z - 0.75,
        wallMaterial, room);

    addBox(0.07, 0.92, 1.5, x - 1, y, z,
        wallMaterial, room);

    // Glowing window panel
    addBox(
        1.65, 0.62, 0.025,
        x, y + 0.04, z - 0.70,
        glassMaterial("#5bcaff", 0.25),
        room
    );

    // Window frame
    const windowFrame = new THREE.LineSegments(
        new THREE.EdgesGeometry(
            new THREE.BoxGeometry(1.7, 0.68, 0.04)
        ),
        new THREE.LineBasicMaterial({
            color: "#58cfff"
        })
    );

    windowFrame.position.set(x, y + 0.04, z - 0.72);
    room.add(windowFrame);

    // Furniture: desk
    addBox(
        0.85, 0.09, 0.42,
        x, y - 0.25, z + 0.1,
        new THREE.MeshBasicMaterial({color:"#315d78"}),
        room
    );

    // Computer monitor
    addBox(
        0.37, 0.25, 0.035,
        x, y - 0.08, z + 0.0,
        glassMaterial("#47bfff", 0.8),
        room
    );

    // Desk support
    addBox(
        0.07, 0.28, 0.07,
        x, y - 0.42, z + 0.1,
        new THREE.MeshBasicMaterial({color:"#537b92"}),
        room
    );

    // Interior light strip
    addBox(
        1.5, 0.025, 0.025,
        x, y + 0.46, z + 0.35,
        new THREE.MeshBasicMaterial({
            color: floor % 2 === 0 ? "#ffae54" : "#63dcff"
        }),
        room
    );

    return room;
}

// --------------------------------------------------
// SURROUNDING CITY SKYLINE
// Varied-height neighbouring buildings frame the main
// Energy Pulse building without covering its floors.
// These are decorative scene elements, not sensor data.
// --------------------------------------------------

function createSkylineBuilding(x, z, width, depth, floors, tint, seed) {
    const floorPitch = 1.18;
    const height = floors * floorPitch;
    const centerY = height / 2;

    const shellMaterial = new THREE.MeshBasicMaterial({
        color: tint,
        transparent: true,
        opacity: 0.12,
        side: THREE.DoubleSide,
        depthWrite: false
    });

    const shell = new THREE.Mesh(
        new THREE.BoxGeometry(width, height, depth),
        shellMaterial
    );
    shell.position.set(x, centerY, z);
    scene.add(shell);

    const outline = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(width, height, depth)),
        new THREE.LineBasicMaterial({
            color: tint,
            transparent: true,
            opacity: 0.38
        })
    );
    outline.position.copy(shell.position);
    scene.add(outline);

    // Subtle floor bands make each neighbouring tower read as a building.
    const bandMaterial = new THREE.MeshBasicMaterial({
        color: tint,
        transparent: true,
        opacity: 0.28
    });
    for (let f = 1; f < floors; f++) {
        const band = new THREE.Mesh(
            new THREE.BoxGeometry(width + 0.04, 0.025, depth + 0.04),
            bandMaterial
        );
        band.position.set(x, f * floorPitch, z);
        scene.add(band);
    }

    // Small illuminated window panels on the two visible-facing sides.
    const windowColors = ["#55cfff", "#7ce8ff", "#ffc078"];
    const windowMaterial = new THREE.MeshBasicMaterial({
        color: windowColors[seed % windowColors.length],
        transparent: true,
        opacity: 0.48
    });

    const cols = Math.max(2, Math.floor(width / 1.05));
    const rows = Math.max(2, floors);
    for (let row = 0; row < rows; row++) {
        const wy = row * floorPitch + 0.42;
        for (let col = 0; col < cols; col++) {
            const wx = x - width / 2 + (col + 0.5) * width / cols;
            // Front-facing facade (toward the central plaza).
            const frontWindow = new THREE.Mesh(
                new THREE.BoxGeometry(width / cols * 0.48, 0.38, 0.025),
                windowMaterial
            );
            frontWindow.position.set(wx, wy, z + depth / 2 + 0.018);
            scene.add(frontWindow);
        }

        const sideCols = Math.max(2, Math.floor(depth / 1.1));
        for (let col = 0; col < sideCols; col++) {
            const wz = z - depth / 2 + (col + 0.5) * depth / sideCols;
            const sideWindow = new THREE.Mesh(
                new THREE.BoxGeometry(0.025, 0.38, depth / sideCols * 0.48),
                windowMaterial
            );
            sideWindow.position.set(x + width / 2 + 0.018, wy, wz);
            scene.add(sideWindow);
        }
    }

    // A small rooftop cap adds a finished silhouette.
    const roofLine = new THREE.Mesh(
        new THREE.BoxGeometry(width + 0.18, 0.055, depth + 0.18),
        new THREE.MeshBasicMaterial({
            color: tint,
            transparent: true,
            opacity: 0.55
        })
    );
    roofLine.position.set(x, height + 0.035, z);
    scene.add(roofLine);
}

// Varied heights and setbacks create a small city-block skyline.
// Keep the central building's immediate footprint clear.
[
    {x:-15, z:-11, w:5.0, d:4.5, f:8,  c:"#287fa8"},
    {x: 15, z:-12, w:5.5, d:4.8, f:11, c:"#347fae"},
    {x:-16, z: 10, w:4.5, d:4.2, f:6,  c:"#2b91b6"},
    {x: 16, z: 11, w:5.0, d:4.5, f:9,  c:"#286b9a"},
    {x:-22, z:  0, w:4.2, d:4.0, f:12, c:"#326f9d"},
    {x: 22, z:  1, w:4.5, d:4.2, f:7,  c:"#2586a8"},
    {x: -8, z: 19, w:4.0, d:4.0, f:5,  c:"#327fa0"},
    {x:  9, z:-20, w:4.6, d:4.2, f:10, c:"#2a739f"}
].forEach((b, index) => {
    createSkylineBuilding(b.x, b.z, b.w, b.d, b.f, b.c, index);
});

// --------------------------------------------------
// CITY STREETS, SIDEWALKS AND PUBLIC REALM
// Decorative city environment around the Energy Pulse tower.
// All surrounding roads, vehicles and landscaping are visual
// scene elements, not live traffic or sensor information.
// --------------------------------------------------

// Matte city ground beneath the blueprint grid.
const cityGround = new THREE.Mesh(
    new THREE.PlaneGeometry(110, 110),
    new THREE.MeshBasicMaterial({ color: "#081923" })
);
cityGround.rotation.x = -Math.PI / 2;
cityGround.position.y = -0.62;
scene.add(cityGround);

// Road and sidewalk materials.
const asphaltMat = new THREE.MeshBasicMaterial({ color: "#172b38" });
const roadEdgeMat = new THREE.MeshBasicMaterial({
    color: "#3b829d", transparent: true, opacity: 0.62
});
const laneMat = new THREE.MeshBasicMaterial({
    color: "#a9d6df", transparent: true, opacity: 0.72
});
const sidewalkMat = new THREE.MeshBasicMaterial({
    color: "#244454", transparent: true, opacity: 0.9
});

function groundSlab(width, depth, x, z, material, y = -0.48) {
    const slab = new THREE.Mesh(
        new THREE.BoxGeometry(width, 0.12, depth),
        material
    );
    slab.position.set(x, y, z);
    scene.add(slab);
    return slab;
}

// Four sides of a broad perimeter boulevard leave the central
// tower and its neighbouring buildings visible.
const boulevardOffset = 29;
const boulevardWidth = 6.5;
groundSlab(66, boulevardWidth, 0, -boulevardOffset, asphaltMat);
groundSlab(66, boulevardWidth, 0,  boulevardOffset, asphaltMat);
groundSlab(boulevardWidth, 66, -boulevardOffset, 0, asphaltMat);
groundSlab(boulevardWidth, 66,  boulevardOffset, 0, asphaltMat);

// Raised sidewalks outside each road edge.
const sidewalkOffset = boulevardWidth / 2 + 1.15;
for (const sign of [-1, 1]) {
    groundSlab(66, 1.6, 0, sign * (boulevardOffset + sidewalkOffset), sidewalkMat, -0.39);
    groundSlab(1.6, 66, sign * (boulevardOffset + sidewalkOffset), 0, sidewalkMat, -0.39);
}

// Road-edge lines and dashed center markings.
for (const z of [-boulevardOffset, boulevardOffset]) {
    for (const edge of [-1, 1]) {
        const line = new THREE.Mesh(
            new THREE.BoxGeometry(65, 0.025, 0.055),
            roadEdgeMat
        );
        line.position.set(0, -0.405, z + edge * (boulevardWidth / 2 - 0.22));
        scene.add(line);
    }
    for (let x = -31; x <= 31; x += 5) {
        const dash = new THREE.Mesh(
            new THREE.BoxGeometry(2.2, 0.025, 0.10),
            laneMat
        );
        dash.position.set(x, -0.397, z);
        scene.add(dash);
    }
}
for (const x of [-boulevardOffset, boulevardOffset]) {
    for (const edge of [-1, 1]) {
        const line = new THREE.Mesh(
            new THREE.BoxGeometry(0.055, 0.025, 65),
            roadEdgeMat
        );
        line.position.set(x + edge * (boulevardWidth / 2 - 0.22), -0.405, 0);
        scene.add(line);
    }
    for (let z = -31; z <= 31; z += 5) {
        const dash = new THREE.Mesh(
            new THREE.BoxGeometry(0.10, 0.025, 2.2),
            laneMat
        );
        dash.position.set(x, -0.397, z);
        scene.add(dash);
    }
}

// Crosswalks at the four boulevard corners.
for (const xSign of [-1, 1]) {
    for (const zSign of [-1, 1]) {
        for (let stripe = 0; stripe < 5; stripe++) {
            const crosswalk = new THREE.Mesh(
                new THREE.BoxGeometry(0.55, 0.035, 3.8),
                laneMat
            );
            crosswalk.position.set(
                xSign * (boulevardOffset - 4.0 + stripe * 0.8),
                -0.39,
                zSign * boulevardOffset
            );
            scene.add(crosswalk);
        }
    }
}

// Street lamps: slim poles with soft cyan-white light heads.
function addStreetLamp(x, z, rotation = 0) {
    const pole = new THREE.Mesh(
        new THREE.CylinderGeometry(0.055, 0.075, 3.0, 8),
        new THREE.MeshBasicMaterial({ color: "#6d9caf" })
    );
    pole.position.set(x, 1.0, z);
    scene.add(pole);

    const arm = new THREE.Mesh(
        new THREE.BoxGeometry(0.85, 0.06, 0.06),
        new THREE.MeshBasicMaterial({ color: "#6d9caf" })
    );
    arm.position.set(x + Math.cos(rotation) * 0.38, 2.45, z + Math.sin(rotation) * 0.38);
    arm.rotation.y = rotation;
    scene.add(arm);

    const lamp = new THREE.Mesh(
        new THREE.SphereGeometry(0.16, 10, 8),
        new THREE.MeshBasicMaterial({
            color: "#a9efff",
            transparent: true,
            opacity: 0.95
        })
    );
    lamp.position.set(x + Math.cos(rotation) * 0.76, 2.38, z + Math.sin(rotation) * 0.76);
    scene.add(lamp);
}
for (const x of [-23, -10, 10, 23]) {
    addStreetLamp(x, -boulevardOffset - 1.5, 0);
    addStreetLamp(x,  boulevardOffset + 1.5, Math.PI);
}
for (const z of [-22, -8, 8, 22]) {
    addStreetLamp(-boulevardOffset - 1.5, z, Math.PI / 2);
    addStreetLamp( boulevardOffset + 1.5, z, -Math.PI / 2);
}

// Small landscaped tree groups along the sidewalks.
function addTree(x, z, scale = 1) {
    const trunk = new THREE.Mesh(
        new THREE.CylinderGeometry(0.10 * scale, 0.14 * scale, 1.0 * scale, 7),
        new THREE.MeshBasicMaterial({ color: "#8a755c" })
    );
    trunk.position.set(x, 0.05 * scale, z);
    scene.add(trunk);

    const crown = new THREE.Mesh(
        new THREE.IcosahedronGeometry(0.62 * scale, 1),
        new THREE.MeshBasicMaterial({
            color: "#287f78",
            transparent: true,
            opacity: 0.88
        })
    );
    crown.position.set(x, 0.78 * scale, z);
    scene.add(crown);

    const crownGlow = new THREE.Mesh(
        new THREE.IcosahedronGeometry(0.66 * scale, 1),
        new THREE.MeshBasicMaterial({
            color: "#55d7bf",
            wireframe: true,
            transparent: true,
            opacity: 0.30
        })
    );
    crownGlow.position.copy(crown.position);
    scene.add(crownGlow);
}
for (const x of [-20, -14, -5, 5, 14, 20]) {
    addTree(x, -boulevardOffset - 2.4, 0.85);
    addTree(x,  boulevardOffset + 2.4, 0.85);
}
for (const z of [-20, -12, 0, 12, 20]) {
    addTree(-boulevardOffset - 2.4, z, 0.82);
    addTree( boulevardOffset + 2.4, z, 0.82);
}

// A few small stylized vehicles add scale and a lived-in city feel.
function addCityCar(x, z, color, alongX = true) {
    const carGroup = new THREE.Group();
    const body = new THREE.Mesh(
        new THREE.BoxGeometry(alongX ? 1.65 : 0.9, 0.38, alongX ? 0.9 : 1.65),
        new THREE.MeshBasicMaterial({ color })
    );
    body.position.y = 0.02;
    carGroup.add(body);

    const cabin = new THREE.Mesh(
        new THREE.BoxGeometry(alongX ? 0.82 : 0.62, 0.34, alongX ? 0.68 : 0.82),
        new THREE.MeshBasicMaterial({
            color: "#7fdcf0",
            transparent: true,
            opacity: 0.62
        })
    );
    cabin.position.y = 0.36;
    carGroup.add(cabin);

    for (const wx of [-0.48, 0.48]) {
        for (const wz of [-0.34, 0.34]) {
            const wheel = new THREE.Mesh(
                new THREE.CylinderGeometry(0.14, 0.14, 0.09, 8),
                new THREE.MeshBasicMaterial({ color: "#071018" })
            );
            wheel.rotation.z = Math.PI / 2;
            wheel.position.set(wx, -0.16, wz);
            carGroup.add(wheel);
        }
    }
    carGroup.position.set(x, -0.18, z);
    if (!alongX) carGroup.rotation.y = Math.PI / 2;
    scene.add(carGroup);
}
addCityCar(-17, -boulevardOffset, "#e6a65d", true);
addCityCar(13, boulevardOffset, "#56b8d5", true);
addCityCar(-boulevardOffset, 15, "#6bd4b2", false);
addCityCar(boulevardOffset, -13, "#a9b9d5", false);

// --------------------------------------------------
// BUILDING STRUCTURE
// --------------------------------------------------

for (let i = 0; i < floorCount; i++) {

    const number = i + 1;
    const y = i * floorSpacing + 0.5;
    const data = FLOOR_DATA[i];

    // Floor slab
    const slab = addBox(
        buildingWidth,
        0.11,
        buildingDepth,
        0, y, 0,
        glassMaterial("#2086bd", 0.22)
    );

    addEdges(slab);

    // Clickable floor selection surface
    const clickSurface = new THREE.Mesh(
        new THREE.BoxGeometry(
            buildingWidth + 0.2,
            0.95,
            buildingDepth + 0.2
        ),
        new THREE.MeshBasicMaterial({
            transparent: true,
            opacity: 0.001,
            depthWrite: false
        })
    );

    clickSurface.position.set(0, y + 0.45, 0);
    clickSurface.userData.floorNumber = number;
    scene.add(clickSurface);

    floorObjects.push(clickSurface);

    // Floor highlight outline
    const highlight = new THREE.LineSegments(
        new THREE.EdgesGeometry(
            new THREE.BoxGeometry(
                buildingWidth + 0.3,
                1.0,
                buildingDepth + 0.3
            )
        ),
        new THREE.LineBasicMaterial({
            color: "#8beaff",
            transparent: true,
            opacity: 0
        })
    );

    highlight.position.set(0, y + 0.45, 0);
    scene.add(highlight);
    floorHighlights.push(highlight);

    // Four rooms on each floor
    createRoom(number, -2.2, -1.0, y + 0.25);
    createRoom(number,  2.2, -1.0, y + 0.25);
    createRoom(number, -2.2,  2.0, y + 0.25);
    createRoom(number,  2.2,  2.0, y + 0.25);

    // Side glass facade
    const sideGlass = glassMaterial("#238fcb", 0.07);

    addBox(
        0.06, 0.98, buildingDepth,
        -4.5, y + 0.5, 0,
        sideGlass
    );

    addBox(
        0.06, 0.98, buildingDepth,
        4.5, y + 0.5, 0,
        sideGlass
    );

    // Floor lights
    const strip = addBox(
        buildingWidth + 0.1,
        0.025,
        0.04,
        0, y + 0.02, 3.55,
        new THREE.MeshBasicMaterial({
            color: "#32bfff"
        })
    );

    // Front window divisions
    for (let j = -3; j <= 3; j += 2) {

        addBox(
            0.025, 0.8, 0.03,
            j, y + 0.45, 3.53,
            new THREE.MeshBasicMaterial({
                color: "#50bfff"
            })
        );
    }
}

// --------------------------------------------------
// VERTICAL CENTRAL HEATING CORE
// --------------------------------------------------

const coreMaterial = glassMaterial("#ff812e", 0.10);

addBox(
    0.85,
    totalHeight,
    0.85,
    0,
    totalHeight / 2,
    0,
    coreMaterial
);

const coreEdges = new THREE.LineSegments(
    new THREE.EdgesGeometry(
        new THREE.BoxGeometry(
            0.95, totalHeight, 0.95
        )
    ),
    new THREE.LineBasicMaterial({
        color: "#ff983e",
        transparent: true,
        opacity: 0.8
    })
);

coreEdges.position.y = totalHeight / 2;
scene.add(coreEdges);

// --------------------------------------------------
// SIDE COOLING SHAFTS
// --------------------------------------------------

for (const x of [-3.6, 3.6]) {

    const shaft = addBox(
        0.38,
        totalHeight,
        0.38,
        x,
        totalHeight / 2,
        0,
        glassMaterial("#299eff", 0.12)
    );

    addEdges(shaft, "#47caff");
}

// --------------------------------------------------
// ROOF AND ANTENNA
// --------------------------------------------------

const roof = addBox(
    9.5, 0.16, 7.5,
    0, totalHeight + 0.5, 0,
    glassMaterial("#4abfff", 0.22)
);

addEdges(roof, "#71e5ff");

const antenna = addBox(
    0.06, 2.0, 0.06,
    0, totalHeight + 1.5, 0,
    new THREE.MeshBasicMaterial({color:"#7deaff"})
);

// --------------------------------------------------
// MOVING HEATING AND COOLING PARTICLES
// --------------------------------------------------

function createEnergyStream(x, z, color, count, speed) {

    const material = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.95
    });

    for (let i = 0; i < count; i++) {

        const particle = new THREE.Mesh(
            new THREE.SphereGeometry(
                color === "#ff963d" ? 0.09 : 0.07,
                8, 8
            ),
            material
        );

        const start = Math.random() * totalHeight;

        particle.position.set(
            x + (Math.random() - 0.5) * 0.18,
            start,
            z + (Math.random() - 0.5) * 0.18
        );

        scene.add(particle);

        animatedParticles.push({
            mesh: particle,
            speed: speed * (0.75 + Math.random() * 0.6),
            offset: Math.random() * totalHeight,
            x,
            z,
            color
        });
    }
}

// Warm energy rises through the central core
createEnergyStream(0, 0, "#ff963d", 48, 2.0);
createEnergyStream(0.18, 0.15, "#ffc06a", 25, 1.45);

// Cooling energy flows along side shafts
createEnergyStream(-3.6, 0, "#42c9ff", 35, 1.4);
createEnergyStream(3.6, 0, "#36aaff", 35, 1.6);

// --------------------------------------------------
// FLOATING BLUEPRINT PARTICLES
// --------------------------------------------------

const ambientParticles = [];

for (let i = 0; i < 65; i++) {

    const particle = new THREE.Mesh(
        new THREE.SphereGeometry(0.025, 5, 5),
        new THREE.MeshBasicMaterial({
            color: "#5acbff",
            transparent: true,
            opacity: 0.5
        })
    );

    particle.position.set(
        (Math.random() - 0.5) * 20,
        Math.random() * totalHeight,
        (Math.random() - 0.5) * 16
    );

    scene.add(particle);

    ambientParticles.push(particle);
}

// --------------------------------------------------
// TEMPERATURE COLOUR LOGIC
// --------------------------------------------------

function getTemperatureColor(temp) {

    if (temp < 20) return "#55caff";

    if (temp < 26) return "#51e9ba";

    return "#ffb04a";
}

// --------------------------------------------------
// CAMERA AND FLOOR INSPECTION
// --------------------------------------------------

let cameraAnimation = null;
let selectedFloor = null;

function animateCamera(floorNumber = null) {

    const startPosition = camera.position.clone();
    const startTarget = controls.target.clone();

    let endPosition;
    let endTarget;

    if (floorNumber !== null) {

        const y = (floorNumber - 1) * floorSpacing + 0.5;

        endTarget = new THREE.Vector3(0, y + 0.4, 0);

        endPosition = new THREE.Vector3(
            12,
            y + 3.0,
            13
        );

    } else {

        endPosition = new THREE.Vector3(29, 22, 34);
        endTarget = new THREE.Vector3(0, 9, 0);
    }

    cameraAnimation = {
        start: performance.now(),
        duration: 2700,
        startPosition,
        endPosition,
        startTarget,
        endTarget
    };

    controls.enabled = false;
}

// --------------------------------------------------
// FLOOR CLICK DETECTION
// --------------------------------------------------

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

renderer.domElement.addEventListener("pointerdown", event => {

    if (cameraAnimation) return;

    const rect = renderer.domElement.getBoundingClientRect();

    pointer.x =
        ((event.clientX - rect.left) / rect.width) * 2 - 1;

    pointer.y =
        -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(pointer, camera);

    const hits = raycaster.intersectObjects(floorObjects);

    if (hits.length > 0) {

        const number = hits[0].object.userData.floorNumber;

        selectedFloor = number;

        const data = FLOOR_DATA[number - 1];

        document.getElementById("floorName").textContent =
            "FLOOR " + number + " / ENVIRONMENT";

        const temp = document.getElementById("temperature");

        temp.textContent = data.temperature + "°C";
        temp.style.color = getTemperatureColor(data.temperature);

        document.getElementById("humidity").textContent =
            "Humidity: " + data.humidity + "%";

        document.getElementById("floorStatus").textContent =
            "Status: " + data.status;

        floorHighlights.forEach((highlight, index) => {

            highlight.material.opacity =
                index === number - 1 ? 1 : 0;

            highlight.material.color.set(
                getTemperatureColor(data.temperature)
            );
        });

        animateCamera(number);
    }
});

// --------------------------------------------------
// RESET CAMERA
// --------------------------------------------------

document.getElementById("reset").addEventListener("click", () => {

    selectedFloor = null;

    document.getElementById("floorName").textContent =
        "BUILDING OVERVIEW";

    document.getElementById("temperature").textContent =
        "15 FLOORS";

    document.getElementById("temperature").style.color =
        "#68dcff";

    document.getElementById("humidity").textContent =
        "Select a floor to inspect";

    document.getElementById("floorStatus").textContent =
        "Interactive building simulation";

    floorHighlights.forEach(h => {
        h.material.opacity = 0;
    });

    animateCamera(null);
});

// --------------------------------------------------
// ANIMATION LOOP
// --------------------------------------------------

function animate() {

    requestAnimationFrame(animate);

    const now = performance.now();
    const time = now * 0.001;

    // Camera animation with a gentle drone arc
    if (cameraAnimation) {

        const a = cameraAnimation;

        const p = Math.min(
            (now - a.start) / a.duration,
            1
        );

        const smooth = p * p * (3 - 2 * p);

        camera.position.lerpVectors(
            a.startPosition,
            a.endPosition,
            smooth
        );

        const arc = Math.sin(p * Math.PI);

        camera.position.y += arc * 2.0;

        controls.target.lerpVectors(
            a.startTarget,
            a.endTarget,
            smooth
        );

        camera.lookAt(controls.target);

        if (p >= 1) {
            cameraAnimation = null;
            controls.enabled = true;
        }
    }

    // Animate energy particles
    animatedParticles.forEach(p => {

        p.mesh.position.y += p.speed * 0.016;

        if (p.mesh.position.y > totalHeight) {
            p.mesh.position.y = 0;
        }

        // Small sideways movement
        p.mesh.position.x =
            p.x + Math.sin(time * 1.5 + p.offset) * 0.12;

        p.mesh.position.z =
            p.z + Math.cos(time * 1.3 + p.offset) * 0.12;

        const pulse =
            0.75 + Math.sin(time * 4 + p.offset) * 0.2;

        p.mesh.scale.setScalar(pulse);
    });

    // Ambient holographic particles drift
    ambientParticles.forEach((particle, i) => {

        particle.position.y += 0.004;

        particle.position.x +=
            Math.sin(time + i) * 0.0008;

        if (particle.position.y > totalHeight + 1) {
            particle.position.y = 0;
        }
    });

    // Pulsing building lights
    mainLight.intensity =
        170 + Math.sin(time * 1.5) * 25;

    warmLight.intensity =
        90 + Math.sin(time * 3) * 15;

    coolLight.intensity =
        95 + Math.cos(time * 2.2) * 15;

    // Selected floor glow
    if (selectedFloor !== null) {

        const h = floorHighlights[selectedFloor - 1];

        h.material.opacity =
            0.65 + Math.sin(time * 4) * 0.3;
    }

    controls.update();
    renderer.render(scene, camera);
}

animate();

// --------------------------------------------------
// RESPONSIVE VIEWPORT
// --------------------------------------------------

window.addEventListener("resize", () => {

    camera.aspect =
        window.innerWidth / window.innerHeight;

    camera.updateProjectionMatrix();

    renderer.setSize(
        window.innerWidth,
        window.innerHeight
    );
});

</script>

</body>
</html>
"""

    html = html.replace(
        "__FLOOR_DATA__",
        json.dumps(floor_data)
    )

    components.html(
        html,
        height=620,
        scrolling=False
    )


# SINGLE INTEGRATED DASHBOARD — no separate navigation pages.
st.markdown("""
<div class="ep-hero">
  <div class="ep-kicker">BUILDING INTELLIGENCE / MACHINE LEARNING</div>
  <h1 style="margin:8px 0 4px 0">⚡ Energy Pulse</h1>
  <div class="ep-sub">15-floor building visualization · Energy prediction · Model analytics</div>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Building visualization", "15 floors")
m2.metric("Active ML model", str(model_name))
m3.metric("Prediction inputs", len(features))
m4.metric("Model status", "Loaded")

st.markdown("## Interactive building")
st.caption("Click a floor in the 3D model to inspect it. Drag to rotate and scroll to zoom.")
show_building_viewer()
render_floor_inspector()

st.markdown("---")
st.markdown("## Adjust room conditions and predict")
st.info(
    "The model was trained on a residential dataset. T1, T2 and RH variables are "
    "room temperature/humidity sensors, not floor numbers. The input labels below "
    "therefore use room names while preserving the exact original model feature keys."
)
render_prediction()

st.markdown("---")
st.markdown("## Model analytics")
render_analytics()

with st.expander("Project scope and data notes"):
    st.write(
        "Energy Pulse uses the saved Appliances Energy Prediction model. The underlying "
        "dataset describes a residential building, not a measured 15-floor campus. "
        "The 15-floor building, floor temperatures and humidity values are simulated "
        "demonstration data; they are not live sensor readings. The prediction is for "
        "the supplied observation, not a future-time forecast."
    )
    st.markdown("**Original model input fields**")
    st.dataframe(
        pd.DataFrame({
            "Model feature": features,
            "Display label": [FEATURE_LABELS.get(name, name) for name in features],
        }),
        use_container_width=True,
        hide_index=True,
    )
