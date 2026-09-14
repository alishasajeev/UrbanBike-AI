# ============================================================
# 🚲 URBANBIKE AI
# Smart Urban Mobility & Demand Prediction
# Seoul UCI + Washington DC UCI Bike Sharing
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="UrbanBike AI | Smart Mobility",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DARK UI — STREAMLIT-SAFE THEME
# ============================================================

st.markdown(
    """
<style>
:root {
    --ub-bg: #080a0d;
    --ub-bg-2: #0e1217;
    --ub-panel: #151a20;
    --ub-panel-2: #1b2129;
    --ub-panel-3: #222a34;
    --ub-border: #303844;
    --ub-border-soft: #252c35;
    --ub-text: #f7f4ee;
    --ub-text-2: #ded9cf;
    --ub-muted: #a6a19a;
    --ub-cyan: #f0a44b;
    --ub-cyan-soft: #5b3d1d;
    --ub-violet: #d3a85f;
    --ub-violet-soft: #4b3b25;
    --ub-green: #66c79a;
    --ub-amber: #f2c76e;
    --ub-red: #e87575;
}

/* ===== App shell ===== */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background: var(--ub-bg) !important;
    color: var(--ub-text) !important;
}
[data-testid="stHeader"] {
    background: rgba(8,10,13,.94) !important;
    border-bottom: 1px solid var(--ub-border-soft) !important;
}
[data-testid="stToolbar"] { background: transparent !important; }
.block-container {
    max-width: 1460px !important;
    padding: 2rem 2.4rem 4rem !important;
}
.stApp, .stApp * { scrollbar-color: #334155 var(--ub-bg); }

/* ===== Typography ===== */
.stApp p, .stApp li, .stApp span, .stApp label,
.stApp small, [data-testid="stMarkdownContainer"] {
    color: var(--ub-text-2);
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    color: var(--ub-text) !important;
}
[data-testid="stCaptionContainer"], .stCaption, .stCaption * {
    color: var(--ub-muted) !important;
}
.stApp a { color: #f0b15d !important; }
.stApp a:hover { color: #ffd18e !important; }

/* ===== Sidebar ===== */
[data-testid="stSidebar"], [data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] {
    background: #0d1116 !important;
    color: var(--ub-text) !important;
}
[data-testid="stSidebar"] {
    border-right: 1px solid var(--ub-border) !important;
}
[data-testid="stSidebar"] * { color: var(--ub-text-2); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
    color: var(--ub-text) !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: .28rem !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 10px !important;
    padding: .62rem .72rem !important;
    margin: 0 !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: #1b2027 !important;
    border-color: #3a414b !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(90deg, #28241e, #1d2126) !important;
    border-color: #59472d !important;
    box-shadow: inset 3px 0 0 var(--ub-cyan) !important;
}
[data-testid="stSidebar"] hr, hr { border-color: var(--ub-border) !important; }

/* ===== Hero ===== */
.hero {
    padding: 2.35rem 2.7rem;
    border-radius: 22px;
    background:
        radial-gradient(circle at 88% 12%, rgba(56,189,248,.22), transparent 29%),
        radial-gradient(circle at 14% 115%, rgba(167,139,250,.17), transparent 38%),
        linear-gradient(135deg, #11151a 0%, #1b2026 55%, #25221d 100%);
    margin: 0 0 1.8rem;
    border: 1px solid #3d403f;
    box-shadow: 0 20px 50px rgba(0,0,0,.28);
}
.hero * { color: #fff !important; }
.hero h1 {
    font-size: 2.65rem !important;
    font-weight: 850 !important;
    letter-spacing: -.045em;
    margin: 0 0 .45rem !important;
}
.hero p {
    color: #d7d1c7 !important;
    max-width: 820px;
    line-height: 1.6;
    margin: 0 !important;
}

/* ===== Layout rhythm ===== */
[data-testid="stHorizontalBlock"] {
    gap: 1rem !important;
    align-items: stretch !important;
}
[data-testid="stVerticalBlock"] {
    gap: .65rem;
}
[data-testid="stVerticalBlock"] > [data-testid="element-container"] {
    margin-bottom: .1rem;
}
.section-title {
    font-size: 1.35rem !important;
    font-weight: 800 !important;
    color: var(--ub-text) !important;
    margin: 1.8rem 0 .85rem !important;
    letter-spacing: -.015em;
}
.section-subtitle, .table-note { color: var(--ub-muted) !important; }

/* ===== Cards / metrics ===== */
.ui-card {
    height: 100%;
    box-sizing: border-box;
    background: linear-gradient(180deg, #171c22, #12171c) !important;
    border: 1px solid var(--ub-border) !important;
    border-radius: 15px !important;
    padding: 1.15rem 1.2rem !important;
    box-shadow: 0 8px 25px rgba(0,0,0,.16) !important;
}
.ui-card:hover {
    border-color: #4a4d50 !important;
    transform: translateY(-1px);
    transition: .15s ease;
}
.ui-card * { color: var(--ub-text-2) !important; }
[data-testid="stMetric"] {
    min-height: 106px;
    box-sizing: border-box;
    background: linear-gradient(145deg, #181d23, #12171c) !important;
    border: 1px solid var(--ub-border) !important;
    border-radius: 15px !important;
    padding: 1rem 1.05rem !important;
    box-shadow: 0 8px 24px rgba(0,0,0,.14) !important;
}
[data-testid="stMetric"]:hover {
    border-color: #50545a !important;
}
[data-testid="stMetricLabel"] p {
    color: var(--ub-muted) !important;
    font-size: .78rem !important;
    font-weight: 650 !important;
}
[data-testid="stMetricValue"] {
    color: var(--ub-text) !important;
    font-weight: 850 !important;
    letter-spacing: -.025em;
}
[data-testid="stMetricDelta"] { color: var(--ub-text-2) !important; }

/* ===== Widgets ===== */
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
.stRadio label, .stCheckbox label, .stSelectbox label,
.stMultiSelect label, .stSlider label, .stNumberInput label,
.stTextInput label, .stDateInput label, .stTimeInput label {
    color: var(--ub-text-2) !important;
    font-weight: 650 !important;
}
[data-baseweb="input"], [data-baseweb="textarea"],
[data-baseweb="select"] > div,
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input, [data-testid="stTimeInput"] input {
    background: #11161c !important;
    color: var(--ub-text) !important;
    border-color: #3a424c !important;
    -webkit-text-fill-color: var(--ub-text) !important;
    border-radius: 10px !important;
}
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea {
    color: var(--ub-text) !important;
    caret-color: var(--ub-cyan) !important;
}
[data-baseweb="select"] * { color: var(--ub-text) !important; }
[data-baseweb="select"] > div { min-height: 2.65rem !important; }
[data-baseweb="select"] svg { fill: #9fb0c5 !important; }
[data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within,
[data-baseweb="select"] > div:focus-within {
    border-color: var(--ub-cyan) !important;
    box-shadow: 0 0 0 1px var(--ub-cyan) !important;
}
[data-baseweb="popover"], [data-baseweb="popover"] *,
[data-baseweb="menu"], [data-baseweb="menu"] *,
[data-baseweb="select"] [role="listbox"],
[data-baseweb="select"] [role="option"] {
    background: #171c22 !important;
    color: var(--ub-text) !important;
}
[data-baseweb="menu"] [aria-selected="true"],
[data-baseweb="menu"] [role="option"]:hover {
    background: #2a2925 !important;
    color: #fff !important;
}
[data-baseweb="tag"] {
    background: #2c2923 !important;
    border: 1px solid #6a5130 !important;
    border-radius: 7px !important;
}
[data-baseweb="tag"] span, [data-baseweb="tag"] svg {
    color: #e8f8ff !important; fill: #e8f8ff !important;
}

/* ===== Slider ===== */
[data-testid="stSlider"] [role="slider"] {
    background: var(--ub-cyan) !important;
    border-color: var(--ub-cyan) !important;
}
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: var(--ub-muted) !important;
}

/* ===== Buttons ===== */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button,
button[kind="primary"], button[kind="secondary"] {
    min-height: 2.65rem !important;
    border-radius: 10px !important;
    font-weight: 750 !important;
    background: #1b2026 !important;
    color: #f5f1e9 !important;
    border: 1px solid #414850 !important;
}
.stButton > button:hover, .stDownloadButton > button:hover,
.stFormSubmitButton > button:hover, button[kind="secondary"]:hover {
    background: #292d32 !important;
    color: #fff !important;
    border-color: #686e75 !important;
}
button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
    background: linear-gradient(135deg, #c9873b, #8e6a3e) !important;
    color: #fff !important;
    border: 0 !important;
    box-shadow: 0 8px 22px rgba(240,164,75,.18) !important;
}
button[kind="primary"]:hover {
    background: linear-gradient(135deg, #dfa052, #a67d49) !important;
    transform: translateY(-1px);
}
button:disabled { color: #68778a !important; background: #15191e !important; }

/* ===== Tabs / forms / expanders ===== */
.stTabs [data-baseweb="tab-list"] {
    background: #14191f !important;
    border: 1px solid var(--ub-border) !important;
    border-radius: 12px 12px 0 0 !important;
    gap: .2rem !important;
    padding: .28rem .35rem 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--ub-muted) !important;
    font-weight: 700 !important;
    border-radius: 8px 8px 0 0 !important;
    padding: .72rem .9rem !important;
}
.stTabs [data-baseweb="tab"] p, .stTabs [data-baseweb="tab"] span { color: inherit !important; }
.stTabs [data-baseweb="tab"]:hover { color: #fff !important; background: #252a30 !important; }
.stTabs [aria-selected="true"] {
    color: #fff !important;
    background: #2a2a25 !important;
    border-bottom: 3px solid var(--ub-cyan) !important;
}
[data-testid="stForm"] {
    background: #12171c !important;
    border: 1px solid var(--ub-border) !important;
    border-radius: 16px !important;
    padding: 1.35rem !important;
}
[data-testid="stExpander"] {
    background: #171c22 !important;
    border: 1px solid var(--ub-border) !important;
    border-radius: 13px !important;
}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {
    color: var(--ub-text) !important;
}

/* ===== Alerts ===== */
[data-testid="stAlert"] {
    border-radius: 11px !important;
    border: 1px solid var(--ub-border) !important;
}
[data-testid="stAlert"] * { color: var(--ub-text-2) !important; }
[data-testid="stAlert"][kind="success"] { background: #16261f !important; border-color: #3e755e !important; }

/* ===== Plotly / dataframe containers ===== */
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
    background: #0c1116 !important;
    border: 1px solid var(--ub-border) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}
[data-testid="stPlotlyChart"] { padding: .25rem !important; }
[data-testid="stDataFrame"] iframe { background: #0c1116 !important; }

/* ===== Prediction / insight ===== */
.prediction-box {
    padding: 2.35rem;
    border-radius: 22px;
    background:
        radial-gradient(circle at 20% 15%, rgba(240,164,75,.14), transparent 32%),
        linear-gradient(135deg, #14191f, #22211d);
    text-align: center;
    border: 1px solid #4a4235;
    box-shadow: 0 16px 38px rgba(0,0,0,.22);
}
.prediction-box * { color: #fff !important; }
.prediction-number {
    font-size: 4.2rem; font-weight: 900; color: #f2b766 !important;
    line-height: 1; margin: .75rem 0;
}
.prediction-label {
    font-size: .92rem; font-weight: 750; color: #d4cec3 !important;
    text-transform: uppercase; letter-spacing: .06em;
}
.prediction-status { font-size: 1rem; font-weight: 750; color: #fff !important; }
.insight-card {
    background: #171e23 !important;
    border-left: 4px solid #f0a44b !important;
    padding: .9rem 1.05rem;
    border-radius: 10px;
    margin-bottom: .65rem;
    font-size: .9rem;
    color: #f0e9dc !important;
}
.insight-card * { color: inherit !important; }
.footer { color: #718096 !important; text-align: center; font-size: .73rem; padding-top: .5rem; }
[data-testid="stRadio"] label, [data-testid="stCheckbox"] label { color: var(--ub-text-2) !important; }
[data-testid="stRadio"] label p, [data-testid="stCheckbox"] label p { color: inherit !important; }

/* ===== Tooltips ===== */
[role="tooltip"], [data-baseweb="tooltip"] {
    background: #252a30 !important; color: #fff !important;
    border: 1px solid #555c65 !important;
}
[role="tooltip"] *, [data-baseweb="tooltip"] * { color: #fff !important; }

/* ===== Mobile ===== */
@media (max-width: 900px) {
    .block-container { padding: 1.2rem 1rem 3rem !important; }
    .hero { padding: 1.6rem 1.35rem; border-radius: 17px; }
    .hero h1 { font-size: 2rem !important; }
    [data-testid="stHorizontalBlock"] { gap: .7rem !important; }
    .prediction-number { font-size: 3.2rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# PATHS / DATA LOADING
# ============================================================

APP_DIR = Path(__file__).resolve().parent


def first_existing(*names: str) -> Optional[Path]:
    """Return the first dataset path that exists beside the app."""
    for name in names:
        candidate = APP_DIR / name
        if candidate.is_file():
            return candidate
    return None


def read_csv_with_encodings(path: Path) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for encoding in ("utf-8", "utf-8-sig", "cp949", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    raise ValueError(f"Could not read {path.name}: {last_error}")


def require_columns(df: pd.DataFrame, columns: list[str], dataset_name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {', '.join(missing)}"
        )


@st.cache_data(show_spinner=False)
def load_seoul() -> pd.DataFrame:
    path = first_existing(
        "SeoulBikeData.csv",
        "sealbikedata.csv",
        "SeoulBikeData (1).csv",
        "SeoulBikeData (2).csv",
    )
    if path is None:
        raise FileNotFoundError(
            "SeoulBikeData.csv was not found beside the Streamlit app."
        )

    raw = read_csv_with_encodings(path)
    raw.columns = [str(c).strip() for c in raw.columns]

    rename_map: dict[str, str] = {}
    for col in raw.columns:
        clean = col.strip()
        low = clean.lower()
        if "rented bike" in low:
            rename_map[col] = "Rented Bike Count"
        elif clean.lower() == "date":
            rename_map[col] = "Date"
        elif clean.lower() == "hour":
            rename_map[col] = "Hour"
        elif "temperature" in low and "dew" not in low:
            rename_map[col] = "Temperature"
        elif "humidity" in low:
            rename_map[col] = "Humidity"
        elif "wind" in low:
            rename_map[col] = "Wind speed"
        elif "visibility" in low:
            rename_map[col] = "Visibility"
        elif "dew point" in low:
            rename_map[col] = "Dew point temperature"
        elif "solar" in low:
            rename_map[col] = "Solar Radiation"
        elif "rainfall" in low:
            rename_map[col] = "Rainfall"
        elif "snowfall" in low:
            rename_map[col] = "Snowfall"
        elif clean.lower() == "seasons":
            rename_map[col] = "Seasons"
        elif clean.lower() == "holiday":
            rename_map[col] = "Holiday"
        elif "functioning" in low or "functional" in low:
            rename_map[col] = "Functioning Day"

    df = raw.rename(columns=rename_map).copy()
    require_columns(
        df,
        [
            "Rented Bike Count",
            "Date",
            "Hour",
            "Temperature",
            "Humidity",
            "Wind speed",
            "Visibility",
            "Dew point temperature",
            "Solar Radiation",
            "Rainfall",
            "Snowfall",
            "Seasons",
            "Holiday",
            "Functioning Day",
        ],
        "SeoulBikeData.csv",
    )

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    numeric_cols = [
        "Rented Bike Count",
        "Hour",
        "Temperature",
        "Humidity",
        "Wind speed",
        "Visibility",
        "Dew point temperature",
        "Solar Radiation",
        "Rainfall",
        "Snowfall",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Day of Week"] = df["Date"].dt.dayofweek
    df["Day Name"] = df["Date"].dt.day_name()
    df["Day of Year"] = df["Date"].dt.dayofyear
    df["Is Weekend"] = (df["Day of Week"] >= 5).astype(int)
    df["City"] = "Seoul"

    return df.sort_values(["Date", "Hour"], kind="stable").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_washington() -> Optional[pd.DataFrame]:
    path = first_existing("WashingtonBikeData.csv")
    if path is None:
        return None

    raw = read_csv_with_encodings(path)
    require_columns(
        raw,
        ["dteday", "hr", "cnt", "temp", "hum", "windspeed",
         "season", "holiday", "workingday", "weathersit"],
        "WashingtonBikeData.csv",
    )

    w = raw.copy()
    w["Date"] = pd.to_datetime(w["dteday"], errors="coerce")
    w["Hour"] = pd.to_numeric(w["hr"], errors="coerce")
    w["Rented Bike Count"] = pd.to_numeric(w["cnt"], errors="coerce")
    w["Temperature"] = pd.to_numeric(w["temp"], errors="coerce") * 41.0
    w["Humidity"] = pd.to_numeric(w["hum"], errors="coerce") * 100.0
    w["Wind speed"] = pd.to_numeric(w["windspeed"], errors="coerce") * 67.0

    # The Washington UCI hourly dataset does not have Seoul's physical
    # weather fields. Keep a stable common schema, but do not pretend these
    # placeholders are observed measurements.
    w["Visibility"] = 2000.0
    w["Dew point temperature"] = (
        w["Temperature"] - ((100.0 - w["Humidity"]) / 5.0)
    )
    w["Solar Radiation"] = 0.0
    w["Rainfall"] = 0.0
    w["Snowfall"] = 0.0

    w["Seasons"] = (
        pd.to_numeric(w["season"], errors="coerce")
        .map({1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"})
        .fillna("Unknown")
    )
    w["Holiday"] = np.where(
        pd.to_numeric(w["holiday"], errors="coerce").fillna(0).eq(1),
        "Holiday",
        "No Holiday",
    )
    w["Functioning Day"] = np.where(
        pd.to_numeric(w["workingday"], errors="coerce").fillna(0).eq(1),
        "Yes",
        "No",
    )
    w["Weather Condition"] = (
        pd.to_numeric(w["weathersit"], errors="coerce")
        .map({
            1: "Clear",
            2: "Mist / Cloudy",
            3: "Light Rain / Snow",
            4: "Heavy Rain / Snow",
        })
        .fillna("Unknown")
    )

    w["Year"] = w["Date"].dt.year
    w["Month"] = w["Date"].dt.month
    w["Day"] = w["Date"].dt.day
    w["Day of Week"] = w["Date"].dt.dayofweek
    w["Day Name"] = w["Date"].dt.day_name()
    w["Day of Year"] = w["Date"].dt.dayofyear
    w["Is Weekend"] = (w["Day of Week"] >= 5).astype(int)
    w["City"] = "Washington DC"

    return w.sort_values(["Date", "Hour"], kind="stable").reset_index(drop=True)


try:
    seoul = load_seoul()
except Exception as exc:
    st.error(f"**Seoul data error:** {exc}")
    st.stop()

washington = load_washington()


# ============================================================
# MODEL ENGINE
# ============================================================

NUMERIC = [
    "Hour",
    "Temperature",
    "Humidity",
    "Wind speed",
    "Visibility",
    "Dew point temperature",
    "Solar Radiation",
    "Rainfall",
    "Snowfall",
    "Month",
    "Day of Week",
    "Day of Year",
    "Is Weekend",
]

CATEGORICAL = [
    "Seasons",
    "Holiday",
    "Functioning Day",
    "City",
]

FEATURES = NUMERIC + CATEGORICAL
TARGET = "Rented Bike Count"


def make_ohe() -> OneHotEncoder:
    """Support both newer and older scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def clean_training_data(data: pd.DataFrame) -> pd.DataFrame:
    work = data.copy()

    missing = [c for c in FEATURES + [TARGET] if c not in work.columns]
    if missing:
        raise ValueError(f"Training data is missing columns: {', '.join(missing)}")

    for col in NUMERIC + [TARGET]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    for col in NUMERIC:
        median = work[col].median()
        work[col] = work[col].fillna(0.0 if pd.isna(median) else median)

    for col in CATEGORICAL:
        work[col] = work[col].astype("object")
        mode = work[col].mode(dropna=True)
        work[col] = work[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    work = work.dropna(subset=[TARGET]).copy()
    work[TARGET] = work[TARGET].clip(lower=0)
    return work.reset_index(drop=True)


def safe_r2(y_true: pd.Series, predictions: np.ndarray) -> float:
    if len(y_true) < 2 or y_true.nunique(dropna=True) < 2:
        return float("nan")
    return float(r2_score(y_true, predictions))


@st.cache_resource(show_spinner="Training ML engines…")
def train_model_set(data: pd.DataFrame):
    work = clean_training_data(data)

    if len(work) < 10:
        raise ValueError("At least 10 valid observations are required to train the models.")

    X = work[FEATURES]
    y = work[TARGET]

    # Chronological split prevents future observations leaking into training.
    split = int(len(work) * 0.80)
    split = max(1, min(split, len(work) - 1))

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC),
            ("cat", make_ohe(), CATEGORICAL),
        ],
        remainder="drop",
    )

    definitions = {
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=220, random_state=42, n_jobs=-1
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=220, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=160, learning_rate=0.05, max_depth=4, random_state=42
        ),
        "Linear Regression": LinearRegression(),
    }

    results = []
    models = {}

    for name, estimator in definitions.items():
        # Build a fresh preprocessor per pipeline. This avoids accidental
        # state sharing between independently fitted estimators.
        pipeline = Pipeline(
            steps=[
                ("preprocessor", ColumnTransformer(
                    transformers=[
                        ("num", "passthrough", NUMERIC),
                        ("cat", make_ohe(), CATEGORICAL),
                    ],
                    remainder="drop",
                )),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)
        predictions = np.asarray(pipeline.predict(X_test), dtype=float)
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)

        results.append(
            {
                "Model": name,
                "MAE": float(mean_absolute_error(y_test, predictions)),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
                "R² Score": safe_r2(y_test, predictions),
            }
        )
        models[name] = pipeline

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    best_name = str(results_df.iloc[0]["Model"])
    best_model = models[best_name]

    importance_df = pd.DataFrame()
    estimator = best_model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        try:
            pre = best_model.named_steps["preprocessor"]
            encoder = pre.named_transformers_["cat"]
            names = NUMERIC + list(encoder.get_feature_names_out(CATEGORICAL))
            values = np.asarray(estimator.feature_importances_, dtype=float)
            if len(names) == len(values):
                importance_df = (
                    pd.DataFrame({"Feature": names, "Importance": values})
                    .sort_values("Importance", ascending=False)
                    .head(10)
                    .reset_index(drop=True)
                )
        except Exception:
            importance_df = pd.DataFrame()

    return (
        models,
        results_df,
        best_name,
        best_model,
        X_test,
        y_test,
        importance_df,
    )


try:
    (
        seoul_models,
        seoul_results,
        seoul_best_name,
        seoul_best,
        seoul_X_test,
        seoul_y_test,
        seoul_importance,
    ) = train_model_set(seoul)
except Exception as exc:
    st.error(f"**Seoul model error:** {exc}")
    st.stop()

if washington is not None:
    try:
        (
            dc_models,
            dc_results,
            dc_best_name,
            dc_best,
            dc_X_test,
            dc_y_test,
            dc_importance,
        ) = train_model_set(washington)

        combined = pd.concat(
            [
                seoul[FEATURES + [TARGET, "Date"]],
                washington[FEATURES + [TARGET, "Date"]],
            ],
            ignore_index=True,
        ).sort_values(["Date", "Hour"], kind="stable").reset_index(drop=True)

        (
            global_models,
            global_results,
            global_best_name,
            global_best,
            global_X_test,
            global_y_test,
            global_importance,
        ) = train_model_set(combined)
    except Exception as exc:
        st.warning(f"Washington/global model unavailable: {exc}")
        washington = None
        dc_models = dc_results = None
        dc_best_name = dc_best = None
        global_models = global_results = None
        global_best_name = global_best = None
        combined = seoul.copy()
        dc_importance = global_importance = pd.DataFrame()
else:
    dc_models = dc_results = None
    dc_best_name = dc_best = None
    global_models = global_results = None
    global_best_name = global_best = None
    combined = seoul.copy()
    dc_importance = global_importance = pd.DataFrame()


# ============================================================
# HELPERS
# ============================================================

def build_input(
    date,
    hour,
    temperature,
    humidity,
    wind_speed,
    visibility,
    dew_point,
    solar,
    rainfall,
    snowfall,
    season,
    holiday,
    functioning,
    city,
) -> pd.DataFrame:
    date = pd.Timestamp(date)
    return pd.DataFrame(
        {
            "Hour": [float(hour)],
            "Temperature": [float(temperature)],
            "Humidity": [float(humidity)],
            "Wind speed": [float(wind_speed)],
            "Visibility": [float(visibility)],
            "Dew point temperature": [float(dew_point)],
            "Solar Radiation": [float(solar)],
            "Rainfall": [float(rainfall)],
            "Snowfall": [float(snowfall)],
            "Seasons": [str(season)],
            "Holiday": [str(holiday)],
            "Functioning Day": [str(functioning)],
            "City": [str(city)],
            "Month": [date.month],
            "Day of Week": [date.dayofweek],
            "Day of Year": [date.dayofyear],
            "Is Weekend": [int(date.dayofweek >= 5)],
        },
        columns=FEATURES,
    )


def predict_nonnegative(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    values = np.asarray(model.predict(frame), dtype=float)
    return np.clip(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0, None)


CITY_COLORS = {"Seoul": "#00D4C7", "Washington DC": "#FF6B8A"}
SEASON_COLORS = ["#00D4C7", "#72D572", "#FFD166", "#8B7CFF", "#FF6B8A"]
HEATMAP_SCALE = [[0.0, "#11161C"], [0.25, "#173B45"], [0.5, "#176E70"], [0.75, "#00B8AE"], [1.0, "#D8FFF8"]]
CORR_SCALE = [[0.0, "#4F7CFF"], [0.5, "#11161C"], [1.0, "#FF6B8A"]]


def style_chart(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0C1116",
        plot_bgcolor="#0C1116",
        height=height,
        font={"family": "Inter, system-ui, sans-serif", "color": "#E7E3DB", "size": 12},
        title={"font": {"family": "Inter, system-ui, sans-serif", "color": "#FAF7F0", "size": 16}},
        margin={"l": 52, "r": 28, "t": 62, "b": 48},
        legend={
            "bgcolor": "rgba(10,15,23,.78)",
            "bordercolor": "#303844",
            "borderwidth": 1,
            "font": {"color": "#E7E3DB"},
        },
        hoverlabel={"bgcolor": "#20262D", "font": {"color": "#FFFFFF"}},
    )
    fig.update_xaxes(
        gridcolor="#27303A", zerolinecolor="#3A434E",
        color="#B9B5AD", linecolor="#303844", mirror=False,
    )
    fig.update_yaxes(
        gridcolor="#27303A", zerolinecolor="#3A434E",
        color="#B9B5AD", linecolor="#303844", mirror=False,
    )
    return fig



def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def demand_level(value: float) -> tuple[str, str]:
    if value < 500:
        return "LOW", "🟢"
    if value < 1500:
        return "MODERATE", "🟡"
    if value < 2500:
        return "HIGH", "🟠"
    return "CRITICAL", "🔴"


def nonempty_options(series: pd.Series, fallback: str = "Unknown") -> list[str]:
    values = sorted(series.dropna().astype(str).unique().tolist())
    return values or [fallback]


def safe_mean(series: pd.Series) -> float:
    value = float(series.mean()) if len(series) else 0.0
    return 0.0 if not np.isfinite(value) else value


def safe_max(series: pd.Series) -> float:
    value = float(series.max()) if len(series) else 0.0
    return 0.0 if not np.isfinite(value) else value


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🚲 UrbanBike AI")
st.sidebar.markdown("**Smart Urban Mobility Analytics Platform**")
st.sidebar.caption("Seoul + Washington DC")
st.sidebar.divider()

page = st.sidebar.radio(
    "📌 Navigation",
    [
        "🏠 Command Center",
        "📊 Analytics",
        "🌎 City Comparison",
        "🔮 Demand Prediction",
        "🧠 AI Lab",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("### 🏆 Production ML")

if page == "🌎 City Comparison" and washington is not None:
    sb_model_name = global_best_name
    sb_results = global_results
else:
    sb_model_name = seoul_best_name
    sb_results = seoul_results

sb_row = sb_results.loc[sb_results["Model"].eq(sb_model_name)].iloc[0]
sb_r2 = sb_row["R² Score"]
st.sidebar.success(f"Best Engine: **{sb_model_name}**")
st.sidebar.metric("R² Score", "N/A" if pd.isna(sb_r2) else f"{sb_r2:.3f}")
st.sidebar.metric("RMSE", f"{sb_row['RMSE']:.1f}")
st.sidebar.divider()
st.sidebar.caption("UCI Machine Learning Repository")


# ============================================================
# PAGE 1 — COMMAND CENTER
# ============================================================

if page == "🏠 Command Center":
    hero(
        "🚲 UrbanBike AI",
        "Smart urban mobility intelligence across Seoul and Washington DC.",
    )

    total_records = len(seoul) + (len(washington) if washington is not None else 0)
    best_global_r2 = (
        global_results["R² Score"].max()
        if washington is not None
        else seoul_results["R² Score"].max()
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Datasets", "2" if washington is not None else "1")
    c2.metric("Hourly Records", f"{total_records:,}")
    c3.metric("Seoul Peak Demand", f"{safe_max(seoul[TARGET]):,.0f}")
    c4.metric("Best R²", "N/A" if pd.isna(best_global_r2) else f"{best_global_r2:.3f}")

    st.markdown('<div class="section-title">Mobility Overview</div>', unsafe_allow_html=True)

    if washington is not None:
        profile = pd.concat(
            [
                seoul.groupby("Hour")[TARGET].mean().rename("Seoul"),
                washington.groupby("Hour")[TARGET].mean().rename("Washington DC"),
            ],
            axis=1,
        ).reset_index()

        left, right = st.columns([1.35, 1])

        with left:
            fig = px.line(
                profile,
                x="Hour",
                y=["Seoul", "Washington DC"],
                markers=True,
                title="Average Hourly Demand Fingerprint",
                color_discrete_sequence=[CITY_COLORS["Seoul"], CITY_COLORS["Washington DC"]],
            )
            style_chart(fig, 405)
            st.plotly_chart(fig, use_container_width=True)

        with right:
            avg_df = pd.DataFrame(
                {
                    "City": ["Seoul", "Washington DC"],
                    "Average Demand": [
                        safe_mean(seoul[TARGET]),
                        safe_mean(washington[TARGET]),
                    ],
                }
            )
            fig = px.bar(
                avg_df,
                x="City",
                y="Average Demand",
                text_auto=".0f",
                title="Average Hourly Demand",
                color="City",
                color_discrete_sequence=[CITY_COLORS["Seoul"], CITY_COLORS["Washington DC"]],
            )
            style_chart(fig, 405)
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(avg_df.round(1), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Platform</div>', unsafe_allow_html=True)
    platform = st.columns(5)
    cards = [
        ("01", "Analytics", "Explore demand, weather and temporal patterns."),
        ("02", "City Comparison", "Compare Seoul and Washington visually."),
        ("03", "Prediction", "Predict hourly demand with multiple model engines."),
        ("04", "AI Lab", "Run demos, scenarios, anomaly detection and explainability."),
        ("05", "Research Angle", "Study cross-city generalization and mobility behavior."),
    ]
    for col, (number, title, copy) in zip(platform, cards):
        with col:
            st.markdown(
                f"""
                <div class="ui-card" style="min-height:145px;">
                    <div style="color:#f0a44b;font-weight:800;font-size:.68rem;">{number}</div>
                    <h4 style="color:#f5f7fb;margin:.35rem 0 .3rem;">{title}</h4>
                    <div style="color:#a6a19a;font-size:.78rem;line-height:1.5;">{copy}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# PAGE 2 — ANALYTICS
# ============================================================

elif page == "📊 Analytics":
    hero(
        "📊 Seoul Bike Demand Analytics",
        "Interactive demand, weather, temporal and model-performance analysis.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(seoul):,}")
    c2.metric("Model Features", str(len(FEATURES)))
    c3.metric("Average Hourly Demand", f"{safe_mean(seoul[TARGET]):,.0f}")
    c4.metric("Peak Demand", f"{safe_max(seoul[TARGET]):,.0f}")

    st.markdown(
        '<div class="section-title">🎛️ Interactive Subset Filters</div>',
        unsafe_allow_html=True,
    )

    f1, f2, f3 = st.columns(3)
    with f1:
        season_options = nonempty_options(seoul["Seasons"])
        seasons = st.multiselect("Season", season_options, default=season_options, key="analytics_seasons")
    with f2:
        hours = st.slider("Hour Range", 0, 23, (0, 23), key="analytics_hours")
    with f3:
        holiday_options = nonempty_options(seoul["Holiday"])
        holidays = st.multiselect(
            "Holiday Status", holiday_options, default=holiday_options, key="analytics_holidays"
        )

    filtered = seoul[
        seoul["Seasons"].astype(str).isin(seasons)
        & seoul["Hour"].between(hours[0], hours[1])
        & seoul["Holiday"].astype(str).isin(holidays)
    ].copy()

    if filtered.empty:
        st.warning("No rows match the current filters. Widen the filters to continue.")
    else:
        tabs = st.tabs(
            ["📈 Demand Trends", "🌦️ Weather Impact", "⏰ Temporal Patterns", "🤖 Model Evaluation"]
        )

        with tabs[0]:
            daily = filtered.groupby("Date", as_index=False)[TARGET].sum()
            fig = px.line(
                daily, x="Date", y=TARGET, title="Total Daily Bike Demand",
                color_discrete_sequence=[CITY_COLORS["Seoul"]],
            )
            style_chart(fig, 450)
            st.plotly_chart(fig, use_container_width=True)

            hourly = filtered.groupby("Hour", as_index=False)[TARGET].mean()
            fig = px.bar(
                hourly, x="Hour", y=TARGET, title="Average Hourly Demand",
                color=TARGET, color_continuous_scale=HEATMAP_SCALE,
            )
            style_chart(fig, 400)
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            weather = st.selectbox(
                "Select Weather Metric",
                ["Temperature", "Humidity", "Wind speed", "Visibility",
                 "Dew point temperature", "Solar Radiation", "Rainfall", "Snowfall"],
                key="analytics_weather",
            )
            scatter_data = filtered[[weather, TARGET, "Seasons"]].dropna()
            fig = px.scatter(
                scatter_data, x=weather, y=TARGET, color="Seasons", opacity=.55,
                title=f"{weather} vs Bike Demand",
            )
            style_chart(fig, 470)
            st.plotly_chart(fig, use_container_width=True)

            corr_cols = [
                TARGET, "Hour", "Temperature", "Humidity", "Wind speed",
                "Visibility", "Dew point temperature", "Solar Radiation", "Rainfall", "Snowfall",
            ]
            corr = filtered[corr_cols].corr(numeric_only=True)
            fig = px.imshow(
                corr, text_auto=".2f", aspect="auto",
                color_continuous_scale=CORR_SCALE, title="Correlation Matrix",
            )
            style_chart(fig, 560)
            st.plotly_chart(fig, use_container_width=True)

        with tabs[2]:
            left, right = st.columns(2)
            with left:
                season_avg = filtered.groupby("Seasons", as_index=False)[TARGET].mean()
                fig = px.bar(
                    season_avg, x="Seasons", y=TARGET, color=TARGET,
                    color_continuous_scale=SEASON_COLORS, title="Average Demand by Season",
                )
                style_chart(fig, 390)
                st.plotly_chart(fig, use_container_width=True)

            with right:
                holiday_avg = filtered.groupby("Holiday", as_index=False)[TARGET].mean()
                fig = px.bar(
                    holiday_avg, x="Holiday", y=TARGET, color="Holiday",
                    title="Demand on Holidays",
                )
                style_chart(fig, 390)
                st.plotly_chart(fig, use_container_width=True)

            heat = filtered.pivot_table(
                values=TARGET, index="Day Name", columns="Hour", aggfunc="mean"
            ).reindex(
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            )
            fig = px.imshow(
                heat, aspect="auto", color_continuous_scale=[[0.0, "#11161C"], [0.35, "#173B45"], [0.7, "#00B8AE"], [1.0, "#FFD6DE"]],
                title="Weekly Demand Heatmap",
            )
            style_chart(fig, 450)
            st.plotly_chart(fig, use_container_width=True)

        with tabs[3]:
            st.dataframe(
                seoul_results.sort_values("RMSE").round(3),
                use_container_width=True,
                hide_index=True,
            )

            left, right = st.columns(2)
            with left:
                fig = px.bar(
                    seoul_results.sort_values("RMSE"),
                    x="Model", y="RMSE", color="RMSE",
                    color_continuous_scale=HEATMAP_SCALE, title="Model RMSE Comparison",
                )
                style_chart(fig, 400)
                st.plotly_chart(fig, use_container_width=True)

            with right:
                if seoul_importance.empty:
                    st.info("Feature importance is unavailable for the selected best estimator.")
                else:
                    fig = px.bar(
                        seoul_importance.sort_values("Importance"),
                        x="Importance", y="Feature", orientation="h",
                        color="Importance", color_continuous_scale=SEASON_COLORS,
                        title=f"Top Drivers — {seoul_best_name}",
                    )
                    fig.update_layout(yaxis={"autorange": "reversed"})
                    style_chart(fig, 400)
                    st.plotly_chart(fig, use_container_width=True)

            preds = predict_nonnegative(seoul_best, seoul_X_test)
            comparison = pd.DataFrame({"Actual": seoul_y_test.to_numpy(), "Predicted": preds})
            lo = float(comparison.min().min())
            hi = float(comparison.max().max())
            fig = px.scatter(
                comparison, x="Actual", y="Predicted", opacity=.45,
                title=f"Actual vs Predicted — {seoul_best_name}",
            )
            fig.add_trace(
                go.Scatter(
                    x=[lo, hi], y=[lo, hi], mode="lines",
                    name="Perfect Prediction",
                    line={"color": "#FF6B8A", "dash": "dash"},
                )
            )
            style_chart(fig, 430)
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 3 — CITY COMPARISON
# ============================================================

elif page == "🌎 City Comparison":
    hero(
        "🌎 Seoul vs Washington DC",
        "Compare two UCI bike-sharing systems using a shared analytical framework.",
    )

    if washington is None:
        st.warning(
            "WashingtonBikeData.csv is missing or could not be loaded. "
            "Place it beside this app to enable City Comparison."
        )
    else:
        st.markdown(
            '<div class="section-title">🎛️ Comparison Filters</div>',
            unsafe_allow_html=True,
        )
        f1, f2, f3, f4 = st.columns(4)

        with f1:
            selected_seasons = st.multiselect(
                "Season",
                nonempty_options(combined["Seasons"]),
                default=nonempty_options(combined["Seasons"]),
                key="city_seasons",
            )
        with f2:
            selected_hours = st.slider("Hour Range", 0, 23, (0, 23), key="city_hours")
        with f3:
            selected_holidays = st.multiselect(
                "Holiday",
                nonempty_options(combined["Holiday"]),
                default=nonempty_options(combined["Holiday"]),
                key="city_holidays",
            )
        with f4:
            weather_metric = st.selectbox(
                "Weather Lens", ["Temperature", "Humidity", "Wind speed"], key="city_weather"
            )

        cmp = combined[
            combined["Seasons"].astype(str).isin(selected_seasons)
            & combined["Hour"].between(selected_hours[0], selected_hours[1])
            & combined["Holiday"].astype(str).isin(selected_holidays)
        ].copy()

        if cmp.empty:
            st.warning("No rows match the comparison filters.")
        else:
            s = cmp[cmp["City"] == "Seoul"]
            w = cmp[cmp["City"] == "Washington DC"]

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("🇰🇷 Seoul Average / Hour", f"{safe_mean(s[TARGET]):,.0f}")
            k2.metric("🇺🇸 Washington Average / Hour", f"{safe_mean(w[TARGET]):,.0f}")
            k3.metric("🇰🇷 Seoul Peak", f"{safe_max(s[TARGET]):,.0f}")
            k4.metric("🇺🇸 Washington Peak", f"{safe_max(w[TARGET]):,.0f}")

            st.markdown('<div class="section-title">📈 Demand Rhythm</div>', unsafe_allow_html=True)
            hourly_cmp = pd.concat(
                [
                    s.groupby("Hour")[TARGET].mean().rename("Seoul"),
                    w.groupby("Hour")[TARGET].mean().rename("Washington DC"),
                ],
                axis=1,
            ).reset_index()

            fig = px.line(
                hourly_cmp, x="Hour", y=["Seoul", "Washington DC"],
                markers=True, title="Average Hourly Demand",
                color_discrete_sequence=[CITY_COLORS["Seoul"], CITY_COLORS["Washington DC"]],
            )
            fig.update_layout(hovermode="x unified")
            style_chart(fig, 450)
            st.plotly_chart(fig, use_container_width=True)

            left, right = st.columns(2)
            with left:
                normalized = hourly_cmp.copy()
                for col in ("Seoul", "Washington DC"):
                    mean_value = normalized[col].mean()
                    normalized[col] = (
                        normalized[col] / mean_value if mean_value else 0.0
                    )
                fig = px.line(
                    normalized, x="Hour", y=["Seoul", "Washington DC"],
                    markers=True, title="Normalized Demand Shape",
                    color_discrete_sequence=[CITY_COLORS["Seoul"], CITY_COLORS["Washington DC"]],
                )
                fig.add_hline(y=1, line_dash="dot", line_color="#7D8792")
                style_chart(fig, 405)
                st.plotly_chart(fig, use_container_width=True)

            with right:
                heat = cmp.pivot_table(
                    index="City", columns="Hour", values=TARGET, aggfunc="mean"
                )
                fig = px.imshow(
                    heat, aspect="auto", text_auto=".0f",
                    color_continuous_scale=HEATMAP_SCALE, title="City × Hour Demand Heatmap",
                )
                style_chart(fig, 405)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="section-title">🍂 Season & Lifestyle</div>', unsafe_allow_html=True)
            left, right = st.columns(2)

            with left:
                season_cmp = cmp.groupby(["City", "Seasons"], as_index=False)[TARGET].mean()
                fig = px.bar(
                    season_cmp, x="Seasons", y=TARGET, color="City",
                    barmode="group", title="Seasonal Demand",
                    color_discrete_sequence=[CITY_COLORS["Seoul"], CITY_COLORS["Washington DC"]],
                )
                style_chart(fig, 405)
                st.plotly_chart(fig, use_container_width=True)

            with right:
                weekday = cmp.copy()
                weekday["Day Type"] = np.where(weekday["Is Weekend"].eq(1), "Weekend", "Weekday")
                weekday = weekday.groupby(["City", "Day Type"], as_index=False)[TARGET].mean()
                fig = px.bar(
                    weekday, x="Day Type", y=TARGET, color="City",
                    barmode="group", title="Weekday vs Weekend",
                    color_discrete_sequence=[CITY_COLORS["Seoul"], CITY_COLORS["Washington DC"]],
                )
                style_chart(fig, 405)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="section-title">🌦️ Weather Response</div>', unsafe_allow_html=True)
            scatter = cmp[[weather_metric, TARGET, "City"]].dropna()
            # Draw OLS trend lines manually so the app does not depend on
            # statsmodels (Plotly Express requires it for trendline="ols").
            fig = px.scatter(
                scatter, x=weather_metric, y=TARGET, color="City",
                opacity=.38, title=f"{weather_metric} vs Demand",
                color_discrete_sequence=[CITY_COLORS["Seoul"], CITY_COLORS["Washington DC"]],
            )
            for city_name in ("Seoul", "Washington DC"):
                city_data = scatter[scatter["City"].eq(city_name)][[weather_metric, TARGET]].dropna()
                if len(city_data) >= 2 and city_data[weather_metric].nunique() >= 2:
                    x = city_data[weather_metric].to_numpy(dtype=float)
                    y = city_data[TARGET].to_numpy(dtype=float)
                    slope, intercept = np.polyfit(x, y, 1)
                    x_line = np.linspace(x.min(), x.max(), 80)
                    y_line = slope * x_line + intercept
                    fig.add_trace(
                        go.Scatter(
                            x=x_line, y=y_line, mode="lines", name=f"{city_name} trend",
                            line={"color": CITY_COLORS[city_name], "width": 3},
                            showlegend=True,
                        )
                    )
            style_chart(fig, 450)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="section-title">📦 Demand Distribution</div>', unsafe_allow_html=True)
            fig = px.box(
                cmp, x="City", y=TARGET, color="City", points=False,
                title="Demand Distribution by City",
                color_discrete_sequence=[CITY_COLORS["Seoul"], CITY_COLORS["Washington DC"]],
            )
            style_chart(fig, 430)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="section-title">🕸️ Mobility Fingerprint</div>', unsafe_allow_html=True)

            def fingerprint(data: pd.DataFrame) -> np.ndarray:
                if data.empty:
                    return np.zeros(5)
                demand_mean = safe_mean(data[TARGET])
                hourly = data.groupby("Hour")[TARGET].mean()
                weekday = safe_mean(data.loc[data["Is Weekend"].eq(0), TARGET])
                weekend = safe_mean(data.loc[data["Is Weekend"].eq(1), TARGET])
                temp_corr = data[["Temperature", TARGET]].corr().iloc[0, 1]
                humidity_corr = data[["Humidity", TARGET]].corr().iloc[0, 1]
                return np.array(
                    [
                        float(hourly.max() / demand_mean) if demand_mean else 0,
                        float(weekend / weekday) if weekday else 0,
                        abs(float(temp_corr)) if pd.notna(temp_corr) else 0,
                        abs(float(humidity_corr)) if pd.notna(humidity_corr) else 0,
                        float(hourly.std() / demand_mean) if demand_mean else 0,
                    ]
                )

            labels = [
                "Peak Intensity", "Weekend Share", "Temperature Link",
                "Humidity Link", "Hourly Variability",
            ]
            sf, wf = fingerprint(s), fingerprint(w)
            maxima = np.maximum(np.maximum(sf, wf), 1e-9)
            sf, wf = sf / maxima, wf / maxima

            radar = go.Figure()
            radar.add_trace(
                go.Scatterpolar(
                    r=list(sf) + [sf[0]], theta=labels + [labels[0]],
                    fill="toself", name="Seoul", line={"color": CITY_COLORS["Seoul"]},
                )
            )
            radar.add_trace(
                go.Scatterpolar(
                    r=list(wf) + [wf[0]], theta=labels + [labels[0]],
                    fill="toself", name="Washington DC", line={"color": CITY_COLORS["Washington DC"]},
                )
            )
            radar.update_layout(
                template="plotly_dark",
                paper_bgcolor="#111a27",
                plot_bgcolor="#111a27",
                height=480,
                margin={"l": 60, "r": 60, "t": 60, "b": 45},
                polar={
                    "bgcolor": "#111a27",
                    "radialaxis": {"visible": True, "range": [0, 1], "gridcolor": "#253348"},
                    "angularaxis": {"color": "#b8c7da", "gridcolor": "#253348"},
                },
                legend={"font": {"color": "#d7e1ef"}},
            )
            st.plotly_chart(radar, use_container_width=True)


# ============================================================
# PAGE 4 — PREDICTION
# ============================================================

elif page == "🔮 Demand Prediction":
    hero(
        "🔮 Real-Time Demand Predictor",
        "Estimate hourly bike demand with Seoul, Washington DC or the cross-city global engine.",
    )

    city_options = ["Seoul", "Washington DC", "Global Model"] if washington is not None else ["Seoul"]
    city = st.radio("Prediction City", city_options, horizontal=True, key="prediction_city")

    if city == "Seoul":
        model_set, default_model, source, result_set = seoul_models, seoul_best_name, seoul, seoul_results
        model_city = "Seoul"
    elif city == "Washington DC":
        model_set, default_model, source, result_set = dc_models, dc_best_name, washington, dc_results
        model_city = "Washington DC"
    else:
        model_set, default_model, source, result_set = global_models, global_best_name, combined, global_results
        model_city = st.selectbox("Global model target city", ["Seoul", "Washington DC"], key="global_target_city")

    best = result_set.loc[result_set["Model"].eq(default_model)].iloc[0]
    r2_text = "N/A" if pd.isna(best["R² Score"]) else f"{best['R² Score']:.3f}"
    st.success(
        f"🏆 Active ML Engine: **{default_model}** "
        f"(R²: {r2_text} | RMSE: {best['RMSE']:.1f})"
    )

    model_names = list(model_set.keys())
    selected_model = st.selectbox(
        "Select Model Engine",
        model_names,
        index=model_names.index(default_model),
        key="prediction_model",
    )
    pipeline = model_set[selected_model]

    default_season = nonempty_options(source["Seasons"])[0]
    default_holiday = nonempty_options(source["Holiday"])[0]
    default_functioning = nonempty_options(source["Functioning Day"])[0]

    with st.form("prediction_form"):
        st.markdown("### 🕐 Time & Calendar")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            pred_date = st.date_input("Date", value=pd.Timestamp("2018-06-15").date())
        with c2:
            pred_hour = st.slider("Hour of Day", 0, 23, 18)
        with c3:
            holiday = st.selectbox(
                "Holiday", nonempty_options(source["Holiday"]),
                index=nonempty_options(source["Holiday"]).index(default_holiday),
            )
        with c4:
            functioning = st.selectbox(
                "Functioning Day", nonempty_options(source["Functioning Day"]),
                index=nonempty_options(source["Functioning Day"]).index(default_functioning),
            )

        st.markdown("### 🌦️ Weather Parameters")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            temperature = st.number_input("Temperature (°C)", -30.0, 45.0, 22.0, .5)
        with c2:
            humidity = st.number_input("Humidity (%)", 0, 100, 45, 1)
        with c3:
            wind = st.number_input("Wind Speed (m/s)", 0.0, 20.0, 1.8, .1)
        with c4:
            visibility = st.number_input("Visibility (10m)", 0, 5000, 1800, 50)

        st.markdown("### 🌧️ Environmental Parameters")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            dew = st.number_input("Dew Point (°C)", -30.0, 40.0, 9.5, .5)
        with c2:
            solar = st.number_input("Solar Radiation", 0.0, 5.0, 1.5, .1)
        with c3:
            rainfall = st.number_input("Rainfall (mm)", 0.0, 100.0, 0.0, .1)
        with c4:
            snowfall = st.number_input("Snowfall (cm)", 0.0, 30.0, 0.0, .1)

        season_options = nonempty_options(source["Seasons"])
        season = st.selectbox(
            "Season", season_options, index=season_options.index(default_season)
        )

        submitted = st.form_submit_button(
            "🚀 PREDICT HOURLY BIKE DEMAND", use_container_width=True, type="primary"
        )

    # Form values are always valid, but only execute the explicit action on submit.
    if submitted:
        input_data = build_input(
            pred_date, pred_hour, temperature, humidity, wind, visibility,
            dew, solar, rainfall, snowfall, season, holiday, functioning, model_city,
        )
        prediction = float(predict_nonnegative(pipeline, input_data)[0])
        level, icon = demand_level(prediction)

        st.markdown(
            f"""
            <div class="prediction-box">
                <div class="prediction-label">{model_city} • {selected_model}</div>
                <div class="prediction-number">{prediction:,.0f}</div>
                <div class="prediction-status">{icon} {level} DEMAND</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        left, right = st.columns(2)

        with left:
            gauge_max = max(3000.0, safe_max(source[TARGET]))
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prediction,
                    title={"text": "Expected Bikes / Hour"},
                    gauge={
                        "axis": {"range": [0, gauge_max]},
                        "steps": [
                            {"range": [0, gauge_max * .25], "color": "#123a2b"},
                            {"range": [gauge_max * .25, gauge_max * .60], "color": "#3d3513"},
                            {"range": [gauge_max * .60, gauge_max], "color": "#42202a"},
                        ],
                        "bar": {"color": "#38bdf8"},
                    },
                )
            )
            gauge.update_layout(
                height=350, template="plotly_dark", paper_bgcolor="#111a27",
                font={"family": "Inter, sans-serif", "color": "#d7e1ef"},
                margin={"l": 35, "r": 35, "t": 55, "b": 20},
            )
            st.plotly_chart(gauge, use_container_width=True)

        with right:
            st.markdown("### 🧠 AI Smart Insights")
            insights = []
            if pred_hour in [8, 18]:
                insights.append("⏰ **Commute Peak:** Selected hour is a major mobility window.")
            if temperature >= 20:
                insights.append("🌡️ **Favorable Weather:** Temperature is supportive of outdoor mobility.")
            if rainfall > 0:
                insights.append("🌧️ **Rainfall Alert:** Wet conditions can suppress bike usage.")
            if snowfall > 0:
                insights.append("❄️ **Snow Alert:** Snow can sharply reduce cycling activity.")
            if functioning == "No":
                insights.append("🚫 **Non-Working Day:** Demand behavior can differ from working days.")

            p90 = source[TARGET].quantile(.90)
            if pd.notna(p90) and prediction >= p90:
                insights.append("🔥 **High-Demand Alert:** Prediction is above the historical 90th percentile.")
            if not insights:
                insights.append("✅ Regular operating conditions.")

            for insight in insights:
                st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)

        st.divider()
        st.subheader("📈 24-Hour Demand Outlook")

        outlook = []
        for hour in range(24):
            hour_input = input_data.copy()
            hour_input["Hour"] = hour
            hour_prediction = float(predict_nonnegative(pipeline, hour_input)[0])
            outlook.append({"Hour": hour, "Predicted Demand": hour_prediction})

        fig = px.area(
            pd.DataFrame(outlook),
            x="Hour", y="Predicted Demand",
            title=f"{model_city} — Full Day Demand Profile",
        )
        style_chart(fig, 430)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Set the inputs above and press **Predict Hourly Bike Demand** to run the model.")


# ============================================================
# PAGE 5 — AI LAB
# ============================================================

elif page == "🧠 AI Lab":
    hero(
        "🧠 AI Lab",
        "Scenario simulation, anomaly detection and model explainability in one workspace.",
    )

    ai_tabs = st.tabs(["⚡ Quick Demo", "🧪 What-If", "🚨 Anomalies", "🤖 Explainability"])

    scenarios = [
        {
            "name": "☀️ Ideal Spring Commute",
            "hour": 18, "temp": 22.0, "rain": 0.0, "season": "Spring",
            "description": "Pleasant temperature during the evening commute.",
        },
        {
            "name": "🌧️ Heavy Rain",
            "hour": 18, "temp": 18.0, "rain": 12.0, "season": "Summer",
            "description": "Heavy rain during the evening rush.",
        },
        {
            "name": "❄️ Winter Night",
            "hour": 23, "temp": -8.0, "rain": 0.0, "season": "Winter",
            "description": "Sub-zero late-night conditions.",
        },
        {
            "name": "🌞 Warm Afternoon",
            "hour": 14, "temp": 30.0, "rain": 0.0, "season": "Summer",
            "description": "Warm, dry daytime conditions.",
        },
    ]

    with ai_tabs[0]:
        demo_city = st.radio(
            "Demo City",
            ["Seoul", "Washington DC"] if washington is not None else ["Seoul"],
            horizontal=True,
            key="demo_city",
        )
        demo_model = seoul_best if demo_city == "Seoul" else dc_best
        demo_source = seoul if demo_city == "Seoul" else washington

        demo_cards = st.columns(4)
        demo_seasons = set(nonempty_options(demo_source["Seasons"]))
        for col, sc in zip(demo_cards, scenarios):
            with col:
                demo_season = sc["season"] if sc["season"] in demo_seasons else nonempty_options(demo_source["Seasons"])[0]
                data = build_input(
                    "2018-06-15", sc["hour"], sc["temp"], 50, 2.0, 1500,
                    10.0, 1.5, sc["rain"], 0.0, demo_season,
                    "No Holiday", "Yes", demo_city,
                )
                pred = float(predict_nonnegative(demo_model, data)[0])
                st.markdown(
                    f"""
                    <div class="ui-card" style="min-height:180px;">
                        <h4 style="color:#f5f7fb;margin:.1rem 0 .35rem;">{sc["name"]}</h4>
                        <div style="color:#a6a19a;font-size:.76rem;">{sc["description"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.metric("Predicted Demand", f"{pred:,.0f}")

        if washington is not None:
            st.subheader("🌎 Scenario Battle")
            battle = []
            for sc in scenarios:
                seoul_season = sc["season"] if sc["season"] in set(nonempty_options(seoul["Seasons"])) else nonempty_options(seoul["Seasons"])[0]
                dc_season = sc["season"] if sc["season"] in set(nonempty_options(washington["Seasons"])) else nonempty_options(washington["Seasons"])[0]
                s_input = build_input(
                    "2018-06-15", sc["hour"], sc["temp"], 50, 2.0, 1500,
                    10.0, 1.5, sc["rain"], 0.0, seoul_season,
                    "No Holiday", "Yes", "Seoul",
                )
                w_input = build_input(
                    "2018-06-15", sc["hour"], sc["temp"], 50, 2.0, 1500,
                    10.0, 1.5, sc["rain"], 0.0, dc_season,
                    "No Holiday", "Yes", "Washington DC",
                )
                battle.append(
                    {
                        "Scenario": sc["name"],
                        "Seoul": float(predict_nonnegative(seoul_best, s_input)[0]),
                        "Washington DC": float(predict_nonnegative(dc_best, w_input)[0]),
                    }
                )
            battle_df = pd.DataFrame(battle)
            fig = px.bar(
                battle_df, x="Scenario", y=["Seoul", "Washington DC"],
                barmode="group", title="Same scenario inputs, two city models",
                color_discrete_sequence=[CITY_COLORS["Seoul"], CITY_COLORS["Washington DC"]],
            )
            style_chart(fig, 430)
            st.plotly_chart(fig, use_container_width=True)

    with ai_tabs[1]:
        st.subheader("🧪 What-If Mobility Simulator")
        scenario_city = st.radio(
            "Model",
            ["Seoul", "Washington DC", "Global Model"] if washington is not None else ["Seoul"],
            horizontal=True, key="whatif_city",
        )

        if scenario_city == "Seoul":
            scenario_model, scenario_source, scenario_model_city = seoul_best, seoul, "Seoul"
        elif scenario_city == "Washington DC":
            scenario_model, scenario_source, scenario_model_city = dc_best, washington, "Washington DC"
        else:
            scenario_model, scenario_source = global_best, combined
            scenario_model_city = st.selectbox(
                "Global model target city", ["Seoul", "Washington DC"], key="whatif_global_city"
            )

        a, b, c, d = st.columns(4)
        with a:
            scenario_hour = st.slider("Hour", 0, 23, 18, key="whatif_hour")
        with b:
            scenario_temp = st.slider("Temperature °C", -20.0, 40.0, 22.0, .5, key="whatif_temp")
        with c:
            scenario_humidity = st.slider("Humidity %", 0, 100, 50, key="whatif_humidity")
        with d:
            scenario_rain = st.slider("Rainfall mm", 0.0, 30.0, 0.0, .5, key="whatif_rain")

        scenario_seasons = nonempty_options(scenario_source["Seasons"])
        scenario_season = st.selectbox("Season", scenario_seasons, key="whatif_season")

        base = build_input(
            "2018-06-15", scenario_hour, scenario_temp, scenario_humidity,
            2.0, 1500, 10.0, 1.5, scenario_rain, 0.0,
            scenario_season, "No Holiday", "Yes", scenario_model_city,
        )
        base_prediction = float(predict_nonnegative(scenario_model, base)[0])

        experiment = []
        variants = [
            ("Baseline", scenario_hour, scenario_temp, scenario_rain),
            ("Heavy Rain", scenario_hour, scenario_temp, max(12.0, scenario_rain)),
            ("Cold Weather", scenario_hour, min(5.0, scenario_temp), scenario_rain),
            ("Morning Rush", 8, scenario_temp, scenario_rain),
            ("Evening Rush", 18, scenario_temp, scenario_rain),
        ]
        for label, hour_value, temp_value, rain_value in variants:
            test_input = build_input(
                "2018-06-15", hour_value, temp_value, scenario_humidity,
                2.0, 1500, 10.0, 1.5, rain_value, 0.0,
                scenario_season, "No Holiday", "Yes", scenario_model_city,
            )
            prediction_value = float(predict_nonnegative(scenario_model, test_input)[0])
            change = ((prediction_value / base_prediction) - 1) * 100 if base_prediction else 0.0
            experiment.append(
                {"Scenario": label, "Predicted Demand": prediction_value, "Change vs Baseline": change}
            )

        experiment_df = pd.DataFrame(experiment)
        st.metric("Baseline Demand", f"{base_prediction:,.0f} bikes/hour")
        fig = px.bar(
            experiment_df, x="Scenario", y="Predicted Demand",
            color="Predicted Demand", color_continuous_scale=HEATMAP_SCALE,
            title="Scenario Sensitivity",
        )
        style_chart(fig, 420)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            experiment_df.style.format(
                {"Predicted Demand": "{:,.0f}", "Change vs Baseline": "{:+.1f}%"}
            ),
            use_container_width=True, hide_index=True,
        )

    with ai_tabs[2]:
        st.subheader("🚨 Demand Anomaly Detection")
        anomaly_city = st.selectbox(
            "Analyze City",
            ["Seoul", "Washington DC"] if washington is not None else ["Seoul"],
            key="anomaly_city",
        )
        anomaly_source = seoul if anomaly_city == "Seoul" else washington

        anomaly_data = (
            anomaly_source.copy()
            .sort_values(["Date", "Hour"])
            .reset_index(drop=True)
        )

        if len(anomaly_data) < 20:
            st.warning("At least 20 observations are recommended for anomaly detection.")
        else:
            anomaly_data["Rolling Mean"] = anomaly_data[TARGET].rolling(24, min_periods=6).mean()
            anomaly_data["Residual"] = anomaly_data[TARGET] - anomaly_data["Rolling Mean"]

            contamination = min(.025, max(1 / len(anomaly_data), .001))
            detector = IsolationForest(
                contamination=contamination, random_state=42, n_estimators=150
            )
            detector_input = anomaly_data[[TARGET, "Hour", "Temperature", "Humidity"]].fillna(0)
            anomaly_data["Flag"] = detector.fit_predict(detector_input)

            unusual = anomaly_data[anomaly_data["Flag"] == -1]
            st.metric("Detected Unusual Observations", f"{len(unusual):,}")

            plot_data = anomaly_data.copy()
            plot_data["Status"] = plot_data["Flag"].map({1: "Normal", -1: "Anomaly"})
            fig = px.scatter(
                plot_data, x="Date", y=TARGET, color="Status",
                title=f"{anomaly_city} Demand Anomalies",
                color_discrete_map={"Normal": "#00D4C7", "Anomaly": "#FF6B8A"},
            )
            style_chart(fig, 460)
            st.plotly_chart(fig, use_container_width=True)

            largest = (
                anomaly_data.assign(AbsoluteResidual=anomaly_data["Residual"].abs())
                .sort_values("AbsoluteResidual", ascending=False)
                [[
                    "Date", "Hour", TARGET, "Rolling Mean", "Residual",
                    "Temperature", "Humidity",
                ]]
                .head(15)
            )
            st.dataframe(largest.round(2), use_container_width=True, hide_index=True)

    with ai_tabs[3]:
        st.subheader("🤖 Model Explainability")
        explain_city = st.selectbox(
            "Explain Model For",
            ["Seoul", "Washington DC", "Global Model"] if washington is not None else ["Seoul"],
            key="explain_city",
        )

        if explain_city == "Seoul":
            importance, explain_name = seoul_importance, seoul_best_name
        elif explain_city == "Washington DC":
            importance, explain_name = dc_importance, dc_best_name
        else:
            importance, explain_name = global_importance, global_best_name

        st.success(f"Active explainability engine: **{explain_name}**")

        if importance.empty:
            st.info("Feature importance is not available for this estimator.")
        else:
            fig = px.bar(
                importance.sort_values("Importance"),
                x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale=SEASON_COLORS,
                title="Top Model Drivers",
            )
            style_chart(fig, 500)
            st.plotly_chart(fig, use_container_width=True)

            for _, row in importance.head(5).iterrows():
                st.markdown(
                    f"""
                    <div class="insight-card">
                        <b>{row["Feature"]}</b><br>
                        Learned importance: {row["Importance"]:.3f}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.caption(
            "Feature importance describes model reliance; it does not establish causality."
        )


# ============================================================
# EXPORT
# ============================================================

st.divider()
with st.expander("📥 Export datasets", expanded=False):
    e1, e2 = st.columns(2)
    with e1:
        st.download_button(
            "Download cleaned Seoul dataset",
            data=seoul.to_csv(index=False).encode("utf-8"),
            file_name="SeoulBikeData_Cleaned.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with e2:
        if washington is not None:
            st.download_button(
                "Download standardized multi-city dataset",
                data=combined.to_csv(index=False).encode("utf-8"),
                file_name="UrbanBikeAI_Combined.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("Add WashingtonBikeData.csv to enable multi-city export.")

st.markdown(
    '<div class="footer">🚲 UrbanBike AI | Seoul + Washington DC | Streamlit + Scikit-Learn + Plotly | Dark UI</div>',
    unsafe_allow_html=True,
)
