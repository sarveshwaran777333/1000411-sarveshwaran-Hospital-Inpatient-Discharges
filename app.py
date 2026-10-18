import re
import html
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import accuracy_score, classification_report

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MediScope | Hospital Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "MediScope"

DATASET_FILENAME = (
    "Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv"
)

SAMPLE_ROWS = 250_000

AGE_ORDER = [
    "0 to 17",
    "18 to 29",
    "30 to 49",
    "50 to 69",
    "70 or Older",
    "Unknown",
]

SEVERITY_ORDER = [
    "Minor",
    "Moderate",
    "Major",
    "Extreme",
    "Unknown",
]

PAYMENT_ORDER = [
    "Medicare",
    "Medicaid",
    "Private Health Insurance",
    "Self-Pay",
    "Other",
    "Unknown",
]

COLORS = [
    "#00E5FF",
    "#8B5CF6",
    "#00E676",
    "#FFB300",
    "#FF4081",
    "#448AFF",
    "#FF7043",
]


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --cyan: #00e5ff;
        --purple: #8b5cf6;
        --green: #00e676;
        --text: #eef7ff;
        --muted: #9aacc4;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(0,229,255,.075), transparent 27%),
            radial-gradient(circle at 90% 5%, rgba(139,92,246,.09), transparent 30%),
            #07111f;
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(7,17,31,.92);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,#0b192b,#081321);
        border-right: 1px solid rgba(0,229,255,.20);
    }

    h1, h2, h3 {
        color: #f4faff !important;
        letter-spacing: -.5px;
    }

    p, label, li {
        color: #c8d7e9;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg,#12243a,#0d1a2b);
        border: 1px solid rgba(0,229,255,.18);
        border-radius: 16px;
        padding: 18px;
        transition: transform .25s ease,border-color .25s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: rgba(0,229,255,.65);
    }

    [data-testid="stMetricLabel"] {
        color: #a7bed8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #00e5ff !important;
        font-weight: 800 !important;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 30px;
        margin-bottom: 22px;
        border-radius: 22px;
        background:
            linear-gradient(115deg,rgba(0,229,255,.11),transparent 48%),
            linear-gradient(145deg,#132740,#0d192a);
        border: 1px solid rgba(0,229,255,.25);
        box-shadow: 0 12px 45px rgba(0,0,0,.18);
    }

    .hero-title {
        font-size: clamp(28px,4vw,43px);
        font-weight: 800;
        color: #f4fbff;
        letter-spacing: -1.5px;
        line-height: 1.15;
    }

    .hero-title span {
        color: #00e5ff;
        text-shadow: 0 0 22px rgba(0,229,255,.35);
    }

    .hero-subtitle {
        color: #b4c9df;
        margin-top: 12px;
        font-size: 14px;
        line-height: 1.7;
        max-width: 750px;
    }

    .hero-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 18px;
    }

    .hero-tag {
        border: 1px solid rgba(0,229,255,.24);
        background: rgba(0,229,255,.06);
        color: #bdefff;
        padding: 6px 11px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 600;
    }

    .hospital-brand {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 13px 8px 18px;
    }

    .hospital-icon {
        position: relative;
        flex: 0 0 58px;
        width: 58px;
        height: 58px;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .hospital-building {
        position: relative;
        z-index: 3;
        width: 44px;
        height: 44px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 13px;
        background: linear-gradient(145deg,#00e5ff,#0088ff);
        color: #07111f;
        font-size: 31px;
        font-weight: 800;
        line-height: 1;
        box-shadow: 0 0 25px rgba(0,229,255,.35);
        animation: hospital-glow 2.2s ease-in-out infinite;
    }

    .hospital-ring {
        position: absolute;
        inset: 0;
        border: 2px solid rgba(0,229,255,.7);
        border-radius: 17px;
        opacity: 0;
        animation: hospital-expand 2.4s ease-out infinite;
    }

    .hospital-ring:nth-child(2) {
        animation-delay: .8s;
    }

    .hospital-ring:nth-child(3) {
        animation-delay: 1.6s;
    }

    @keyframes hospital-expand {
        0% {
            transform: scale(.72);
            opacity: .8;
        }
        100% {
            transform: scale(1.75);
            opacity: 0;
        }
    }

    @keyframes hospital-glow {
        0%,100% {
            box-shadow: 0 0 13px rgba(0,229,255,.25);
        }
        50% {
            box-shadow: 0 0 28px rgba(0,229,255,.65);
        }
    }

    .brand-name {
        font-size: 21px;
        font-weight: 800;
        color: #f4fbff;
    }

    .brand-name span {
        color: #00e5ff;
    }

    .brand-caption {
        font-size: 10px;
        color: #91a9c4;
        margin-top: 3px;
        letter-spacing: 1.3px;
        text-transform: uppercase;
    }

    .mascot-wrap {
        position: relative;
        overflow: hidden;
        margin: 16px 0 18px;
        padding: 18px 12px;
        text-align: center;
        border-radius: 18px;
        background:
            radial-gradient(circle at 50% 35%,rgba(0,229,255,.12),transparent 65%),
            linear-gradient(145deg,#12263c,#0b1727);
        border: 1px solid rgba(0,229,255,.18);
    }

    .robot {
        position: relative;
        width: 94px;
        height: 84px;
        margin: 3px auto 12px;
        animation: robot-float 3.2s ease-in-out infinite;
    }

    .robot-head {
        position: absolute;
        left: 12px;
        top: 8px;
        width: 70px;
        height: 57px;
        border-radius: 20px;
        background: linear-gradient(145deg,#e9fbff,#8bdcf4);
        border: 3px solid #00c8ee;
        box-shadow: 0 0 25px rgba(0,229,255,.18);
    }

    .robot-ear {
        position: absolute;
        top: 26px;
        width: 10px;
        height: 20px;
        border-radius: 5px;
        background: #00c8ee;
    }

    .robot-ear.left {
        left: 3px;
    }

    .robot-ear.right {
        right: 3px;
    }

    .robot-antenna {
        position: absolute;
        top: 0;
        left: 45px;
        height: 12px;
        width: 4px;
        background: #00e5ff;
        border-radius: 4px;
    }

    .robot-antenna:after {
        content: "";
        position: absolute;
        top: -5px;
        left: -4px;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #00e5ff;
        box-shadow: 0 0 12px #00e5ff;
    }

    .robot-eyes {
        display: flex;
        justify-content: center;
        gap: 17px;
        padding-top: 17px;
    }

    .robot-eye {
        width: 9px;
        height: 12px;
        border-radius: 50%;
        background: #12334c;
        animation: robot-blink 4.5s infinite;
    }

    .robot-mouth {
        width: 20px;
        height: 6px;
        margin: 7px auto 0;
        border-bottom: 3px solid #12334c;
        border-radius: 0 0 12px 12px;
    }

    .robot-body {
        position: absolute;
        left: 31px;
        bottom: 0;
        width: 32px;
        height: 14px;
        border-radius: 8px 8px 5px 5px;
        background: #00c8ee;
    }

    .robot-name {
        color: #00e5ff;
        font-size: 14px;
        font-weight: 800;
    }

    .robot-description {
        color: #a7bed8;
        font-size: 10px;
        line-height: 1.6;
        margin-top: 5px;
    }

    @keyframes robot-float {
        0%,100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-7px);
        }
    }

    @keyframes robot-blink {
        0%,46%,50%,100% {
            transform: scaleY(1);
        }
        48% {
            transform: scaleY(.12);
        }
    }

    .section-label {
        color: #00e5ff;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        margin: 5px 0 12px;
    }

    .insight-card {
        height: 100%;
        min-height: 115px;
        padding: 17px;
        border-radius: 15px;
        border: 1px solid rgba(0,229,255,.17);
        background: linear-gradient(145deg,#12243a,#0d1929);
        transition: transform .2s ease,border-color .2s ease;
    }

    .insight-card:hover {
        transform: translateY(-3px);
        border-color: rgba(0,229,255,.5);
    }

    .insight-title {
        color: #9db5d0;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .insight-value {
        color: #00e5ff;
        font-size: 23px;
        font-weight: 800;
        margin-top: 12px;
    }

    .insight-description {
        color: #9fb3ca;
        font-size: 11px;
        margin-top: 5px;
        line-height: 1.5;
    }

    .footer {
        margin-top: 35px;
        padding: 18px 0 8px;
        text-align: center;
        border-top: 1px solid rgba(0,229,255,.14);
        color: #7189a5;
        font-size: 11px;
        line-height: 1.8;
    }

    .stButton > button {
        border-radius: 10px;
        border: 1px solid rgba(0,229,255,.32);
        background: rgba(0,229,255,.08);
        color: #dffaff;
        font-weight: 600;
        transition: all .2s ease;
    }

    .stButton > button:hover {
        border-color: #00e5ff;
        background: rgba(0,229,255,.17);
        color: white;
        transform: translateY(-1px);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid rgba(0,229,255,.16);
        border-radius: 12px;
        overflow: hidden;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BRANDING
# ============================================================

def render_hospital_brand():
    st.markdown(
        """
        <div class="hospital-brand">
            <div class="hospital-icon">
                <div class="hospital-ring"></div>
                <div class="hospital-ring"></div>
                <div class="hospital-ring"></div>
                <div class="hospital-building">+</div>
            </div>
            <div>
                <div class="brand-name">Medi<span>Scope</span></div>
                <div class="brand-caption">Health analytics platform</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mascot():
    st.markdown(
        """
        <div class="mascot-wrap">
            <div class="robot">
                <div class="robot-antenna"></div>
                <div class="robot-ear left"></div>
                <div class="robot-ear right"></div>
                <div class="robot-head">
                    <div class="robot-eyes">
                        <div class="robot-eye"></div>
                        <div class="robot-eye"></div>
                    </div>
                    <div class="robot-mouth"></div>
                </div>
                <div class="robot-body"></div>
            </div>
            <div class="robot-name">MediBot</div>
            <div class="robot-description">
                Your hospital data assistant<br>
                Turning data into useful insights
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DATASET LOADING
# ============================================================

def find_dataset():
    base = Path(__file__).resolve().parent

    candidates = [
        base / DATASET_FILENAME,
        base / "data" / DATASET_FILENAME,
        base / "dataset" / DATASET_FILENAME,
        Path.cwd() / DATASET_FILENAME,
        Path.cwd() / "data" / DATASET_FILENAME,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    for directory in [base, base / "data", base / "dataset"]:
        if directory.is_dir():
            try:
                matches = [
                    path for path in directory.glob("*.csv")
                    if "hospital" in path.name.lower()
                    and "discharge" in path.name.lower()
                ]
                if matches:
                    return matches[0]
            except OSError:
                pass

    return None


@st.cache_data(show_spinner=False, max_entries=2)
def load_dataset(path_string, file_size, modified_time, sample_rows):
    path = Path(path_string)

    try:
        data = pd.read_csv(
            path,
            nrows=sample_rows,
            low_memory=False,
            encoding="utf-8",
            on_bad_lines="skip",
        )
    except UnicodeDecodeError:
        data = pd.read_csv(
            path,
            nrows=sample_rows,
            low_memory=False,
            encoding="latin-1",
            on_bad_lines="skip",
        )

    data.columns = [
        re.sub(r"\s+", " ", str(column)).strip()
        for column in data.columns
    ]

    return data.loc[:, ~pd.Index(data.columns).duplicated()].copy()


# ============================================================
# COLUMN DETECTION
# ============================================================

def normalize_column_name(column):
    return re.sub(r"[^a-z0-9]+", "", str(column).lower())


def find_column(data, possible_names):
    normalized = {
        normalize_column_name(column): column
        for column in data.columns
    }

    for name in possible_names:
        key = normalize_column_name(name)
        if key in normalized:
            return normalized[key]

    for name in possible_names:
        key = normalize_column_name(name)

        if len(key) < 5:
            continue

        for normalized_name, original_name in normalized.items():
            if key in normalized_name:
                return original_name

    return None


# ============================================================
# DATA CLEANING
# ============================================================

def parse_numeric(series):
    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace(r"[^\d.\-]", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def clean_text_series(series):
    return (
        series.astype("string")
        .str.strip()
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "NaN": pd.NA,
                "None": pd.NA,
                "NULL": pd.NA,
                "null": pd.NA,
                "N/A": pd.NA,
            }
        )
        .fillna("Unknown")
        .astype(str)
    )


def standardize_severity(value):
    value = str(value).strip().lower()

    if value in {"1", "minor", "low", "mild"} or "minor" in value:
        return "Minor"

    if value in {"2", "moderate", "medium"} or "moderate" in value:
        return "Moderate"

    if value in {"3", "major", "high", "severe"} or "major" in value:
        return "Major"

    if value in {"4", "extreme", "critical"} or "extreme" in value:
        return "Extreme"

    return "Unknown"


def standardize_payment(value):
    value = str(value).strip().lower()

    if value in {"", "nan", "none", "unknown"}:
        return "Unknown"

    if "medicare" in value:
        return "Medicare"

    if "medicaid" in value:
        return "Medicaid"

    if "private" in value or "commercial" in value:
        return "Private Health Insurance"

    if "self" in value:
        return "Self-Pay"

    if any(
        token in value
        for token in [
            "blue cross",
            "aetna",
            "united healthcare",
            "unitedhealthcare",
            "cigna",
            "healthfirst",
            "empire",
        ]
    ):
        return "Private Health Insurance"

    return "Other"


def standardize_gender(value):
    value = str(value).strip().lower()

    if value in {"m", "male", "man"}:
        return "Male"

    if value in {"f", "female", "woman"}:
        return "Female"

    if value in {"u", "unknown", "other", "nan", "none", ""}:
        return "Unknown"

    return "Other"


def categorize_age(value):
    value = str(value).strip().lower()

    if value in {"", "nan", "none", "unknown", "not available"}:
        return "Unknown"

    if "0 to 17" in value or "0-17" in value:
        return "0 to 17"

    if "18 to 29" in value or "18-29" in value:
        return "18 to 29"

    if "30 to 49" in value or "30-49" in value:
        return "30 to 49"

    if "50 to 69" in value or "50-69" in value:
        return "50 to 69"

    if "70 or older" in value or "70+" in value:
        return "70 or Older"

    match = re.search(r"\d+", value)

    if match:
        age = int(match.group())

        if age <= 17:
            return "0 to 17"
        if age <= 29:
            return "18 to 29"
        if age <= 49:
            return "30 to 49"
        if age <= 69:
            return "50 to 69"
        return "70 or Older"

    return "Unknown"


def categorize_diagnosis(value):
    value = str(value).strip()

    if value.lower() in {"", "nan", "none", "unknown"}:
        return "Unknown"

    value = re.sub(r"\s+", " ", value)

    return value[:62] + "..." if len(value) > 65 else value


def prepare_data(raw_data):
    data = raw_data.copy()

    aliases = {
        "length_of_stay": [
            "Length of Stay",
            "LOS",
            "Length of Stay (Days)",
        ],
        "total_charges": [
            "Total Charges",
            "Charges",
        ],
        "age": [
            "Age Group",
            "Age",
            "Patient Age",
        ],
        "gender": [
            "Gender",
            "Sex",
        ],
        "race": [
            "Race",
            "Patient Race",
        ],
        "severity": [
            "APR Severity of Illness Description",
            "Severity of Illness Description",
            "Severity of Illness",
            "Severity",
        ],
        "diagnosis": [
            "APR DRG Description",
            "APR DRG Code",
            "Diagnosis Description",
            "Diagnosis",
            "Principal Diagnosis Description",
        ],
        "payment": [
            "Payment Typology 1",
            "Payment Type",
            "Payment Typology",
            "Primary Payer",
        ],
        "facility": [
            "Facility Name",
            "Hospital Name",
            "Facility",
        ],
        "county": [
            "Facility County",
            "County",
        ],
        "region": [
            "Health Service Area",
            "Region",
        ],
        "year": [
            "Discharge Year",
            "Year",
        ],
        "date": [
            "Discharge Date",
            "Discharge Date/Time",
        ],
        "record_id": [
            "Record ID",
            "Patient ID",
            "Case ID",
        ],
    }

    source_columns = {
        key: find_column(data, names)
        for key, names in aliases.items()
    }

    original_rows = len(data)

    if source_columns["length_of_stay"]:
        data["LOS_Days"] = parse_numeric(
            data[source_columns["length_of_stay"]]
        )
    else:
        data["LOS_Days"] = np.nan

    if source_columns["total_charges"]:
        data["Total_Charges"] = parse_numeric(
            data[source_columns["total_charges"]]
        )
    else:
        data["Total_Charges"] = np.nan

    if source_columns["age"]:
        data["Age_Group"] = data[
            source_columns["age"]
        ].map(categorize_age)
    else:
        data["Age_Group"] = "Unknown"

    if source_columns["gender"]:
        data["Gender"] = data[
            source_columns["gender"]
        ].map(standardize_gender)
    else:
        data["Gender"] = "Unknown"

    if source_columns["race"]:
        data["Race"] = clean_text_series(
            data[source_columns["race"]]
        )
    else:
        data["Race"] = "Unknown"

    if source_columns["severity"]:
        data["Severity"] = data[
            source_columns["severity"]
        ].map(standardize_severity)
    else:
        data["Severity"] = "Unknown"

    if source_columns["diagnosis"]:
        data["Diagnosis"] = data[
            source_columns["diagnosis"]
        ].map(categorize_diagnosis)
    else:
        data["Diagnosis"] = "Unknown"

    if source_columns["payment"]:
        data["Payment_Type"] = data[
            source_columns["payment"]
        ].map(standardize_payment)
    else:
        data["Payment_Type"] = "Unknown"

    if source_columns["facility"]:
        data["Facility"] = clean_text_series(
            data[source_columns["facility"]]
        )
    else:
        data["Facility"] = "Unknown"

    if source_columns["county"]:
        data["County"] = clean_text_series(
            data[source_columns["county"]]
        )
    else:
        data["County"] = "Unknown"

    if source_columns["region"]:
        data["Region"] = clean_text_series(
            data[source_columns["region"]]
        )
    else:
        data["Region"] = "Unknown"

    if source_columns["year"]:
        data["Discharge_Year"] = parse_numeric(
            data[source_columns["year"]]
        )
    elif source_columns["date"]:
        dates = pd.to_datetime(
            data[source_columns["date"]],
            errors="coerce",
        )
        data["Discharge_Year"] = dates.dt.year
    else:
        data["Discharge_Year"] = np.nan

    if source_columns["record_id"]:
        data["Record_ID"] = clean_text_series(
            data[source_columns["record_id"]]
        )
    else:
        data["Record_ID"] = np.arange(
            1,
            len(data) + 1,
        ).astype(str)

    invalid_los = (
        data["LOS_Days"].isna()
        | (data["LOS_Days"] <= 0)
    )

    invalid_charges = data["Total_Charges"] < 0

    cleaned = data.loc[
        ~invalid_los & ~invalid_charges
    ].copy()

    cleaned["LOS_Days"] = cleaned["LOS_Days"].clip(
        upper=3650
    )

    cleaned["Total_Charges"] = cleaned[
        "Total_Charges"
    ].replace([np.inf, -np.inf], np.nan)

    cleaned["Total_Charges"] = cleaned[
        "Total_Charges"
    ].clip(lower=0)

    cleaned["Age_Group"] = pd.Categorical(
        cleaned["Age_Group"],
        categories=AGE_ORDER,
        ordered=True,
    )

    cleaned["Severity"] = pd.Categorical(
        cleaned["Severity"],
        categories=SEVERITY_ORDER,
        ordered=True,
    )

    cleaned["Payment_Type"] = pd.Categorical(
        cleaned["Payment_Type"],
        categories=PAYMENT_ORDER,
        ordered=True,
    )

    stats = {
        "original_rows": original_rows,
        "cleaned_rows": len(cleaned),
        "removed_rows": original_rows - len(cleaned),
        "invalid_los_rows": int(invalid_los.sum()),
        "invalid_charge_rows": int(invalid_charges.sum()),
        "source_columns": source_columns,
    }

    return cleaned, stats


# ============================================================
# DISPLAY HELPERS
# ============================================================

def format_number(value):
    try:
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return "N/A"


def format_currency(value):
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "N/A"


def format_days(value):
    try:
        return f"{float(value):,.2f} days"
    except (TypeError, ValueError):
        return "N/A"


def style_plot(fig, height=390):
    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, sans-serif",
            color="#dcecff",
            size=11,
        ),
        margin=dict(l=18, r=18, t=45, b=35),
        title_font=dict(size=15, color="#eff9ff"),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        hoverlabel=dict(
            bgcolor="#10253a",
            font_color="#ffffff",
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="rgba(140,180,220,.2)",
    )

    fig.update_yaxes(
        gridcolor="rgba(140,180,220,.10)",
        zeroline=False,
        linecolor="rgba(140,180,220,.2)",
    )

    return fig


def show_chart(fig, key):
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
        },
    )


def insight_card(title, value, description):
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">{html.escape(str(title))}</div>
            <div class="insight-value">{html.escape(str(value))}</div>
            <div class="insight-description">{html.escape(str(description))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def top_values(data, column, limit=15):
    return (
        data[column]
        .astype(str)
        .value_counts()
        .head(limit)
        .index.tolist()
    )


# ============================================================
# LOAD AND PREPROCESS DATA
# ============================================================

dataset_path = find_dataset()

if dataset_path is None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Medi<span>Scope</span></div>
            <div class="hero-subtitle">
                Hospital inpatient discharge analytics dashboard.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.error(
        "The hospital discharge CSV file was not found. "
        "Place the dataset in the same folder as app.py or in a data/ folder."
    )

    st.code(
        "Project folder/\n"
        "├── app.py\n"
        "├── requirements.txt\n"
        "└── Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv",
        language="text",
    )

    st.stop()


file_size = dataset_path.stat().st_size
modified_time = dataset_path.stat().st_mtime

try:
    with st.spinner("Loading hospital discharge data..."):
        raw_df = load_dataset(
            str(dataset_path),
            file_size,
            modified_time,
            SAMPLE_ROWS,
        )
        df, cleaning_stats = prepare_data(raw_df)

except Exception as exc:
    st.error("The dataset could not be loaded or processed.")
    st.code(str(exc))
    st.stop()


if df.empty:
    st.error(
        "No valid records remain after preprocessing. "
        "Check the source dataset and its column names."
    )
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    render_hospital_brand()
    render_mascot()

    st.markdown(
        '<div class="section-label">Navigation</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Dashboard page",
        [
            "Overview",
            "Diagnosis & Stay",
            "Charges & Demographics",
            "Facilities & Counties",
            "Data Quality",
            "AI Experiment",
            "Data Explorer",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown(
        '<div class="section-label">Dataset information</div>',
        unsafe_allow_html=True,
    )

    st.caption(f"File: {dataset_path.name}")
    st.caption(f"File size: {file_size / (1024 ** 2):,.1f} MB")
    st.caption(f"Loaded rows: {len(raw_df):,}")

    if file_size > 250 * 1024 * 1024:
        st.warning(
            "Large dataset detected. A sample is loaded to reduce memory usage."
        )

    st.divider()

    st.markdown(
        '<div class="section-label">Filters</div>',
        unsafe_allow_html=True,
    )

    available_ages = [
        value for value in AGE_ORDER
        if value in df["Age_Group"].astype(str).unique()
    ]

    selected_ages = st.multiselect(
        "Age group",
        available_ages,
        default=available_ages,
    )

    available_genders = sorted(
        df["Gender"].astype(str).dropna().unique().tolist()
    )

    selected_genders = st.multiselect(
        "Gender",
        available_genders,
        default=available_genders,
    )

    available_severities = [
        value for value in SEVERITY_ORDER
        if value in df["Severity"].astype(str).unique()
    ]

    selected_severities = st.multiselect(
        "Severity of illness",
        available_severities,
        default=available_severities,
    )

    available_payments = sorted(
        df["Payment_Type"].astype(str).dropna().unique().tolist()
    )

    selected_payments = st.multiselect(
        "Payment type",
        available_payments,
        default=available_payments,
    )

    st.caption(
        "Selections apply to the dashboard charts and data explorer."
    )


# ============================================================
# FILTER DATA
# ============================================================

filtered_df = df.copy()

filtered_df = filtered_df[
    filtered_df["Age_Group"].astype(str).isin(selected_ages)
    & filtered_df["Gender"].astype(str).isin(selected_genders)
    & filtered_df["Severity"].astype(str).isin(selected_severities)
    & filtered_df["Payment_Type"].astype(str).isin(selected_payments)
]


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            Medi<span>Scope</span> Health Intelligence
        </div>
        <div class="hero-subtitle">
            Explore hospital inpatient discharges, length of stay,
            financial patterns, patient demographics, and facility
            statistics through an interactive analytics dashboard.
        </div>
        <div class="hero-tags">
            <div class="hero-tag">SPARCS Inpatient Data</div>
            <div class="hero-tag">Interactive Analytics</div>
            <div class="hero-tag">Data Quality Monitoring</div>
            <div class="hero-tag">Machine Learning Demonstration</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


if filtered_df.empty:
    st.warning(
        "No records match the selected filters. Change the filters in the sidebar."
    )
    st.stop()


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="section-label">System overview</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Discharge records",
        format_number(len(filtered_df)),
    )

    c2.metric(
        "Average length of stay",
        format_days(filtered_df["LOS_Days"].mean()),
    )

    c3.metric(
        "Average total charges",
        format_currency(filtered_df["Total_Charges"].mean()),
    )

    c4.metric(
        "Facilities represented",
        format_number(filtered_df["Facility"].nunique()),
    )

    c5, c6, c7 = st.columns(3)

    with c5:
        insight_card(
            "Median stay",
            format_days(filtered_df["LOS_Days"].median()),
            "The midpoint of recorded inpatient stays.",
        )

    with c6:
        insight_card(
            "Total recorded charges",
            format_currency(filtered_df["Total_Charges"].sum()),
            "Sum of charges in the current filtered records.",
        )

    with c7:
        insight_card(
            "Counties represented",
            format_number(filtered_df["County"].nunique()),
            "Distinct facility counties in the filtered data.",
        )

    left, right = st.columns(2)

    with left:
        st.subheader("Patient records by age group")

        age_counts = (
            filtered_df["Age_Group"]
            .astype(str)
            .value_counts()
            .reindex(AGE_ORDER, fill_value=0)
            .rename_axis("Age Group")
            .reset_index(name="Records")
        )

        fig = px.bar(
            age_counts,
            x="Age Group",
            y="Records",
            color="Age Group",
            color_discrete_sequence=COLORS,
            text="Records",
            title="Distribution of inpatient discharges",
        )

        fig.update_traces(
            texttemplate="%{text:,.0f}",
            textposition="outside",
        )

        show_chart(
            style_plot(fig),
            "overview_age_distribution",
        )

    with right:
        st.subheader("Primary payment type")

        payment_counts = (
            filtered_df["Payment_Type"]
            .astype(str)
            .value_counts()
            .rename_axis("Payment Type")
            .reset_index(name="Records")
        )

        fig = px.pie(
            payment_counts,
            names="Payment Type",
            values="Records",
            hole=0.60,
            color_discrete_sequence=COLORS,
            title="Discharges by payment category",
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
        )

        show_chart(
            style_plot(fig),
            "overview_payment_pie",
        )

    left, right = st.columns(2)

    with left:
        st.subheader("Length-of-stay distribution")

        los_data = filtered_df["LOS_Days"].dropna()

        if len(los_data) > 20_000:
            los_data = los_data.sample(
                20_000,
                random_state=42,
            )

        fig = px.histogram(
            x=los_data,
            nbins=45,
            color_discrete_sequence=["#00E5FF"],
            title="How long do patients stay?",
            labels={"x": "Length of stay (days)", "count": "Records"},
        )

        show_chart(
            style_plot(fig),
            "overview_los_histogram",
        )

    with right:
        st.subheader("Average stay by severity")

        severity_summary = (
            filtered_df.groupby("Severity", observed=False)["LOS_Days"]
            .mean()
            .reset_index()
        )

        severity_summary["Severity"] = (
            severity_summary["Severity"].astype(str)
        )

        fig = px.bar(
            severity_summary,
            x="Severity",
            y="LOS_Days",
            color="Severity",
            category_orders={"Severity": SEVERITY_ORDER},
            color_discrete_sequence=COLORS,
            title="Average inpatient stay by severity",
            labels={
                "LOS_Days": "Average stay (days)",
                "Severity": "Severity category",
            },
        )

        show_chart(
            style_plot(fig),
            "overview_severity_los",
        )

    st.subheader("Quick findings")

    severity_summary = severity_summary.dropna(
        subset=["LOS_Days"]
    )

    if not severity_summary.empty:
        longest = severity_summary.loc[
            severity_summary["LOS_Days"].idxmax()
        ]

        st.markdown(
            f"- The highest average length of stay among displayed "
            f"severity categories is **{longest['Severity']}**, at "
            f"**{longest['LOS_Days']:.2f} days**."
        )

    age_counts_series = (
        filtered_df["Age_Group"].astype(str).value_counts()
    )

    if not age_counts_series.empty:
        st.markdown(
            f"- The most represented age group is "
            f"**{age_counts_series.index[0]}**, with "
            f"**{age_counts_series.iloc[0]:,} records**."
        )

    payment_series = (
        filtered_df["Payment_Type"].astype(str).value_counts()
    )

    if not payment_series.empty:
        st.markdown(
            f"- The most frequent payment category is "
            f"**{payment_series.index[0]}**."
        )

    st.caption(
        "These findings are descriptive and do not establish causation "
        "or measure hospital quality."
    )


# ============================================================
# DIAGNOSIS AND STAY
# ============================================================

elif page == "Diagnosis & Stay":

    st.title("Diagnosis & Length of Stay")

    st.write(
        "Compare inpatient stay across diagnosis categories and examine "
        "how length of stay varies with illness severity."
    )

    diagnosis_limit = st.slider(
        "Number of diagnosis categories",
        min_value=5,
        max_value=30,
        value=15,
        step=5,
    )

    diagnosis_summary = (
        filtered_df.groupby("Diagnosis", observed=False)
        .agg(
            Records=("Record_ID", "count"),
            Average_Stay=("LOS_Days", "mean"),
            Median_Stay=("LOS_Days", "median"),
            Average_Charges=("Total_Charges", "mean"),
        )
        .reset_index()
    )

    diagnosis_summary = diagnosis_summary[
        diagnosis_summary["Diagnosis"] != "Unknown"
    ]

    diagnosis_summary = diagnosis_summary.sort_values(
        "Records",
        ascending=False,
    ).head(diagnosis_limit)

    st.subheader("Average stay per diagnosis")

    if not diagnosis_summary.empty:
        fig = px.bar(
            diagnosis_summary.sort_values(
                "Average_Stay",
                ascending=True,
            ),
            x="Average_Stay",
            y="Diagnosis",
            orientation="h",
            color="Average_Stay",
            color_continuous_scale=[
                "#14263b",
                "#007cba",
                "#00e5ff",
            ],
            title="Average length of stay by diagnosis category",
            labels={
                "Average_Stay": "Average stay (days)",
                "Diagnosis": "Diagnosis category",
            },
        )

        show_chart(
            style_plot(
                fig,
                height=max(400, diagnosis_limit * 28),
            ),
            "diagnosis_average_stay",
        )

    else:
        st.info("No diagnosis categories are available for these filters.")

    st.subheader("Length of stay by severity")

    box_data = filtered_df.copy()

    if len(box_data) > 15_000:
        box_data = box_data.sample(
            15_000,
            random_state=42,
        )

    fig = px.box(
        box_data,
        x="Severity",
        y="LOS_Days",
        color="Severity",
        category_orders={"Severity": SEVERITY_ORDER},
        color_discrete_sequence=COLORS,
        points="outliers",
        title="Variation in length of stay by severity",
        labels={
            "LOS_Days": "Length of stay (days)",
            "Severity": "Severity category",
        },
    )

    show_chart(
        style_plot(fig),
        "diagnosis_severity_box",
    )

    st.subheader("Diagnosis summary table")

    display_summary = diagnosis_summary.rename(
        columns={
            "Records": "Number of records",
            "Average_Stay": "Average stay (days)",
            "Median_Stay": "Median stay (days)",
            "Average_Charges": "Average charges ($)",
        }
    ).copy()

    for column in [
        "Average stay (days)",
        "Median stay (days)",
        "Average charges ($)",
    ]:
        if column in display_summary.columns:
            display_summary[column] = display_summary[column].round(2)

    st.dataframe(
        display_summary,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download diagnosis summary",
        data=display_summary.to_csv(index=False).encode("utf-8"),
        file_name="diagnosis_stay_summary.csv",
        mime="text/csv",
    )


# ============================================================
# CHARGES AND DEMOGRAPHICS
# ============================================================

elif page == "Charges & Demographics":

    st.title("Charges & Patient Demographics")

    st.write(
        "Explore recorded charges across age groups, gender, "
        "and severity categories."
    )

    charge_data = filtered_df[
        filtered_df["Total_Charges"].notna()
    ].copy()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Average charges",
        format_currency(charge_data["Total_Charges"].mean()),
    )

    c2.metric(
        "Median charges",
        format_currency(charge_data["Total_Charges"].median()),
    )

    c3.metric(
        "Total charges",
        format_currency(charge_data["Total_Charges"].sum()),
    )

    st.subheader("Charges by severity")

    box_charge_data = charge_data.copy()

    if len(box_charge_data) > 15_000:
        box_charge_data = box_charge_data.sample(
            15_000,
            random_state=42,
        )

    fig = px.box(
        box_charge_data,
        x="Severity",
        y="Total_Charges",
        color="Severity",
        category_orders={"Severity": SEVERITY_ORDER},
        color_discrete_sequence=COLORS,
        points="outliers",
        title="Distribution of total charges by severity",
        labels={
            "Total_Charges": "Total charges ($)",
            "Severity": "Severity category",
        },
    )

    fig.update_yaxes(tickprefix="$")

    show_chart(
        style_plot(fig),
        "charges_severity_box",
    )

    st.subheader("Average charges by age group and gender")

    demographic_summary = (
        charge_data.groupby(
            ["Age_Group", "Gender"],
            observed=False,
        )["Total_Charges"]
        .mean()
        .reset_index()
    )

    demographic_summary["Age_Group"] = (
        demographic_summary["Age_Group"].astype(str)
    )

    demographic_summary["Gender"] = (
        demographic_summary["Gender"].astype(str)
    )

    fig = px.bar(
        demographic_summary,
        x="Age_Group",
        y="Total_Charges",
        color="Gender",
        barmode="group",
        category_orders={"Age_Group": AGE_ORDER},
        color_discrete_sequence=COLORS,
        title="Average charges by age group and gender",
        labels={
            "Age_Group": "Age group",
            "Total_Charges": "Average charges ($)",
            "Gender": "Gender",
        },
    )

    fig.update_yaxes(tickprefix="$")

    show_chart(
        style_plot(fig),
        "charges_age_gender",
    )

    st.subheader("Charge distribution by age group")

    if not charge_data.empty:
        violin_data = charge_data.sample(
            min(len(charge_data), 12_000),
            random_state=42,
        )

        fig = px.violin(
            violin_data,
            x="Age_Group",
            y="Total_Charges",
            color="Age_Group",
            category_orders={"Age_Group": AGE_ORDER},
            color_discrete_sequence=COLORS,
            box=True,
            points=False,
            title="Spread of recorded charges across age groups",
            labels={
                "Age_Group": "Age group",
                "Total_Charges": "Total charges ($)",
            },
        )

        fig.update_yaxes(tickprefix="$")

        show_chart(
            style_plot(fig),
            "charges_age_violin",
        )

    demographic_table = (
        charge_data.groupby(
            ["Age_Group", "Gender"],
            observed=False,
        )
        .agg(
            Records=("Record_ID", "count"),
            Average_Charges=("Total_Charges", "mean"),
            Median_Charges=("Total_Charges", "median"),
            Average_Stay=("LOS_Days", "mean"),
        )
        .reset_index()
    )

    demographic_table["Age_Group"] = (
        demographic_table["Age_Group"].astype(str)
    )

    for column in [
        "Average_Charges",
        "Median_Charges",
        "Average_Stay",
    ]:
        demographic_table[column] = demographic_table[column].round(2)

    st.subheader("Demographic summary")

    st.dataframe(
        demographic_table,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FACILITIES AND COUNTIES
# ============================================================

elif page == "Facilities & Counties":

    st.title("Facilities & County Analysis")

    st.write(
        "Compare discharge volume, average stay, and recorded charges "
        "across hospitals and counties."
    )

    facility_summary = (
        filtered_df.groupby("Facility", observed=False)
        .agg(
            Discharges=("Record_ID", "count"),
            Average_Stay=("LOS_Days", "mean"),
            Average_Charges=("Total_Charges", "mean"),
            Total_Charges=("Total_Charges", "sum"),
            County=("County", "first"),
        )
        .reset_index()
    )

    facility_limit = st.slider(
        "Number of facilities to display",
        min_value=5,
        max_value=30,
        value=15,
        step=5,
    )

    top_facilities = facility_summary.sort_values(
        "Discharges",
        ascending=False,
    ).head(facility_limit)

    st.subheader("Facilities with the most discharge records")

    fig = px.bar(
        top_facilities.sort_values("Discharges"),
        x="Discharges",
        y="Facility",
        orientation="h",
        color="Discharges",
        color_continuous_scale=[
            "#14263b",
            "#007cba",
            "#00e5ff",
        ],
        title="Discharge volume by facility",
        labels={
            "Discharges": "Number of records",
            "Facility": "Facility",
        },
    )

    show_chart(
        style_plot(
            fig,
            height=max(420, facility_limit * 28),
        ),
        "facility_discharge_volume",
    )

    st.subheader("Average length of stay by facility")

    fig = px.bar(
        top_facilities.sort_values("Average_Stay"),
        x="Average_Stay",
        y="Facility",
        orientation="h",
        color="Average_Stay",
        color_continuous_scale=[
            "#14263b",
            "#7C4DFF",
            "#00E5FF",
        ],
        title="Average stay among highest-volume facilities",
        labels={
            "Average_Stay": "Average stay (days)",
            "Facility": "Facility",
        },
    )

    show_chart(
        style_plot(
            fig,
            height=max(420, facility_limit * 28),
        ),
        "facility_average_stay",
    )

    st.subheader("County and facility heatmap")

    county_limit = st.slider(
        "Number of counties in heatmap",
        min_value=3,
        max_value=15,
        value=8,
        step=1,
    )

    top_counties = top_values(
        filtered_df,
        "County",
        county_limit,
    )

    heat_facilities = top_values(
        filtered_df,
        "Facility",
        15,
    )

    heat_data = filtered_df[
        filtered_df["County"].astype(str).isin(top_counties)
        & filtered_df["Facility"].astype(str).isin(heat_facilities)
    ]

    if not heat_data.empty:
        heat_matrix = heat_data.pivot_table(
            index="County",
            columns="Facility",
            values="LOS_Days",
            aggfunc="mean",
            observed=False,
        )

        fig = px.imshow(
            heat_matrix,
            aspect="auto",
            color_continuous_scale="Viridis",
            title="Average length of stay by county and facility",
            labels={
                "x": "Facility",
                "y": "County",
                "color": "Average stay (days)",
            },
        )

        fig.update_layout(xaxis_tickangle=-35)

        show_chart(
            style_plot(
                fig,
                height=max(420, len(heat_matrix.index) * 38),
            ),
            "county_facility_heatmap",
        )
    else:
        st.info("There is not enough county and facility data for this heatmap.")

    st.subheader("Facility summary table")

    facility_display = facility_summary.sort_values(
        "Discharges",
        ascending=False,
    ).head(100).copy()

    facility_display = facility_display.rename(
        columns={
            "Discharges": "Discharge records",
            "Average_Stay": "Average stay (days)",
            "Average_Charges": "Average charges ($)",
            "Total_Charges": "Total charges ($)",
        }
    )

    for column in [
        "Average stay (days)",
        "Average charges ($)",
        "Total charges ($)",
    ]:
        facility_display[column] = facility_display[column].round(2)

    st.dataframe(
        facility_display,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download facility summary",
        data=facility_display.to_csv(index=False).encode("utf-8"),
        file_name="facility_summary.csv",
        mime="text/csv",
    )


# ============================================================
# DATA QUALITY
# ============================================================

elif page == "Data Quality":

    st.title("Data Quality & Preprocessing")

    st.write(
        "Review the data-cleaning process, detected source columns, "
        "missing values, and potential outliers."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Original loaded rows",
        format_number(cleaning_stats["original_rows"]),
    )

    c2.metric(
        "Rows after cleaning",
        format_number(cleaning_stats["cleaned_rows"]),
    )

    c3.metric(
        "Rows removed",
        format_number(cleaning_stats["removed_rows"]),
    )

    c4.metric(
        "Removal percentage",
        f"{cleaning_stats['removed_rows'] / max(cleaning_stats['original_rows'], 1) * 100:.2f}%",
    )

    st.subheader("Preprocessing steps")

    steps = [
        (
            "Column identification",
            "Identify relevant source columns for length of stay, charges, "
            "age, gender, race, severity, diagnosis, payment, facility, and county.",
        ),
        (
            "Numeric conversion",
            "Convert length of stay and total charges into numeric values.",
        ),
        (
            "Invalid length of stay",
            "Remove rows with missing, nonnumeric, or nonpositive length of stay.",
        ),
        (
            "Invalid charges",
            "Remove rows with negative total charges and preserve missing charges as missing.",
        ),
        (
            "Category standardization",
            "Standardize age group, gender, severity, and payment categories.",
        ),
        (
            "Missing categorical values",
            "Represent missing categorical values with an Unknown label.",
        ),
        (
            "Analysis fields",
            "Create consistent column names for charts, filtering, and summaries.",
        ),
    ]

    for name, description in steps:
        with st.expander(name):
            st.write(description)

    st.subheader("Detected source columns")

    detected_rows = []

    for logical_name, source_name in cleaning_stats[
        "source_columns"
    ].items():
        detected_rows.append(
            {
                "Analysis field": logical_name.replace("_", " ").title(),
                "Detected source column": source_name or "Not found",
                "Status": "Detected" if source_name else "Unavailable",
            }
        )

    st.dataframe(
        pd.DataFrame(detected_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Missing-value profile")

    missing_counts = df.isna().sum()

    missing_df = pd.DataFrame(
        {
            "Column": missing_counts.index,
            "Missing values": missing_counts.values,
        }
    )

    missing_df["Missing percentage"] = (
        missing_df["Missing values"] / max(len(df), 1) * 100
    ).round(2)

    missing_df = missing_df.sort_values(
        "Missing percentage",
        ascending=False,
    ).head(25)

    fig = px.bar(
        missing_df.sort_values("Missing percentage"),
        x="Missing percentage",
        y="Column",
        orientation="h",
        color="Missing percentage",
        color_continuous_scale=[
            "#14263b",
            "#FFB300",
            "#FF4081",
        ],
        title="Columns with the highest missing-value percentages",
        labels={
            "Missing percentage": "Missing values (%)",
            "Column": "Column name",
        },
    )

    show_chart(
        style_plot(
            fig,
            height=max(420, len(missing_df) * 25),
        ),
        "quality_missing_values",
    )

    st.subheader("Length-of-stay outliers")

    q1 = df["LOS_Days"].quantile(0.25)
    q3 = df["LOS_Days"].quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    lower_fence = max(0, q1 - 1.5 * iqr)

    outlier_count = int(
        (
            (df["LOS_Days"] < lower_fence)
            | (df["LOS_Days"] > upper_fence)
        ).sum()
    )

    left, right = st.columns(2)

    left.metric(
        "Potential LOS outliers",
        format_number(outlier_count),
    )

    right.metric(
        "IQR upper fence",
        format_days(upper_fence),
    )

    st.caption(
        "IQR outliers are statistical flags, not automatically invalid records. "
        "Long stays may be legitimate. LOS values are capped at 3,650 days "
        "to limit extreme values."
    )

    st.subheader("Cleaned dataset preview")

    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download cleaned dataset sample",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="mediscope_cleaned_sample.csv",
        mime="text/csv",
    )


# ============================================================
# AI EXPERIMENT
# ============================================================

elif page == "AI Experiment":

    st.title("AI Experiment: Predicting Severity")

    st.write(
        "Train a demonstration machine-learning model to predict standardized "
        "severity categories using demographic and administrative features. "
        "This experiment is educational and is not intended for clinical use."
    )

    if not SKLEARN_AVAILABLE:
        st.error(
            "scikit-learn is not installed. Add scikit-learn to requirements.txt "
            "and redeploy the application."
        )
        st.stop()

    st.warning(
        "The model uses the loaded dataset sample. Evaluation results may not "
        "represent performance on other hospitals or years."
    )

    ai_data = filtered_df[
        filtered_df["Severity"].astype(str) != "Unknown"
    ].copy()

    ai_data["Target"] = ai_data["Severity"].astype(str)

    features_available = [
        "Age_Group",
        "Gender",
        "Race",
        "Payment_Type",
        "Diagnosis",
        "County",
    ]

    selected_features = st.multiselect(
        "Select model input features",
        features_available,
        default=[
            "Age_Group",
            "Gender",
            "Payment_Type",
            "Diagnosis",
        ],
    )

    model_type = st.selectbox(
        "Choose a model",
        [
            "Random Forest",
            "Dummy baseline",
        ],
    )

    test_percentage = st.slider(
        "Test data percentage",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
    )

    if st.button(
        "Train and evaluate model",
        type="primary",
        use_container_width=True,
    ):
        if not selected_features:
            st.error("Select at least one input feature.")
            st.stop()

        model_data = ai_data[
            selected_features + ["Target"]
        ].copy()

        class_counts = model_data["Target"].value_counts()
        eligible_classes = class_counts[
            class_counts >= 2
        ].index.tolist()

        model_data = model_data[
            model_data["Target"].isin(eligible_classes)
        ]

        if len(model_data) < 100:
            st.error(
                "There are not enough records for this model demonstration."
            )
            st.stop()

        X = model_data[selected_features].copy()
        y = model_data["Target"].astype(str)

        class_counts = y.value_counts()

        if len(class_counts) < 2:
            st.error("At least two severity classes are required.")
            st.stop()

        test_count = max(
            len(class_counts),
            int(np.ceil(len(model_data) * test_percentage / 100)),
        )

        if len(model_data) - test_count < len(class_counts):
            st.error(
                "There are too few training records for the number of classes."
            )
            st.stop()

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_count,
                random_state=42,
                stratify=y,
            )

            preprocessing = ColumnTransformer(
                transformers=[
                    (
                        "categorical",
                        Pipeline(
                            steps=[
                                (
                                    "imputer",
                                    SimpleImputer(
                                        strategy="most_frequent"
                                    ),
                                ),
                                (
                                    "encoder",
                                    OneHotEncoder(
                                        handle_unknown="ignore"
                                    ),
                                ),
                            ]
                        ),
                        selected_features,
                    )
                ],
                remainder="drop",
            )

            if model_type == "Random Forest":
                classifier = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=12,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                )
            else:
                classifier = DummyClassifier(
                    strategy="most_frequent"
                )

            model = Pipeline(
                steps=[
                    ("preprocessing", preprocessing),
                    ("classifier", classifier),
                ]
            )

            with st.spinner("Training the model..."):
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                predictions,
            )

            st.session_state["ai_results"] = {
                "accuracy": accuracy,
                "model_type": model_type,
                "y_test": y_test.tolist(),
                "predictions": predictions.tolist(),
                "features": selected_features,
                "training_rows": len(X_train),
                "testing_rows": len(X_test),
            }

        except Exception as exc:
            st.error("The model could not be trained.")
            st.code(str(exc))

    if "ai_results" in st.session_state:
        results = st.session_state["ai_results"]

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Test accuracy",
            f"{results['accuracy'] * 100:.2f}%",
        )

        c2.metric(
            "Training records",
            format_number(results["training_rows"]),
        )

        c3.metric(
            "Testing records",
            format_number(results["testing_rows"]),
        )

        st.subheader("Actual versus predicted severity")

        comparison = pd.DataFrame(
            {
                "Actual severity": results["y_test"],
                "Predicted severity": results["predictions"],
            }
        )

        confusion = pd.crosstab(
            comparison["Actual severity"],
            comparison["Predicted severity"],
        )

        fig = px.imshow(
            confusion,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Viridis",
            title="Confusion matrix",
            labels={
                "x": "Predicted severity",
                "y": "Actual severity",
                "color": "Records",
            },
        )

        show_chart(
            style_plot(fig),
            "ai_confusion_matrix",
        )

        st.subheader("Classification report")

        report = classification_report(
            results["y_test"],
            results["predictions"],
            output_dict=True,
            zero_division=0,
        )

        st.dataframe(
            pd.DataFrame(report).transpose().round(3),
            use_container_width=True,
        )

        st.write(f"**Model:** {results['model_type']}")
        st.write(
            "**Features:** " + ", ".join(results["features"])
        )

        st.caption(
            "Accuracy alone does not describe all model errors. "
            "Review precision, recall, class imbalance, and the confusion matrix. "
            "This model is not validated for patient care."
        )


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.title("Data Explorer")

    st.write(
        "Search processed records, narrow the data by length of stay and "
        "charges, inspect the results, and download a CSV."
    )

    search_text = st.text_input(
        "Search facility, diagnosis, county, region, or record ID",
        placeholder="Enter a search term...",
    )

    explorer_df = filtered_df.copy()

    if search_text.strip():
        search_mask = pd.Series(
            False,
            index=explorer_df.index,
        )

        for column in [
            "Facility",
            "Diagnosis",
            "County",
            "Record_ID",
            "Region",
        ]:
            search_mask |= explorer_df[column].astype(str).str.contains(
                search_text.strip(),
                case=False,
                na=False,
                regex=False,
            )

        explorer_df = explorer_df[search_mask]

    if explorer_df.empty:
        st.info("No records match the current search.")
    else:
        minimum_los = float(
            max(0, np.floor(explorer_df["LOS_Days"].min()))
        )

        maximum_los = float(
            np.ceil(explorer_df["LOS_Days"].max())
        )

        if maximum_los <= minimum_los:
            maximum_los = minimum_los + 1

        los_range = st.slider(
            "Length-of-stay range (days)",
            min_value=minimum_los,
            max_value=maximum_los,
            value=(minimum_los, maximum_los),
        )

        explorer_df = explorer_df[
            explorer_df["LOS_Days"].between(
                los_range[0],
                los_range[1],
            )
        ]

        if explorer_df.empty:
            st.info("No records fall within the selected stay range.")
        else:
            charge_maximum = float(
                explorer_df["Total_Charges"].max()
            )

            charge_maximum = max(charge_maximum, 1.0)

            charge_limit = st.number_input(
                "Maximum total charges ($)",
                min_value=0.0,
                max_value=charge_maximum,
                value=charge_maximum,
                step=max(charge_maximum / 20, 1.0),
            )

            explorer_df = explorer_df[
                explorer_df["Total_Charges"] <= charge_limit
            ]

            st.metric(
                "Records matching explorer filters",
                format_number(len(explorer_df)),
            )

            st.dataframe(
                explorer_df,
                use_container_width=True,
                hide_index=True,
                height=500,
            )

            st.download_button(
                "Download displayed records as CSV",
                data=explorer_df.to_csv(index=False).encode("utf-8"),
                file_name="mediscope_filtered_records.csv",
                mime="text/csv",
                use_container_width=True,
            )

            numeric_columns = [
                column for column in [
                    "LOS_Days",
                    "Total_Charges",
                    "Discharge_Year",
                ]
                if column in explorer_df.columns
                and pd.api.types.is_numeric_dtype(explorer_df[column])
            ]

            if numeric_columns:
                st.subheader("Numeric summary")

                st.dataframe(
                    explorer_df[numeric_columns].describe().transpose(),
                    use_container_width=True,
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <strong style="color:#00e5ff;">MediScope Health Analytics</strong><br>
        Hospital Inpatient Discharge Intelligence Dashboard<br>
        Built with Streamlit, Pandas, Plotly, and scikit-learn<br>
        <span style="color:#526b85;">
            Analytics are descriptive and educational. They should not be
            interpreted as clinical advice or proof of hospital quality.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
