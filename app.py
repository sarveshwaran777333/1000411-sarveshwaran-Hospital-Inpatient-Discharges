import os
import glob
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(
    page_title="MediScope | Hospital Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

TARGET_FILE = "Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #07111f;
    --panel: #0d1b2b;
    --panel2: #102438;
    --cyan: #00e5ff;
    --teal: #00f5c3;
    --text: #e8f7ff;
    --muted: #91a9bb;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(0,229,255,0.09), transparent 32%),
        radial-gradient(ellipse at 90% 10%, rgba(0,245,195,0.06), transparent 30%),
        var(--bg);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(7,17,31,0.75);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1928 0%, #08121f 100%);
    border-right: 1px solid rgba(0,229,255,0.18);
}

[data-testid="stSidebar"] * {
    color: #dff7ff;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

@keyframes hospital-pulse {
    0% { transform: scale(0.78); opacity: 0.8; }
    70% { transform: scale(1.35); opacity: 0; }
    100% { transform: scale(1.35); opacity: 0; }
}

@keyframes cross-glow {
    0%, 100% {
        filter: drop-shadow(0 0 4px #00e5ff);
        transform: scale(1);
    }
    50% {
        filter: drop-shadow(0 0 16px #00e5ff);
        transform: scale(1.08);
    }
}

@keyframes mascot-float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-7px); }
}

@keyframes eye-glow {
    0%, 100% { box-shadow: 0 0 4px #00e5ff; }
    50% { box-shadow: 0 0 13px #00e5ff; }
}

@keyframes heartbeat {
    0%, 100% { transform: scaleX(0.85); opacity: 0.65; }
    50% { transform: scaleX(1.08); opacity: 1; }
}

.hospital-brand {
    display: flex;
    align-items: center;
    gap: 16px;
    margin: 4px 0 12px;
}

.hospital-icon {
    width: 72px;
    height: 72px;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.pulse-ring {
    position: absolute;
    inset: 6px;
    border: 2px solid #00e5ff;
    border-radius: 20px;
    animation: hospital-pulse 2.2s ease-out infinite;
}

.pulse-ring.delay {
    animation-delay: 1.1s;
}

.hospital-cross {
    z-index: 2;
    position: relative;
    color: #00e5ff;
    font-size: 52px;
    font-weight: 800;
    line-height: 1;
    animation: cross-glow 2s ease-in-out infinite;
}

.hospital-brand h1 {
    color: #e8fbff;
    font-size: 30px;
    font-weight: 800;
    margin: 0;
    padding: 0;
    letter-spacing: -0.8px;
}

.hospital-brand p {
    color: #86aebc;
    font-size: 11px;
    letter-spacing: 2px;
    margin: 5px 0 0;
}

.hero-subtitle {
    color: #9cb8c9;
    font-size: 14px;
    margin: 0 0 18px;
}

.sidebar-mascot {
    margin: 10px 0 22px;
    padding: 18px 10px 14px;
    border: 1px solid rgba(0,229,255,0.3);
    border-radius: 18px;
    background: linear-gradient(145deg, #10283a, #091522);
    text-align: center;
    overflow: hidden;
    box-shadow: 0 8px 28px rgba(0,0,0,0.15);
}

.robot {
    position: relative;
    width: 92px;
    height: 86px;
    margin: 0 auto 12px;
    animation: mascot-float 3s ease-in-out infinite;
}

.robot-antenna {
    position: absolute;
    top: 0;
    left: 44px;
    width: 4px;
    height: 14px;
    background: #00e5ff;
    border-radius: 4px;
}

.robot-antenna::before {
    content: "";
    position: absolute;
    top: -5px;
    left: -4px;
    width: 12px;
    height: 12px;
    background: #00e5ff;
    border-radius: 50%;
    box-shadow: 0 0 10px #00e5ff;
}

.robot-head {
    position: absolute;
    top: 14px;
    left: 9px;
    width: 74px;
    height: 55px;
    background: linear-gradient(145deg, #e5fbff, #81d8e7);
    border: 3px solid #00e5ff;
    border-radius: 18px;
    box-shadow: 0 0 16px rgba(0,229,255,0.25);
}

.robot-ear {
    position: absolute;
    top: 34px;
    width: 9px;
    height: 18px;
    background: #00bcd4;
    border-radius: 5px;
}

.robot-ear.left { left: 1px; }
.robot-ear.right { right: 1px; }

.robot-eyes {
    display: flex;
    justify-content: center;
    gap: 19px;
    padding-top: 13px;
}

.robot-eye {
    width: 9px;
    height: 9px;
    background: #07354b;
    border-radius: 50%;
    animation: eye-glow 1.8s ease-in-out infinite;
}

.robot-mouth {
    width: 23px;
    height: 7px;
    margin: 7px auto 0;
    border-bottom: 3px solid #087e98;
    border-radius: 50%;
}

.mascot-title {
    color: #e8fbff;
    font-size: 15px;
    font-weight: 750;
}

.mascot-caption {
    color: #91b4c2;
    font-size: 11px;
    margin-top: 4px;
}

.heartbeat-line {
    height: 2px;
    width: 70%;
    margin: 12px auto 0;
    background: linear-gradient(90deg, transparent, #00e5ff, transparent);
    animation: heartbeat 1.5s ease-in-out infinite;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(16,36,56,0.96), rgba(10,25,40,0.96));
    border: 1px solid rgba(0,229,255,0.2);
    padding: 17px 18px;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    transition: transform 0.25s ease, border-color 0.25s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: rgba(0,229,255,0.65);
}

div[data-testid="stMetricLabel"] {
    color: #9cb8c9;
}

div[data-testid="stMetricValue"] {
    color: #e8fbff;
    font-weight: 750;
}

div[data-testid="stTabs"] button {
    color: #a9c4d3;
    font-weight: 600;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00e5ff;
}

.stButton button, .stDownloadButton button {
    border: 1px solid rgba(0,229,255,0.4);
    border-radius: 10px;
    background: linear-gradient(100deg, #0c3445, #10283a);
    color: #e8fbff;
    transition: all 0.2s ease;
}

.stButton button:hover, .stDownloadButton button:hover {
    border-color: #00e5ff;
    box-shadow: 0 0 14px rgba(0,229,255,0.18);
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,229,255,0.16);
    border-radius: 12px;
    overflow: hidden;
}

.section-note {
    color: #91a9bb;
    font-size: 12px;
}

@media (prefers-reduced-motion: reduce) {
    .pulse-ring, .hospital-cross, .robot, .robot-eye, .heartbeat-line {
        animation: none !important;
    }
}
</style>
""", unsafe_allow_html=True)

def render_hospital_brand():
    st.markdown("""
    <div class="hospital-brand">
        <div class="hospital-icon">
            <div class="pulse-ring"></div>
            <div class="pulse-ring delay"></div>
            <div class="hospital-cross">+</div>
        </div>
        <div>
            <h1>MediScope</h1>
            <p>HEALTH ANALYTICS · SPARCS 2021</p>
        </div>
    </div>
    <div class="hero-subtitle">Hospital inpatient discharges, utilization, and financial performance</div>
    """, unsafe_allow_html=True)

def render_mascot():
    st.sidebar.markdown("""
    <div class="sidebar-mascot">
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
        </div>
        <div class="mascot-title">MediBot</div>
        <div class="mascot-caption">Your healthcare data companion</div>
        <div class="heartbeat-line"></div>
    </div>
    """, unsafe_allow_html=True)

def find_column(df, aliases):
    lookup = {str(col).strip().lower(): col for col in df.columns}
    for alias in aliases:
        key = alias.strip().lower()
        if key in lookup:
            return lookup[key]
    return None

COLUMN_ALIASES = {
    "Length_of_Stay": [
        "Length of Stay", "Length_of_Stay", "LOS"
    ],
    "Total_Charges": [
        "Total Charges", "Total_Charges", "Charges"
    ],
    "Facility_Name": [
        "Facility Name", "Facility_Name", "Hospital Name"
    ],
    "County": [
        "Hospital County", "County"
    ],
    "Diagnosis_Category": [
        "APR MDC Description", "MDC Description",
        "APR DRG Description", "CCS Diagnosis Description",
        "Diagnosis Category"
    ],
    "Severity_Level": [
        "APR Severity of Illness Description",
        "Severity of Illness Description", "Severity Level",
        "Severity"
    ],
    "Payment_Type": [
        "Payment Typology 1", "Payment Type", "Payment_Type"
    ],
    "Age_Group": ["Age Group", "Age_Group"],
    "Gender": ["Gender"],
    "Race": ["Race"],
    "Discharge_Year": ["Discharge Year", "Year"],
    "Discharge_Date": ["Discharge Date"]
}

@st.cache_data(ttl=3600, show_spinner="Loading and cleaning the SPARCS dataset...")
def load_sparcs_data(file_path, sample_size):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found: {file_path}. Put the CSV in the same folder as app.py."
        )

    nrows = None if sample_size == "Full Dataset" else int(sample_size)
    df = pd.read_csv(file_path, nrows=nrows, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {}
    for standard_name, aliases in COLUMN_ALIASES.items():
        original = find_column(df, aliases)
        if original is not None and original != standard_name:
            rename_map[original] = standard_name
    df = df.rename(columns=rename_map)

    defaults = {
        "Facility_Name": "Unknown Facility",
        "County": "Unknown County",
        "Diagnosis_Category": "General Medical",
        "Severity_Level": "Unknown",
        "Payment_Type": "Unknown",
        "Age_Group": "Unknown",
        "Gender": "Unknown",
        "Race": "Unknown"
    }
    for column, default_value in defaults.items():
        if column not in df.columns:
            df[column] = default_value
        df[column] = (
            df[column].fillna(default_value).astype(str).str.strip()
            .replace("", default_value)
        )

    if "Length_of_Stay" not in df.columns:
        raise ValueError(
            "Could not find a Length of Stay column. Check the dataset's column names."
        )

    df["Length_of_Stay"] = (
        df["Length_of_Stay"].astype(str)
        .str.replace("+", "", regex=False)
    )
    df["Length_of_Stay"] = pd.to_numeric(
        df["Length_of_Stay"], errors="coerce"
    )

    if "Total_Charges" not in df.columns:
        raise ValueError(
            "Could not find a Total Charges column. Check the dataset's column names."
        )

    df["Total_Charges"] = (
        df["Total_Charges"].astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df["Total_Charges"] = pd.to_numeric(
        df["Total_Charges"], errors="coerce"
    )

    year_filtered = False
    if "Discharge_Year" in df.columns:
        year_values = pd.to_numeric(df["Discharge_Year"], errors="coerce")
        before = len(df)
        df = df[year_values == 2021].copy()
        year_filtered = len(df) != before
    elif "Discharge_Date" in df.columns:
        dates = pd.to_datetime(df["Discharge_Date"], errors="coerce")
        df = df[dates.dt.year == 2021].copy()
        year_filtered = True

    df = df.dropna(subset=["Length_of_Stay", "Total_Charges"])
    df = df[df["Length_of_Stay"] > 0].copy()

    df["Total_Charges"] = df["Total_Charges"].replace(
        [np.inf, -np.inf], np.nan
    )
    df = df.dropna(subset=["Total_Charges"])
    df = df[df["Total_Charges"] >= 0].copy()

    for col in ["Facility_Name", "County", "Diagnosis_Category",
                "Severity_Level", "Payment_Type", "Age_Group",
                "Gender", "Race"]:
        df[col] = df[col].astype(str).str.strip()

    df["Severity_Level"] = df["Severity_Level"].replace({
        "Minor": "1 - Minor",
        "Moderate": "2 - Moderate",
        "Major": "3 - Major",
        "Extreme": "4 - Extreme"
    })

    df["Payment_Type"] = df["Payment_Type"].replace({
        "Private Health Insurance": "Private Insurance",
        "Blue Cross": "Private Insurance",
        "Self-Pay": "Self-Pay"
    })

    df.attrs["year_filtered"] = year_filtered
    return df

def make_plot_theme(fig, height=400):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d9efff", family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=45, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(
            bgcolor="#102438", bordercolor="#00e5ff",
            font_color="#e8fbff"
        )
    )
    fig.update_xaxes(
        gridcolor="rgba(145,169,187,0.14)",
        zerolinecolor="rgba(145,169,187,0.2)"
    )
    fig.update_yaxes(
        gridcolor="rgba(145,169,187,0.14)",
        zerolinecolor="rgba(145,169,187,0.2)"
    )
    return fig

def currency(value):
    return f"${value:,.2f}"

def get_ai_experiment(df):
    feature_candidates = [
        "Age_Group", "Gender", "Race", "County",
        "Severity_Level", "Payment_Type", "Diagnosis_Category"
    ]
    features = [c for c in feature_candidates if c in df.columns]
    model_df = df[features + ["Length_of_Stay"]].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(model_df) < 100 or len(features) == 0:
        return None, "Not enough complete records for a meaningful demonstration."

    threshold = model_df["Length_of_Stay"].median()
    model_df["Stay_Group"] = np.where(
        model_df["Length_of_Stay"] > threshold, "Longer stay", "Shorter stay"
    )
    y = model_df["Stay_Group"]
    X = model_df[features]

    if y.nunique() < 2 or y.value_counts().min() < 2:
        return None, "The selected data does not contain enough examples in both stay groups."

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numeric_features = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    transformers = []
    if categorical_features:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        ))
    if numeric_features:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_features
        ))

    preprocessor = ColumnTransformer(transformers=transformers)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight="balanced"
            ))
        ]),
        "Dummy baseline": DummyClassifier(strategy="most_frequent")
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions)
        })

    return pd.DataFrame(results), None

def main():
    render_hospital_brand()
    st.markdown(
        "Explore discharge patterns, hospital utilization, patient demographics, "
        "and billing metrics using the SPARCS 2021 dataset."
    )

    render_mascot()

    st.sidebar.markdown("## Data Settings & Filters")
    st.sidebar.caption("MediScope · SPARCS 2021")

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), TARGET_FILE)

    if not os.path.exists(data_path):
        nearby_csvs = glob.glob(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "*.csv")
        )
        if nearby_csvs:
            data_path = nearby_csvs[0]
        else:
            st.error(
                f"Dataset file not found: `{TARGET_FILE}`. "
                "Place the CSV in the same directory as app.py."
            )
            st.stop()

    sample_size = st.sidebar.selectbox(
        "Rows to load",
        options=[10000, 25000, 50000, 100000, "Full Dataset"],
        index=2,
        help="Loading fewer rows can improve startup speed."
    )

    try:
        df_raw = load_sparcs_data(data_path, sample_size)
    except Exception as exc:
        st.error(f"Unable to load or preprocess the dataset: {exc}")
        st.stop()

    st.sidebar.success(f"Loaded {len(df_raw):,} cleaned records")

    if df_raw.empty:
        st.error("No valid records remain after cleaning. Check the source columns.")
        st.stop()

    all_counties = sorted(df_raw["County"].dropna().unique().tolist())
    all_severity = sorted(df_raw["Severity_Level"].dropna().unique().tolist())
    all_payers = sorted(df_raw["Payment_Type"].dropna().unique().tolist())

    selected_counties = st.sidebar.multiselect(
        "County",
        options=all_counties,
        default=all_counties[:5] if len(all_counties) > 5 else all_counties
    )
    selected_severity = st.sidebar.multiselect(
        "Severity level",
        options=all_severity,
        default=all_severity
    )
    selected_payers = st.sidebar.multiselect(
        "Payment type",
        options=all_payers,
        default=all_payers
    )

    df_filtered = df_raw[
        df_raw["County"].isin(selected_counties)
        & df_raw["Severity_Level"].isin(selected_severity)
        & df_raw["Payment_Type"].isin(selected_payers)
    ].copy()

    if df_filtered.empty:
        st.warning("No records match these filters. Select additional counties, severity levels, or payment types.")
        st.stop()

    total_records = len(df_filtered)
    average_stay = df_filtered["Length_of_Stay"].mean()
    total_charges = df_filtered["Total_Charges"].sum()
    average_charge = df_filtered["Total_Charges"].mean()
    median_stay = df_filtered["Length_of_Stay"].median()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Discharges analyzed", f"{total_records:,}")
    k2.metric("Average stay", f"{average_stay:.1f} days")
    k3.metric("Total charges", currency(total_charges))
    k4.metric("Average charge", currency(average_charge))

    st.markdown("---")

    tab_overview, tab_diagnosis, tab_demographics, tab_facilities, tab_quality, tab_ai, tab_data = st.tabs([
        "Overview", "Diagnosis & Stay", "Charges & Demographics",
        "Facilities & Counties", "Data Quality", "AI Experiment", "Data Explorer"
    ])

    with tab_overview:
        st.subheader("Discharge overview")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Discharges by severity")
            severity_data = (
                df_filtered["Severity_Level"].value_counts()
                .rename_axis("Severity level").reset_index(name="Discharges")
            )
            fig = px.bar(
                severity_data, x="Severity level", y="Discharges",
                color="Severity level", text_auto=True,
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(make_plot_theme(fig), use_container_width=True)

        with c2:
            st.markdown("#### Patient volume by payment type")
            payer_data = (
                df_filtered["Payment_Type"].value_counts()
                .head(8).rename_axis("Payment type")
                .reset_index(name="Discharges")
            )
            fig = px.pie(
                payer_data, names="Payment type", values="Discharges",
                hole=0.58, color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(make_plot_theme(fig), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("#### Length-of-stay distribution")
            stay_data = df_filtered[df_filtered["Length_of_Stay"] <= 60]
            fig = px.histogram(
                stay_data, x="Length_of_Stay", nbins=30,
                labels={"Length_of_Stay": "Length of stay (days)"},
                color_discrete_sequence=["#00e5ff"]
            )
            st.plotly_chart(make_plot_theme(fig), use_container_width=True)

        with c4:
            st.markdown("#### Discharges by age group")
            age_data = (
                df_filtered["Age_Group"].value_counts()
                .rename_axis("Age group").reset_index(name="Discharges")
            )
            fig = px.bar(
                age_data, x="Age group", y="Discharges",
                color="Discharges", text_auto=True,
                color_continuous_scale=["#075985", "#00e5ff"]
            )
            st.plotly_chart(make_plot_theme(fig), use_container_width=True)

    with tab_diagnosis:
        st.subheader("Diagnosis and length of stay")
        diagnosis_data = (
            df_filtered.groupby("Diagnosis_Category", dropna=False)
            .agg(
                Average_Stay=("Length_of_Stay", "mean"),
                Discharges=("Length_of_Stay", "size")
            )
            .reset_index()
            .sort_values("Average_Stay", ascending=False)
            .head(15)
        )
        fig = px.bar(
            diagnosis_data.sort_values("Average_Stay"),
            x="Average_Stay", y="Diagnosis_Category", orientation="h",
            color="Average_Stay", text_auto=".1f",
            color_continuous_scale=["#075985", "#00e5ff"],
            labels={
                "Average_Stay": "Average stay (days)",
                "Diagnosis_Category": "Diagnosis category"
            }
        )
        st.plotly_chart(make_plot_theme(fig, 520), use_container_width=True)

        st.markdown("#### Charges by severity")
        fig = px.box(
            df_filtered, x="Severity_Level", y="Total_Charges",
            color="Severity_Level",
            points="outliers",
            labels={
                "Severity_Level": "Severity level",
                "Total_Charges": "Total charges ($)"
            },
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(make_plot_theme(fig, 450), use_container_width=True)

    with tab_demographics:
        st.subheader("Charges and demographics")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Average charges by age group")
            age_charges = (
                df_filtered.groupby("Age_Group", dropna=False)["Total_Charges"]
                .mean().reset_index(name="Average_Charges")
                .sort_values("Average_Charges", ascending=False)
            )
            fig = px.bar(
                age_charges, x="Age_Group", y="Average_Charges",
                color="Average_Charges", text_auto=".2s",
                color_continuous_scale=["#075985", "#00e5ff"],
                labels={"Average_Charges": "Average charges ($)", "Age_Group": "Age group"}
            )
            st.plotly_chart(make_plot_theme(fig), use_container_width=True)

        with c2:
            st.markdown("#### Average charges by gender")
            gender_charges = (
                df_filtered.groupby("Gender", dropna=False)["Total_Charges"]
                .mean().reset_index(name="Average_Charges")
            )
            fig = px.bar(
                gender_charges, x="Gender", y="Average_Charges",
                color="Gender", text_auto=".2s",
                labels={"Average_Charges": "Average charges ($)", "Gender": "Gender"}
            )
            st.plotly_chart(make_plot_theme(fig), use_container_width=True)

        st.markdown("#### Severity and age-group comparison")
        severity_age = pd.pivot_table(
            df_filtered, index="Age_Group", columns="Severity_Level",
            values="Length_of_Stay", aggfunc="mean"
        )
        fig = px.imshow(
            severity_age, aspect="auto",
            labels={"x": "Severity level", "y": "Age group", "color": "Average stay"},
            color_continuous_scale="Viridis", text_auto=".1f"
        )
        st.plotly_chart(make_plot_theme(fig, 430), use_container_width=True)

    with tab_facilities:
        st.subheader("Facility and county performance")
        top_facilities = (
            df_filtered["Facility_Name"].value_counts().head(12)
            .rename_axis("Facility").reset_index(name="Discharges")
        )
        fig = px.bar(
            top_facilities.sort_values("Discharges"),
            x="Discharges", y="Facility", orientation="h",
            color="Discharges", text_auto=True,
            color_continuous_scale=["#075985", "#00e5ff"]
        )
        st.markdown("#### Facilities with the most discharges")
        st.plotly_chart(make_plot_theme(fig, 500), use_container_width=True)

        st.markdown("#### Average stay by facility and county")
        top_facility_names = df_filtered["Facility_Name"].value_counts().head(10).index
        heatmap_df = df_filtered[
            df_filtered["Facility_Name"].isin(top_facility_names)
        ].pivot_table(
            index="Facility_Name", columns="County",
            values="Length_of_Stay", aggfunc="mean"
        )
        if not heatmap_df.empty:
            fig = px.imshow(
                heatmap_df, aspect="auto",
                labels={"x": "County", "y": "Facility", "color": "Average stay (days)"},
                color_continuous_scale="Viridis", text_auto=".1f"
            )
            st.plotly_chart(make_plot_theme(fig, 480), use_container_width=True)

        st.markdown("#### County summary")
        county_summary = (
            df_filtered.groupby("County")
            .agg(
                Discharges=("Length_of_Stay", "size"),
                Average_Stay=("Length_of_Stay", "mean"),
                Median_Stay=("Length_of_Stay", "median"),
                Average_Charges=("Total_Charges", "mean"),
                Total_Charges=("Total_Charges", "sum")
            )
            .sort_values("Discharges", ascending=False)
            .round(2)
        )
        st.dataframe(county_summary, use_container_width=True)

    with tab_quality:
        st.subheader("Data quality and cleaning summary")
        q1, q2, q3 = st.columns(3)
        q1.metric("Clean records", f"{len(df_raw):,}")
        q2.metric("Filtered records", f"{len(df_filtered):,}")
        q3.metric("Columns", f"{df_raw.shape[1]:,}")

        st.markdown("#### Missing values in cleaned dataset")
        missing_df = (
            df_raw.isna().sum().rename("Missing values")
            .to_frame().assign(
                Missing_Percent=lambda x: (
                    x["Missing values"] / max(len(df_raw), 1) * 100
                ).round(2)
            )
            .sort_values("Missing values", ascending=False)
        )
        st.dataframe(missing_df, use_container_width=True)

        st.markdown("#### Length-of-stay outliers")
        q_low = df_raw["Length_of_Stay"].quantile(0.25)
        q_high = df_raw["Length_of_Stay"].quantile(0.75)
        iqr = q_high - q_low
        upper_limit = q_high + 1.5 * iqr
        outliers = df_raw[df_raw["Length_of_Stay"] > upper_limit]
        st.write(
            f"Using the 1.5×IQR rule, **{len(outliers):,}** records have "
            f"length of stay above **{upper_limit:.1f} days**."
        )
        st.dataframe(
            outliers[[
                "Facility_Name", "County", "Diagnosis_Category",
                "Length_of_Stay", "Total_Charges"
            ]].head(100),
            use_container_width=True
        )

    with tab_ai:
        st.subheader("AI experiment: predicting longer hospital stays")
        st.write(
            "This demonstration predicts whether a stay is above the median "
            "length of stay in the loaded dataset. It is an educational model, "
            "not a clinical decision-making tool."
        )
        if st.button("Train and compare models"):
            with st.spinner("Training models..."):
                try:
                    results, error = get_ai_experiment(df_filtered)
                    if error:
                        st.warning(error)
                    else:
                        st.dataframe(results, use_container_width=True)
                        fig = px.bar(
                            results, x="Model", y="Accuracy",
                            color="Model", text_auto=".3f",
                            range_y=[0, 1],
                            color_discrete_sequence=["#00e5ff", "#00f5c3"]
                        )
                        st.plotly_chart(make_plot_theme(fig, 350), use_container_width=True)
                        st.info(
                            "Accuracy alone does not establish clinical usefulness. "
                            "Review class balance and additional metrics before drawing conclusions."
                        )
                except Exception as exc:
                    st.error(f"Model training failed: {exc}")

    with tab_data:
        st.subheader("Data explorer")
        st.caption("Preview of cleaned records based on your current filters.")
        max_preview = st.slider(
            "Number of preview rows", min_value=10, max_value=500,
            value=100, step=10
        )
        st.dataframe(
            df_filtered.head(max_preview),
            use_container_width=True,
            height=420
        )
        st.download_button(
            "Download filtered data as CSV",
            data=df_filtered.to_csv(index=False).encode("utf-8"),
            file_name="mediscope_filtered_discharges.csv",
            mime="text/csv"
        )
        st.download_button(
            "Download summary by county",
            data=(
                df_filtered.groupby("County")
                .agg(
                    Discharges=("Length_of_Stay", "size"),
                    Average_Stay=("Length_of_Stay", "mean"),
                    Average_Charges=("Total_Charges", "mean")
                )
                .round(2).reset_index().to_csv(index=False).encode("utf-8")
            ),
            file_name="mediscope_county_summary.csv",
            mime="text/csv"
        )

    st.markdown("---")
    st.markdown(
        '<p class="section-note">MediScope Health Analytics · SPARCS 2021 · '
        'For educational and analytical purposes only.</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
<CodeBlock language="python" editable>
import os
import glob
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="MediScope | Hospital Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

TARGET_FILE = "Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #07111f;
    --panel: #0d1b2b;
    --panel2: #102438;
    --cyan: #00e5ff;
    --teal: #00f5c3;
    --text: #e8f7ff;
    --muted: #91a9bb;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(0,229,255,0.09), transparent 32%),
        radial-gradient(ellipse at 90% 10%, rgba(0,245,195,0.06), transparent 30%),
        var(--bg);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(7,17,31,0.75);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1928 0%, #08121f 100%);
    border-right: 1px solid rgba(0,229,255,0.18);
}

[data-testid="stSidebar"] * {
    color: #dff7ff;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

@keyframes hospital-pulse {
    0% { transform: scale(0.78); opacity: 0.8; }
    70% { transform: scale(1.35); opacity: 0; }
    100% { transform: scale(1.35); opacity: 0; }
}

@keyframes cross-glow {
    0%, 100% {
        filter: drop-shadow(0 0 4px #00e5ff);
        transform: scale(1);
    }
    50% {
        filter: drop-shadow(0 0 16px #00e5ff);
        transform: scale(1.08);
    }
}

@keyframes mascot-float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-7px); }
}

@keyframes eye-glow {
    0%, 100% { box-shadow: 0 0 4px #00e5ff; }
    50% { box-shadow: 0 0 13px #00e5ff; }
}

@keyframes heartbeat {
    0%, 100% { transform: scaleX(0.85); opacity: 0.65; }
    50% { transform: scaleX(1.08); opacity: 1; }
}

.hospital-brand {
    display: flex;
    align-items: center;
    gap: 16px;
    margin: 4px 0 12px;
}

.hospital-icon {
    width: 72px;
    height: 72px;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.pulse-ring {
    position: absolute;
    inset: 6px;
    border: 2px solid #00e5ff;
    border-radius: 20px;
    animation: hospital-pulse 2.2s ease-out infinite;
}

.pulse-ring.delay {
    animation-delay: 1.1s;
}

.hospital-cross {
    z-index: 2;
    position: relative;
    color: #00e5ff;
    font-size: 52px;
    font-weight: 800;
    line-height: 1;
    animation: cross-glow 2s ease-in-out infinite;
}

.hospital-brand h1 {
    color: #e8fbff;
    font-size: 30px;
    font-weight: 800;
    margin: 0;
    padding: 0;
    letter-spacing: -0.8px;
}

.hospital-brand p {
    color: #86aebc;
    font-size: 11px;
    letter-spacing: 2px;
    margin: 5px 0 0;
}

.hero-subtitle {
    color: #9cb8c9;
    font-size: 14px;
    margin: 0 0 18px;
}

.sidebar-mascot {
    margin: 10px 0 22px;
    padding: 18px 10px 14px;
    border: 1px solid rgba(0,229,255,0.3);
    border-radius: 18px;
    background: linear-gradient(145deg, #10283a, #091522);
    text-align: center;
    overflow: hidden;
    box-shadow: 0 8px 28px rgba(0,0,0,0.15);
}

.robot {
    position: relative;
    width: 92px;
    height: 86px;
    margin: 0 auto 12px;
    animation: mascot-float 3s ease-in-out infinite;
}

.robot-antenna {
    position: absolute;
    top: 0;
    left: 44px;
    width: 4px;
    height: 14px;
    background: #00e5ff;
    border-radius: 4px;
}

.robot-antenna::before {
    content: "";
    position: absolute;
    top: -5px;
    left: -4px;
    width: 12px;
    height: 12px;
    background: #00e5ff;
    border-radius: 50%;
    box-shadow: 0 0 10px #00e5ff;
}

.robot-head {
    position: absolute;
    top: 14px;
    left: 9px;
    width: 74px;
    height: 55px;
    background: linear-gradient(145deg, #e5fbff, #81d8e7);
    border: 3px solid #00e5ff;
    border-radius: 18px;
    box-shadow: 0 0 16px rgba(0,229,255,0.25);
}

.robot-ear {
    position: absolute;
    top: 34px;
    width: 9px;
    height: 18px;
    background: #00bcd4;
    border-radius: 5px;
}

.robot-ear.left { left: 1px; }
.robot-ear.right { right: 1px; }

.robot-eyes {
    display: flex;
    justify-content: center;
    gap: 19px;
    padding-top: 13px;
}

.robot-eye {
    width: 9px;
    height: 9px;
    background: #07354b;
    border-radius: 50%;
    animation: eye-glow 1.8s ease-in-out infinite;
}

.robot-mouth {
    width: 23px;
    height: 7px;
    margin: 7px auto 0;
    border-bottom: 3px solid #087e98;
    border-radius: 50%;
}

.mascot-title {
    color: #e8fbff;
    font-size: 15px;
    font-weight: 750;
}

.mascot-caption {
    color: #91b4c2;
    font-size: 11px;
    margin-top: 4px;
}

.heartbeat-line {
    height: 2px;
    width: 70%;
    margin: 12px auto 0;
    background: linear-gradient(90deg, transparent, #00e5ff, transparent);
    animation: heartbeat 1.5s ease-in-out infinite;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(16,36,56,0.96), rgba(10,25,40,0.96));
    border: 1px solid rgba(0,229,255,0.2);
    padding: 17px 18px;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    transition: transform 0.25s ease, border-color 0.25s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: rgba(0,229,255,0.65);
}

div[data-testid="stMetricLabel"] {
    color: #9cb8c9;
}

div[data-testid="stMetricValue"] {
    color: #e8fbff;
    font-weight: 750;
}

div[data-testid="stTabs"] button {
    color: #a9c4d3;
    font-weight: 600;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00e5ff;
}

.stButton button, .stDownloadButton button {
    border: 1px solid rgba(0,229,255,0.4);
    border-radius: 10px;
    background: linear-gradient(100deg, #0c3445, #10283a);
    color: #e8fbff;
    transition: all 0.2s ease;
}

.stButton button:hover, .stDownloadButton button:hover {
    border-color: #00e5ff;
    box-shadow: 0 0 14px rgba(0,229,255,0.18);
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,229,255,0.16);
    border-radius: 12px;
    overflow: hidden;
}

.section-note {
    color: #91a9bb;
    font-size: 12px;
}

@media (prefers-reduced-motion: reduce) {
    .pulse-ring, .hospital-cross, .robot, .robot-eye, .heartbeat-line {
        animation: none !important;
    }
}
</style>
""", unsafe_allow_html=True)


def render_hospital_brand():
    st.markdown("""
    <div class="hospital-brand">
        <div class="hospital-icon">
            <div class="pulse-ring"></div>
            <div class="pulse-ring delay"></div>
            <div class="hospital-cross">+</div>
        </div>
        <div>
            <h1>MediScope</h1>
            <p>HEALTH ANALYTICS · SPARCS 2021</p>
        </div>
    </div>
    <div class="hero-subtitle">
        Hospital inpatient discharges, utilization, and financial performance
    </div>
    """, unsafe_allow_html=True)


def render_mascot():
    st.sidebar.markdown("""
    <div class="sidebar-mascot">
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
        </div>
        <div class="mascot-title">MediBot</div>
        <div class="mascot-caption">Your healthcare data companion</div>
        <div class="heartbeat-line"></div>
    </div>
    """, unsafe_allow_html=True)


def find_column(df, aliases):
    lookup = {str(col).strip().lower(): col for col in df.columns}
    for alias in aliases:
        key = alias.strip().lower()
        if key in lookup:
            return lookup[key]
    return None


COLUMN_ALIASES = {
    "Length_of_Stay": [
        "Length of Stay", "Length_of_Stay", "LOS"
    ],
    "Total_Charges": [
        "Total Charges", "Total_Charges", "Charges"
    ],
    "Facility_Name": [
        "Facility Name", "Facility_Name", "Hospital Name"
    ],
    "County": [
        "Hospital County", "County"
    ],
    "Diagnosis_Category": [
        "APR MDC Description", "MDC Description",
        "APR DRG Description", "CCS Diagnosis Description",
        "Diagnosis Category"
    ],
    "Severity_Level": [
        "APR Severity of Illness Description",
        "Severity of Illness Description", "Severity Level",
        "Severity"
    ],
    "Payment_Type": [
        "Payment Typology 1", "Payment Type", "Payment_Type"
    ],
    "Age_Group": ["Age Group", "Age_Group"],
    "Gender": ["Gender"],
    "Race": ["Race"],
    "Discharge_Year": ["Discharge Year", "Year"],
    "Discharge_Date": ["Discharge Date"]
}


@st.cache_data(ttl=3600, show_spinner="Loading and cleaning the SPARCS dataset...")
def load_sparcs_data(file_path, sample_size):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found: {file_path}. Put the CSV in the same folder as app.py."
        )

    nrows = None if sample_size == "Full Dataset" else int(sample_size)
    df = pd.read_csv(file_path, nrows=nrows, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {}
    for standard_name, aliases in COLUMN_ALIASES.items():
        original = find_column(df, aliases)
        if original is not None and original != standard_name:
            rename_map[original] = standard_name

    df = df.rename(columns=rename_map)

    defaults = {
        "Facility_Name": "Unknown Facility",
        "County": "Unknown County",
        "Diagnosis_Category": "General Medical",
        "Severity_Level": "Unknown",
        "Payment_Type": "Unknown",
        "Age_Group": "Unknown",
        "Gender": "Unknown",
        "Race": "Unknown"
    }

    for column, default_value in defaults.items():
        if column not in df.columns:
            df[column] = default_value

        df[column] = (
            df[column].fillna(default_value).astype(str).str.strip()
            .replace("", default_value)
        )

    if "Length_of_Stay" not in df.columns:
        raise ValueError(
            "Could not find a Length of Stay column. Check the dataset's column names."
        )

    df["Length_of_Stay"] = (
        df["Length_of_Stay"].astype(str)
        .str.replace("+", "", regex=False)
    )
    df["Length_of_Stay"] = pd.to_numeric(
        df["Length_of_Stay"], errors="coerce"
    )

    if "Total_Charges" not in df.columns:
        raise ValueError(
            "Could not find a Total Charges column. Check the dataset's column names."
        )

    df["Total_Charges"] = (
        df["Total_Charges"].astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df["Total_Charges"] = pd.to_numeric(
        df["Total_Charges"], errors="coerce"
    )

    year_filtered = False

    if "Discharge_Year" in df.columns:
        year_values = pd.to_numeric(df["Discharge_Year"], errors="coerce")
        before = len(df)
        df = df[year_values == 2021].copy()
        year_filtered = len(df) != before

    elif "Discharge_Date" in df.columns:
        dates = pd.to_datetime(df["Discharge_Date"], errors="coerce")
        df = df[dates.dt.year == 2021].copy()
        year_filtered = True

    df = df.dropna(subset=["Length_of_Stay", "Total_Charges"])
    df = df[df["Length_of_Stay"] > 0].copy()

    df["Total_Charges"] = df["Total_Charges"].replace(
        [np.inf, -np.inf], np.nan
    )
    df = df.dropna(subset=["Total_Charges"])
    df = df[df["Total_Charges"] >= 0].copy()

    for col in [
        "Facility_Name", "County", "Diagnosis_Category",
        "Severity_Level", "Payment_Type", "Age_Group",
        "Gender", "Race"
    ]:
        df[col] = df[col].astype(str).str.strip()

    df["Severity_Level"] = df["Severity_Level"].replace({
        "Minor": "1 - Minor",
        "Moderate": "2 - Moderate",
        "Major": "3 - Major",
        "Extreme": "4 - Extreme"
    })

    df["Payment_Type"] = df["Payment_Type"].replace({
        "Private Health Insurance": "Private Insurance",
        "Blue Cross": "Private Insurance",
        "Self-Pay": "Self-Pay"
    })

    df.attrs["year_filtered"] = year_filtered
    return df


def make_plot_theme(fig, height=400):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d9efff", family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=45, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(
            bgcolor="#102438",
            bordercolor="#00e5ff",
            font_color="#e8fbff"
        )
    )

    fig.update_xaxes(
        gridcolor="rgba(145,169,187,0.14)",
        zerolinecolor="rgba(145,169,187,0.2)"
    )

    fig.update_yaxes(
        gridcolor="rgba(145,169,187,0.14)",
        zerolinecolor="rgba(145,169,187,0.2)"
    )

    return fig


def currency(value):
    return f"${value:,.2f}"


def get_ai_experiment(df):
    feature_candidates = [
        "Age_Group", "Gender", "Race", "County",
        "Severity_Level", "Payment_Type", "Diagnosis_Category"
    ]

    features = [c for c in feature_candidates if c in df.columns]
    model_df = df[features + ["Length_of_Stay"]].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(model_df) < 100 or not features:
        return None, "Not enough complete records for a meaningful demonstration."

    threshold = model_df["Length_of_Stay"].median()

    model_df["Stay_Group"] = np.where(
        model_df["Length_of_Stay"] > threshold,
        "Longer stay",
        "Shorter stay"
    )

    y = model_df["Stay_Group"]
    X = model_df[features]

    if y.nunique() < 2 or y.value_counts().min() < 2:
        return None, "The selected data does not contain enough examples in both stay groups."

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_features = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    transformers = []

    if categorical_features:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        ))

    if numeric_features:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_features
        ))

    preprocessor = ColumnTransformer(transformers=transformers)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    models = {
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced"
            ))
        ]),
        "Dummy baseline": DummyClassifier(strategy="most_frequent")
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions)
        })

    return pd.DataFrame(results), None


def main():
    render_hospital_brand()

    st.markdown(
        "Explore discharge patterns, hospital utilization, patient demographics, "
        "and billing metrics using the SPARCS 2021 dataset."
    )

    render_mascot()

    st.sidebar.markdown("## Data Settings & Filters")
    st.sidebar.caption("MediScope · SPARCS 2021")

    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        TARGET_FILE
    )

    if not os.path.exists(data_path):
        nearby_csvs = glob.glob(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "*.csv"
            )
        )

        if nearby_csvs:
            data_path = nearby_csvs[0]
        else:
            st.error(
                f"Dataset file not found: `{TARGET_FILE}`. "
                "Place the CSV in the same directory as app.py."
            )
            st.stop()

    sample_size = st.sidebar.selectbox(
        "Rows to load",
        options=[10000, 25000, 50000, 100000, "Full Dataset"],
        index=2,
        help="Loading fewer rows can improve startup speed."
    )

    try:
        df_raw = load_sparcs_data(data_path, sample_size)
    except Exception as exc:
        st.error(f"Unable to load or preprocess the dataset: {exc}")
        st.stop()

    st.sidebar.success(f"Loaded {len(df_raw):,} cleaned records")

    if df_raw.empty:
        st.error("No valid records remain after cleaning. Check the source columns.")
        st.stop()

    all_counties = sorted(df_raw["County"].dropna().unique().tolist())
    all_severity = sorted(df_raw["Severity_Level"].dropna().unique().tolist())
    all_payers = sorted(df_raw["Payment_Type"].dropna().unique().tolist())

    selected_counties = st.sidebar.multiselect(
        "County",
        options=all_counties,
        default=all_counties[:5] if len(all_counties) > 5 else all_counties
    )

    selected_severity = st.sidebar.multiselect(
        "Severity level",
        options=all_severity,
        default=all_severity
    )

    selected_payers = st.sidebar.multiselect(
        "Payment type",
        options=all_payers,
        default=all_payers
    )

    df_filtered = df_raw[
        df_raw["County"].isin(selected_counties)
        & df_raw["Severity_Level"].isin(selected_severity)
        & df_raw["Payment_Type"].isin(selected_payers)
    ].copy()

    if df_filtered.empty:
        st.warning(
            "No records match these filters. Select additional counties, "
            "severity levels, or payment types."
        )
        st.stop()

    total_records = len(df_filtered)
    average_stay = df_filtered["Length_of_Stay"].mean()
    total_charges = df_filtered["Total_Charges"].sum()
    average_charge = df_filtered["Total_Charges"].mean()

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Discharges analyzed", f"{total_records:,}")
    k2.metric("Average stay", f"{average_stay:.1f} days")
    k3.metric("Total charges", currency(total_charges))
    k4.metric("Average charge", currency(average_charge))

    st.markdown("---")

    (
        tab_overview,
        tab_diagnosis,
        tab_demographics,
        tab_facilities,
        tab_quality,
        tab_ai,
        tab_data
    ) = st.tabs([
        "Overview",
        "Diagnosis & Stay",
        "Charges & Demographics",
        "Facilities & Counties",
        "Data Quality",
        "AI Experiment",
        "Data Explorer"
    ])

    with tab_overview:
        st.subheader("Discharge overview")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Discharges by severity")

            severity_data = (
                df_filtered["Severity_Level"].value_counts()
                .rename_axis("Severity level")
                .reset_index(name="Discharges")
            )

            fig = px.bar(
                severity_data,
                x="Severity level",
                y="Discharges",
                color="Severity level",
                text_auto=True,
                color_discrete_sequence=px.colors.qualitative.Safe
            )

            st.plotly_chart(
                make_plot_theme(fig),
                use_container_width=True
            )

        with c2:
            st.markdown("#### Patient volume by payment type")

            payer_data = (
                df_filtered["Payment_Type"].value_counts()
                .head(8)
                .rename_axis("Payment type")
                .reset_index(name="Discharges")
            )

            fig = px.pie(
                payer_data,
                names="Payment type",
                values="Discharges",
                hole=0.58,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )

            st.plotly_chart(
                make_plot_theme(fig),
                use_container_width=True
            )

        c3, c4 = st.columns(2)

        with c3:
            st.markdown("#### Length-of-stay distribution")

            stay_data = df_filtered[
                df_filtered["Length_of_Stay"] <= 60
            ]

            fig = px.histogram(
                stay_data,
                x="Length_of_Stay",
                nbins=30,
                labels={"Length_of_Stay": "Length of stay (days)"},
                color_discrete_sequence=["#00e5ff"]
            )

            st.plotly_chart(
                make_plot_theme(fig),
                use_container_width=True
            )

        with c4:
            st.markdown("#### Discharges by age group")

            age_data = (
                df_filtered["Age_Group"].value_counts()
                .rename_axis("Age group")
                .reset_index(name="Discharges")
            )

            fig = px.bar(
                age_data,
                x="Age group",
                y="Discharges",
                color="Discharges",
                text_auto=True,
                color_continuous_scale=["#075985", "#00e5ff"]
            )

            st.plotly_chart(
                make_plot_theme(fig),
                use_container_width=True
            )

    with tab_diagnosis:
        st.subheader("Diagnosis and length of stay")

        diagnosis_data = (
            df_filtered.groupby("Diagnosis_Category", dropna=False)
            .agg(
                Average_Stay=("Length_of_Stay", "mean"),
                Discharges=("Length_of_Stay", "size")
            )
            .reset_index()
            .sort_values("Average_Stay", ascending=False)
            .head(15)
        )

        fig = px.bar(
            diagnosis_data.sort_values("Average_Stay"),
            x="Average_Stay",
            y="Diagnosis_Category",
            orientation="h",
            color="Average_Stay",
            text_auto=".1f",
            color_continuous_scale=["#075985", "#00e5ff"],
            labels={
                "Average_Stay": "Average stay (days)",
                "Diagnosis_Category": "Diagnosis category"
            }
        )

        st.plotly_chart(
            make_plot_theme(fig, 520),
            use_container_width=True
        )

        st.markdown("#### Charges by severity")

        fig = px.box(
            df_filtered,
            x="Severity_Level",
            y="Total_Charges",
            color="Severity_Level",
            points="outliers",
            labels={
                "Severity_Level": "Severity level",
                "Total_Charges": "Total charges ($)"
            },
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        st.plotly_chart(
            make_plot_theme(fig, 450),
            use_container_width=True
        )

    with tab_demographics:
        st.subheader("Charges and demographics")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Average charges by age group")

            age_charges = (
                df_filtered.groupby("Age_Group", dropna=False)["Total_Charges"]
                .mean()
                .reset_index(name="Average_Charges")
                .sort_values("Average_Charges", ascending=False)
            )

            fig = px.bar(
                age_charges,
                x="Age_Group",
                y="Average_Charges",
                color="Average_Charges",
                text_auto=".2s",
                color_continuous_scale=["#075985", "#00e5ff"],
                labels={
                    "Average_Charges": "Average charges ($)",
                    "Age_Group": "Age group"
                }
            )

            st.plotly_chart(
                make_plot_theme(fig),
                use_container_width=True
            )

        with c2:
            st.markdown("#### Average charges by gender")

            gender_charges = (
                df_filtered.groupby("Gender", dropna=False)["Total_Charges"]
                .mean()
                .reset_index(name="Average_Charges")
            )

            fig = px.bar(
                gender_charges,
                x="Gender",
                y="Average_Charges",
                color="Gender",
                text_auto=".2s",
                labels={
                    "Average_Charges": "Average charges ($)",
                    "Gender": "Gender"
                }
            )

            st.plotly_chart(
                make_plot_theme(fig),
                use_container_width=True
            )

        st.markdown("#### Severity and age-group comparison")

        severity_age = pd.pivot_table(
            df_filtered,
            index="Age_Group",
            columns="Severity_Level",
            values="Length_of_Stay",
            aggfunc="mean"
        )

        fig = px.imshow(
            severity_age,
            aspect="auto",
            labels={
                "x": "Severity level",
                "y": "Age group",
                "color": "Average stay"
            },
            color_continuous_scale="Viridis",
            text_auto=".1f"
        )

        st.plotly_chart(
            make_plot_theme(fig, 430),
            use_container_width=True
        )

    with tab_facilities:
        st.subheader("Facility and county performance")

        top_facilities = (
            df_filtered["Facility_Name"].value_counts()
            .head(12)
            .rename_axis("Facility")
            .reset_index(name="Discharges")
        )

        fig = px.bar(
            top_facilities.sort_values("Discharges"),
            x="Discharges",
            y="Facility",
            orientation="h",
            color="Discharges",
            text_auto=True,
            color_continuous_scale=["#075985", "#00e5ff"]
        )

        st.markdown("#### Facilities with the most discharges")

        st.plotly_chart(
            make_plot_theme(fig, 500),
            use_container_width=True
        )

        st.markdown("#### Average stay by facility and county")

        top_facility_names = (
            df_filtered["Facility_Name"].value_counts().head(10).index
        )

        heatmap_df = df_filtered[
            df_filtered["Facility_Name"].isin(top_facility_names)
        ].pivot_table(
            index="Facility_Name",
            columns="County",
            values="Length_of_Stay",
            aggfunc="mean"
        )

        if not heatmap_df.empty:
            fig = px.imshow(
                heatmap_df,
                aspect="auto",
                labels={
                    "x": "County",
                    "y": "Facility",
                    "color": "Average stay (days)"
                },
                color_continuous_scale="Viridis",
                text_auto=".1f"
            )

            st.plotly_chart(
                make_plot_theme(fig, 480),
                use_container_width=True
            )

        st.markdown("#### County summary")

        county_summary = (
            df_filtered.groupby("County")
            .agg(
                Discharges=("Length_of_Stay", "size"),
                Average_Stay=("Length_of_Stay", "mean"),
                Median_Stay=("Length_of_Stay", "median"),
                Average_Charges=("Total_Charges", "mean"),
                Total_Charges=("Total_Charges", "sum")
            )
            .sort_values("Discharges", ascending=False)
            .round(2)
        )

        st.dataframe(
            county_summary,
            use_container_width=True
        )

    with tab_quality:
        st.subheader("Data quality and cleaning summary")

        q1, q2, q3 = st.columns(3)

        q1.metric("Clean records", f"{len(df_raw):,}")
        q2.metric("Filtered records", f"{len(df_filtered):,}")
        q3.metric("Columns", f"{df_raw.shape[1]:,}")

        st.markdown("#### Missing values in cleaned dataset")

        missing_df = (
            df_raw.isna().sum()
            .rename("Missing values")
            .to_frame()
            .assign(
                Missing_Percent=lambda x: (
                    x["Missing values"] / max(len(df_raw), 1) * 100
                ).round(2)
            )
            .sort_values("Missing values", ascending=False)
        )

        st.dataframe(
            missing_df,
            use_container_width=True
        )

        st.markdown("#### Length-of-stay outliers")

        q_low = df_raw["Length_of_Stay"].quantile(0.25)
        q_high = df_raw["Length_of_Stay"].quantile(0.75)
        iqr = q_high - q_low
        upper_limit = q_high + 1.5 * iqr

        outliers = df_raw[
            df_raw["Length_of_Stay"] > upper_limit
        ]

        st.write(
            f"Using the 1.5×IQR rule, **{len(outliers):,}** records have "
            f"length of stay above **{upper_limit:.1f} days**."
        )

        st.dataframe(
            outliers[[
                "Facility_Name",
                "County",
                "Diagnosis_Category",
                "Length_of_Stay",
                "Total_Charges"
            ]].head(100),
            use_container_width=True
        )

    with tab_ai:
        st.subheader("AI experiment: predicting longer hospital stays")

        st.write(
            "This demonstration predicts whether a stay is above the median "
            "length of stay in the loaded dataset. It is an educational model, "
            "not a clinical decision-making tool."
        )

        if st.button("Train and compare models"):
            with st.spinner("Training models..."):
                try:
                    results, error = get_ai_experiment(df_filtered)

                    if error:
                        st.warning(error)
                    else:
                        st.dataframe(
                            results,
                            use_container_width=True
                        )

                        fig = px.bar(
                            results,
                            x="Model",
                            y="Accuracy",
                            color="Model",
                            text_auto=".3f",
                            range_y=[0, 1],
                            color_discrete_sequence=["#00e5ff", "#00f5c3"]
                        )

                        st.plotly_chart(
                            make_plot_theme(fig, 350),
                            use_container_width=True
                        )

                        st.info(
                            "Accuracy alone does not establish clinical usefulness. "
                            "Review class balance and additional metrics before drawing conclusions."
                        )

                except Exception as exc:
                    st.error(f"Model training failed: {exc}")

    with tab_data:
        st.subheader("Data explorer")
        st.caption("Preview of cleaned records based on your current filters.")

        max_preview = st.slider(
            "Number of preview rows",
            min_value=10,
            max_value=500,
            value=100,
            step=10
        )

        st.dataframe(
            df_filtered.head(max_preview),
            use_container_width=True,
            height=420
        )

        st.download_button(
            "Download filtered data as CSV",
            data=df_filtered.to_csv(index=False).encode("utf-8"),
            file_name="mediscope_filtered_discharges.csv",
            mime="text/csv"
        )

        county_download = (
            df_filtered.groupby("County")
            .agg(
                Discharges=("Length_of_Stay", "size"),
                Average_Stay=("Length_of_Stay", "mean"),
                Average_Charges=("Total_Charges", "mean")
            )
            .round(2)
            .reset_index()
        )

        st.download_button(
            "Download summary by county",
            data=county_download.to_csv(index=False).encode("utf-8"),
            file_name="mediscope_county_summary.csv",
            mime="text/csv"
        )

    st.markdown("---")

    st.markdown(
        '<p class="section-note">MediScope Health Analytics · SPARCS 2021 · '
        'For educational and analytical purposes only.</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
</CodeBlock>
