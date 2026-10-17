
import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="MediScope | SPARCS 2021",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

TARGET_FILE = "Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv"
CHUNK_SIZE = 50000

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: radial-gradient(ellipse at top left, #142345 0%, #080d19 48%, #060910 100%);
    color: #edf4ff;
    font-family: Inter, sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(5, 10, 22, 0.65);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e1830, #080d19);
    border-right: 1px solid rgba(58, 210, 255, 0.22);
}

h1, h2, h3 {
    color: #f1f7ff !important;
}

p, label, [data-testid="stMarkdownContainer"] {
    color: #d2def3;
}

.hero {
    padding: 26px 28px;
    border-radius: 18px;
    margin-bottom: 18px;
    border: 1px solid rgba(45, 212, 255, 0.35);
    background: linear-gradient(115deg, rgba(20, 50, 91, 0.94), rgba(25, 20, 62, 0.92));
    box-shadow: 0 0 26px rgba(0, 194, 255, 0.10);
    animation: heroGlow 5s ease-in-out infinite alternate;
}

.hero-title {
    font-size: clamp(1.7rem, 3vw, 2.7rem);
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #ffffff, #63e8ff, #a7a4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #b9c9e5;
    margin-top: 8px;
    font-size: 1rem;
}

.metric-card {
    background: linear-gradient(145deg, rgba(20, 35, 63, 0.96), rgba(12, 20, 39, 0.98));
    border: 1px solid rgba(71, 205, 255, 0.25);
    border-radius: 15px;
    padding: 17px;
    min-height: 112px;
    box-shadow: 0 0 18px rgba(35, 183, 255, 0.06);
}

.metric-label {
    font-size: 0.84rem;
    color: #a9bddc;
}

.metric-value {
    font-size: clamp(1.25rem, 2vw, 1.9rem);
    color: #70e7ff;
    font-weight: 800;
    margin-top: 9px;
    overflow-wrap: anywhere;
}

.section-card {
    background: rgba(16, 27, 49, 0.65);
    border: 1px solid rgba(130, 160, 205, 0.18);
    border-radius: 14px;
    padding: 15px 18px;
    margin: 10px 0;
}

.stButton button, .stDownloadButton button {
    border: 1px solid #39cfff;
    border-radius: 10px;
    background: linear-gradient(100deg, #123e61, #30265e);
    color: white;
    transition: 0.2s ease;
}

.stButton button:hover, .stDownloadButton button:hover {
    border-color: #9cf2ff;
    box-shadow: 0 0 16px rgba(57, 207, 255, 0.28);
    transform: translateY(-1px);
}

div[data-testid="stPlotlyChart"] {
    background: rgba(12, 21, 40, 0.55);
    border: 1px solid rgba(105, 154, 215, 0.16);
    border-radius: 12px;
    padding: 4px;
}

@keyframes heroGlow {
    from { box-shadow: 0 0 14px rgba(0, 194, 255, 0.08); }
    to { box-shadow: 0 0 28px rgba(103, 83, 255, 0.18); }
}

@media (max-width: 700px) {
    .hero { padding: 18px; }
    .metric-card { padding: 12px; }
}
</style>
""", unsafe_allow_html=True)


def find_column(columns, candidates):
    normalized = {
        str(column).strip().lower(): column
        for column in columns
    }

    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    for candidate in candidates:
        for normalized_name, original_name in normalized.items():
            if candidate.lower() in normalized_name:
                return original_name

    return None


def clean_category(series, fallback="Unknown"):
    result = series.astype("string").str.strip()
    result = result.replace(
        ["", "nan", "NaN", "None", "NULL", "N/A", "Not Available"],
        pd.NA
    )
    return result.fillna(fallback).astype(str)


def normalize_gender(series):
    values = clean_category(series).str.upper()

    mapping = {
        "M": "Male",
        "MALE": "Male",
        "F": "Female",
        "FEMALE": "Female",
        "U": "Unknown",
        "UNKNOWN": "Unknown",
        "OTHER": "Other"
    }

    return values.map(mapping).fillna(
        values.replace({"UNKNOWN": "Unknown"})
    )


def parse_numeric(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.strip(),
        errors="coerce"
    )


def standardize_columns(data):
    data = data.copy()
    data.columns = [str(c).strip() for c in data.columns]
    columns = list(data.columns)

    mapping_candidates = {
        "Length_of_Stay": [
            "Length of Stay", "Length_of_Stay", "LOS"
        ],
        "Total_Charges": [
            "Total Charges", "Total_Charges"
        ],
        "Facility_Name": [
            "Facility Name", "Facility_Name"
        ],
        "County": [
            "Hospital County", "County"
        ],
        "Severity_Level": [
            "APR Severity of Illness Description",
            "Severity of Illness Description",
            "Severity Level"
        ],
        "Payment_Type": [
            "Payment Typology 1", "Payment Type"
        ],
        "Age_Group": [
            "Age Group", "Age_Group"
        ],
        "Gender": [
            "Gender"
        ],
        "Race": [
            "Race"
        ],
        "Discharge_Date": [
            "Discharge Date", "Discharge_Date"
        ],
        "Diagnosis_Category": [
            "CCS Diagnosis Description",
            "APR DRG Description",
            "APR MDC Description",
            "Diagnosis Description",
            "Diagnosis Category"
        ],
        "Diagnosis_Code": [
            "CCS Diagnosis Code",
            "APR DRG Code",
            "Diagnosis Code"
        ]
    }

    for standard_name, candidates in mapping_candidates.items():
        original = find_column(columns, candidates)
        if original is not None and original != standard_name:
            if standard_name not in data.columns:
                data = data.rename(columns={original: standard_name})

    if "Length_of_Stay" not in data.columns:
        data["Length_of_Stay"] = np.nan

    if "Total_Charges" not in data.columns:
        data["Total_Charges"] = np.nan

    defaults = {
        "Facility_Name": "Unknown Facility",
        "County": "Unknown County",
        "Severity_Level": "Unknown",
        "Payment_Type": "Unknown",
        "Age_Group": "Unknown",
        "Gender": "Unknown",
        "Race": "Unknown",
        "Diagnosis_Category": "Unknown Diagnosis",
        "Diagnosis_Code": "Unknown",
        "Discharge_Date": pd.NA
    }

    for column, default in defaults.items():
        if column not in data.columns:
            data[column] = default

    data["Length_of_Stay"] = parse_numeric(data["Length_of_Stay"])
    data["Total_Charges"] = parse_numeric(data["Total_Charges"])

    for column in [
        "Facility_Name",
        "County",
        "Severity_Level",
        "Payment_Type",
        "Age_Group",
        "Race",
        "Diagnosis_Category",
        "Diagnosis_Code"
    ]:
        data[column] = clean_category(data[column])

    data["Gender"] = normalize_gender(data["Gender"])

    data["Severity_Level"] = (
        data["Severity_Level"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    data["Payment_Type"] = (
        data["Payment_Type"]
        .replace({
            "Private Health Insurance": "Private Insurance",
            "Blue Cross/Blue Shield": "Blue Cross/Blue Shield",
            "Self Pay": "Self-Pay"
        })
    )

    data["Discharge_Date"] = pd.to_datetime(
        data["Discharge_Date"],
        errors="coerce"
    )

    return data


@st.cache_data(show_spinner=False, max_entries=4)
def load_sparcs_data(file_path, sample_limit):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    parts = []
    rows_read = 0
    columns = None

    with open(file_path, "rb") as file_handle:
        for chunk in pd.read_csv(
            file_handle,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ):
            if columns is None:
                columns = list(chunk.columns)

            remaining = sample_limit - rows_read
            if remaining <= 0:
                break

            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()

            parts.append(chunk)
            rows_read += len(chunk)

            if rows_read >= sample_limit:
                break

    if not parts:
        raise ValueError("The CSV file contains no data rows.")

    raw = pd.concat(parts, ignore_index=True)
    original_rows_loaded = len(raw)

    normalized = standardize_columns(raw)

    invalid_los = (
        normalized["Length_of_Stay"].isna()
        | (normalized["Length_of_Stay"] <= 0)
    )
    invalid_los_count = int(invalid_los.sum())

    cleaned = normalized.loc[~invalid_los].copy()

    possible_duplicate_count = int(
        cleaned.duplicated().sum()
    )

    valid_date_count = int(
        cleaned["Discharge_Date"].notna().sum()
    )

    outside_2021_count = 0
    if valid_date_count > 0:
        outside_2021_count = int(
            (
                cleaned["Discharge_Date"].notna()
                & (cleaned["Discharge_Date"].dt.year != 2021)
            ).sum()
        )

    missing_charge_count = int(
        cleaned["Total_Charges"].isna().sum()
    )

    valid_charges = cleaned["Total_Charges"].dropna()
    charge_outlier_count = 0

    if len(valid_charges) >= 4:
        q1 = valid_charges.quantile(0.25)
        q3 = valid_charges.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        charge_outlier_count = int(
            (
                (cleaned["Total_Charges"] < lower_bound)
                | (cleaned["Total_Charges"] > upper_bound)
            ).sum()
        )

    los_outlier_count = 0
    if len(cleaned) >= 4:
        q1_los = cleaned["Length_of_Stay"].quantile(0.25)
        q3_los = cleaned["Length_of_Stay"].quantile(0.75)
        iqr_los = q3_los - q1_los
        los_lower = q1_los - 1.5 * iqr_los
        los_upper = q3_los + 1.5 * iqr_los

        los_outlier_count = int(
            (
                (cleaned["Length_of_Stay"] < los_lower)
                | (cleaned["Length_of_Stay"] > los_upper)
            ).sum()
        )

    metadata = {
        "rows_loaded": original_rows_loaded,
        "rows_after_cleaning": len(cleaned),
        "invalid_stays_removed": invalid_los_count,
        "potential_duplicates": possible_duplicate_count,
        "missing_charges": missing_charge_count,
        "charge_outliers": charge_outlier_count,
        "los_outliers": los_outlier_count,
        "valid_dates": valid_date_count,
        "dates_outside_2021": outside_2021_count,
        "columns_found": len(columns or [])
    }

    return cleaned, metadata


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def style_chart(fig, height=420):
    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(9,16,31,0.55)",
        font=dict(
            family="Inter, sans-serif",
            color="#dce9ff"
        ),
        margin=dict(l=15, r=15, t=55, b=20),
        legend_title_text=""
    )
    return fig


def minimum_group_summary(
    data,
    group_column,
    min_discharges,
    value_column="Length_of_Stay"
):
    result = (
        data.groupby(group_column, dropna=False)
        .agg(
            Discharges=(value_column, "size"),
            Average_Stay=(value_column, "mean"),
            Median_Stay=(value_column, "median")
        )
        .reset_index()
    )

    return result[result["Discharges"] >= min_discharges]


st.markdown("""
<div class="hero">
    <div class="hero-title">MediScope Health Analytics</div>
    <div class="hero-subtitle">
        SPARCS 2021 Hospital Inpatient Discharges · Interactive
        healthcare utilization, length-of-stay and charge analysis
    </div>
</div>
""", unsafe_allow_html=True)

if not os.path.exists(TARGET_FILE):
    st.error(
        f"Dataset not found. Place `{TARGET_FILE}` in the same "
        "folder as app.py."
    )
    st.stop()

st.sidebar.header("Dashboard Controls")

sample_limit = st.sidebar.selectbox(
    "Records to analyze",
    options=[10000, 25000, 50000, 100000],
    index=2,
    format_func=lambda x: f"{x:,} records"
)

st.sidebar.caption(
    "The dashboard reads the CSV in chunks and analyzes the first "
    "selected number of rows. Choose a larger sample to broaden "
    "the analysis."
)

if st.sidebar.button("Clear cached data"):
    st.cache_data.clear()
    st.rerun()

try:
    with st.spinner("Reading dataset in chunks and cleaning records..."):
        df_raw, quality = load_sparcs_data(
            TARGET_FILE,
            sample_limit
        )
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

if df_raw.empty:
    st.error(
        "No records remain after cleaning. Check the dataset's "
        "length-of-stay column and its values."
    )
    st.stop()

st.sidebar.success(
    f"Analyzed {len(df_raw):,} cleaned records"
)

st.sidebar.subheader("Filters")

county_options = sorted(df_raw["County"].unique().tolist())
severity_options = sorted(df_raw["Severity_Level"].unique().tolist())
payment_options = sorted(df_raw["Payment_Type"].unique().tolist())
age_options = sorted(df_raw["Age_Group"].unique().tolist())
gender_options = sorted(df_raw["Gender"].unique().tolist())
race_options = sorted(df_raw["Race"].unique().tolist())

selected_counties = st.sidebar.multiselect(
    "County",
    county_options,
    default=county_options
)

selected_severity = st.sidebar.multiselect(
    "Severity",
    severity_options,
    default=severity_options
)

selected_payment = st.sidebar.multiselect(
    "Payment type",
    payment_options,
    default=payment_options
)

selected_age = st.sidebar.multiselect(
    "Age group",
    age_options,
    default=age_options
)

selected_gender = st.sidebar.multiselect(
    "Gender",
    gender_options,
    default=gender_options
)

selected_race = st.sidebar.multiselect(
    "Race",
    race_options,
    default=race_options
)

minimum_discharges = st.sidebar.number_input(
    "Minimum discharges per ranking group",
    min_value=1,
    max_value=1000,
    value=10,
    step=1
)

hide_charge_outliers = st.sidebar.checkbox(
    "Hide extreme charge values in box plot",
    value=False
)

df = df_raw[
    df_raw["County"].isin(selected_counties)
    & df_raw["Severity_Level"].isin(selected_severity)
    & df_raw["Payment_Type"].isin(selected_payment)
    & df_raw["Age_Group"].isin(selected_age)
    & df_raw["Gender"].isin(selected_gender)
    & df_raw["Race"].isin(selected_race)
].copy()

if df.empty:
    st.warning(
        "No records match the current filters. Change or clear "
        "some filters in the sidebar."
    )
    st.stop()

valid_charges = df["Total_Charges"].dropna()
valid_charge_count = len(valid_charges)
charge_total = valid_charges.sum() if valid_charge_count else np.nan
charge_average = valid_charges.mean() if valid_charge_count else np.nan

k1, k2, k3, k4 = st.columns(4)

with k1:
    metric_card("Discharges analyzed", f"{len(df):,}")

with k2:
    metric_card(
        "Average length of stay",
        f"{df['Length_of_Stay'].mean():.2f} days"
    )

with k3:
    metric_card(
        "Total recorded charges",
        f"${charge_total:,.0f}" if valid_charge_count else "N/A",
        f"{valid_charge_count:,} records with valid charges"
    )

with k4:
    metric_card(
        "Average charge per stay",
        f"${charge_average:,.2f}" if valid_charge_count else "N/A",
        f"Based on {valid_charge_count:,} valid charge records"
    )

st.caption(
    "These figures describe the analyzed records after cleaning "
    "and filtering; they are not automatically statewide totals."
)

tab_overview, tab_diagnosis, tab_finance, tab_facilities, tab_quality, tab_ai, tab_data = st.tabs([
    "Overview",
    "Diagnosis & Stay",
    "Charges & Demographics",
    "Facilities & Counties",
    "Data Quality",
    "AI Experiment",
    "Data Explorer"
])

with tab_overview:
    st.subheader("Hospital discharge overview")

    left, right = st.columns(2)

    with left:
        payment_counts = (
            df["Payment_Type"]
            .value_counts()
            .rename_axis("Payment Type")
            .reset_index(name="Discharges")
        )

        fig = px.pie(
            payment_counts,
            names="Payment Type",
            values="Discharges",
            hole=0.48,
            title="Discharges by payment type"
        )
        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    with right:
        fig = px.histogram(
            df,
            x="Length_of_Stay",
            nbins=35,
            title="Distribution of length of stay",
            labels={"Length_of_Stay": "Length of stay (days)"}
        )
        fig.update_traces(
            marker_line_width=0.4,
            marker_line_color="#a8eaff"
        )
        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    age_summary = (
        df.groupby("Age_Group", dropna=False)
        .agg(
            Discharges=("Length_of_Stay", "size"),
            Average_Stay=("Length_of_Stay", "mean")
        )
        .reset_index()
        .sort_values("Discharges", ascending=False)
    )

    fig = px.bar(
        age_summary,
        x="Age_Group",
        y="Discharges",
        color="Average_Stay",
        title="Discharges and average stay by age group",
        labels={
            "Age_Group": "Age group",
            "Discharges": "Discharges",
            "Average_Stay": "Average stay (days)"
        },
        color_continuous_scale="Tealgrn"
    )
    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )

    st.subheader("Research questions")

    research_questions = [
        "Which diagnosis categories have the longest average stays?",
        "How does severity of illness relate to length of stay?",
        "How do recorded hospital charges vary by severity?",
        "How do average charges differ by age group and gender?",
        "Which facilities have the largest discharge volumes?",
        "How do payment types differ across counties?",
        "How do demographic groups vary in discharge volume and length of stay?"
    ]

    for question in research_questions:
        st.markdown(f"- {question}")

with tab_diagnosis:
    st.subheader("Length of stay by diagnosis and severity")

    diagnosis_summary = (
        df.groupby("Diagnosis_Category")
        .agg(
            Discharges=("Length_of_Stay", "size"),
            Average_Stay=("Length_of_Stay", "mean"),
            Median_Stay=("Length_of_Stay", "median")
        )
        .reset_index()
    )

    diagnosis_summary = diagnosis_summary[
        diagnosis_summary["Discharges"] >= minimum_discharges
    ]

    if not diagnosis_summary.empty:
        diagnosis_summary = diagnosis_summary.nlargest(
            15, "Average_Stay"
        ).sort_values("Average_Stay")

        fig = px.bar(
            diagnosis_summary,
            x="Average_Stay",
            y="Diagnosis_Category",
            orientation="h",
            color="Discharges",
            title=(
                "Diagnosis categories with highest average stay "
                f"(minimum {minimum_discharges} discharges)"
            ),
            labels={
                "Average_Stay": "Average stay (days)",
                "Diagnosis_Category": "Diagnosis category",
                "Discharges": "Discharges"
            },
            hover_data=["Median_Stay"]
        )
        st.plotly_chart(
            style_chart(fig, 520),
            use_container_width=True
        )
    else:
        st.info(
            "No diagnosis group meets the current minimum-discharge "
            "threshold. Reduce the threshold in the sidebar."
        )

    severity_summary = (
        df.groupby("Severity_Level")
        .agg(
            Discharges=("Length_of_Stay", "size"),
            Average_Stay=("Length_of_Stay", "mean"),
            Median_Stay=("Length_of_Stay", "median")
        )
        .reset_index()
        .sort_values("Average_Stay", ascending=False)
    )

    fig = px.bar(
        severity_summary,
        x="Severity_Level",
        y="Average_Stay",
        color="Severity_Level",
        title="Average stay by severity of illness",
        labels={
            "Severity_Level": "Severity",
            "Average_Stay": "Average stay (days)"
        },
        hover_data=["Discharges", "Median_Stay"]
    )
    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )

    severity_table = severity_summary.rename(columns={
        "Severity_Level": "Severity",
        "Average_Stay": "Average stay (days)",
        "Median_Stay": "Median stay (days)"
    })
    st.dataframe(
        severity_table,
        use_container_width=True,
        hide_index=True
    )

with tab_finance:
    st.subheader("Charges and demographic comparisons")

    charge_data = df.dropna(subset=["Total_Charges"]).copy()

    if hide_charge_outliers and len(charge_data) >= 4:
        q1 = charge_data["Total_Charges"].quantile(0.25)
        q3 = charge_data["Total_Charges"].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        charge_data = charge_data[
            charge_data["Total_Charges"].between(low, high)
        ]

    st.caption(
        "Missing charges are excluded from charge charts and averages. "
        "Outlier filtering changes only the box plot display."
    )

    fig = px.box(
        charge_data,
        x="Severity_Level",
        y="Total_Charges",
        color="Severity_Level",
        points="outliers" if not hide_charge_outliers else False,
        title="Recorded charges by severity",
        labels={
            "Severity_Level": "Severity",
            "Total_Charges": "Total charges ($)"
        }
    )
    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )

    charge_col1, charge_col2 = st.columns(2)

    with charge_col1:
        age_charge = (
            charge_data.groupby("Age_Group")
            .agg(
                Valid_Charge_Records=("Total_Charges", "count"),
                Average_Charge=("Total_Charges", "mean")
            )
            .reset_index()
            .sort_values("Average_Charge", ascending=False)
        )

        fig = px.bar(
            age_charge,
            x="Age_Group",
            y="Average_Charge",
            title="Average charges by age group",
            labels={
                "Age_Group": "Age group",
                "Average_Charge": "Average charge ($)"
            },
            hover_data=["Valid_Charge_Records"]
        )
        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    with charge_col2:
        gender_charge = (
            charge_data.groupby("Gender")
            .agg(
                Valid_Charge_Records=("Total_Charges", "count"),
                Average_Charge=("Total_Charges", "mean")
            )
            .reset_index()
        )

        fig = px.bar(
            gender_charge,
            x="Gender",
            y="Average_Charge",
            color="Gender",
            title="Average charges by gender",
            labels={
                "Gender": "Gender",
                "Average_Charge": "Average charge ($)"
            },
            hover_data=["Valid_Charge_Records"]
        )
        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    race_severity = (
        df.groupby(["Race", "Severity_Level"])
        .size()
        .reset_index(name="Discharges")
    )

    fig = px.bar(
        race_severity,
        x="Race",
        y="Discharges",
        color="Severity_Level",
        barmode="group",
        title="Discharges by race and severity",
        labels={"Race": "Race", "Discharges": "Discharges"}
    )
    st.plotly_chart(
        style_chart(fig, 480),
        use_container_width=True
    )

    st.subheader("Charge coverage")
    st.write(f"Filtered records: **{len(df):,}**")
    st.write(
        f"Records with valid charges: **{df['Total_Charges'].notna().sum():,}**"
    )
    st.write(
        f"Records with missing or invalid charges: "
        f"**{df['Total_Charges'].isna().sum():,}**"
    )

with tab_facilities:
    st.subheader("Facility utilization and county comparison")

    facility_summary = (
        df.groupby(["Facility_Name", "County"])
        .agg(
            Discharges=("Length_of_Stay", "size"),
            Average_Stay=("Length_of_Stay", "mean"),
            Median_Stay=("Length_of_Stay", "median")
        )
        .reset_index()
    )

    facility_summary = facility_summary[
        facility_summary["Discharges"] >= minimum_discharges
    ]

    top_facilities = facility_summary.nlargest(
        15, "Discharges"
    )

    if not top_facilities.empty:
        fig = px.bar(
            top_facilities.sort_values("Discharges"),
            x="Discharges",
            y="Facility_Name",
            color="County",
            orientation="h",
            title="Highest-volume facilities",
            labels={
                "Discharges": "Discharges",
                "Facility_Name": "Facility"
            },
            hover_data=["Average_Stay", "Median_Stay"]
        )
        st.plotly_chart(
            style_chart(fig, 520),
            use_container_width=True
        )

        facility_options = top_facilities[
            ["Facility_Name", "County"]
        ].drop_duplicates()

        facility_options["Display"] = (
            facility_options["Facility_Name"]
            + " — "
            + facility_options["County"]
        )

        selected_facilities = st.multiselect(
            "Facilities to include in the heatmap",
            options=facility_options["Display"].tolist(),
            default=facility_options["Display"].head(8).tolist(),
            key="facility_heatmap_selection"
        )

        selected_facility_rows = facility_options[
            facility_options["Display"].isin(selected_facilities)
        ]

        heatmap_source = df.merge(
            selected_facility_rows[["Facility_Name", "County"]],
            on=["Facility_Name", "County"],
            how="inner"
        )

        if not heatmap_source.empty:
            heatmap_data = heatmap_source.pivot_table(
                index="Facility_Name",
                columns="County",
                values="Length_of_Stay",
                aggfunc="mean"
            )

            if not heatmap_data.empty:
                fig = px.imshow(
                    heatmap_data,
                    aspect="auto",
                    color_continuous_scale="Viridis",
                    title="Average stay by facility and county",
                    labels={
                        "x": "County",
                        "y": "Facility",
                        "color": "Average stay (days)"
                    }
                )
                st.plotly_chart(
                    style_chart(fig, 500),
                    use_container_width=True
                )
    else:
        st.info(
            "No facility meets the minimum-discharge threshold."
        )

    county_summary = (
        df.groupby("County")
        .agg(
            Discharges=("Length_of_Stay", "size"),
            Average_Stay=("Length_of_Stay", "mean"),
            Median_Stay=("Length_of_Stay", "median")
        )
        .reset_index()
    )

    county_summary = county_summary[
        county_summary["Discharges"] >= minimum_discharges
    ]

    if not county_summary.empty:
        county_summary = county_summary.sort_values(
            "Average_Stay", ascending=False
        )

        fig = px.bar(
            county_summary,
            x="County",
            y="Average_Stay",
            color="Discharges",
            title="Average stay by county",
            labels={
                "County": "County",
                "Average_Stay": "Average stay (days)"
            },
            hover_data=["Discharges", "Median_Stay"]
        )
        st.plotly_chart(
            style_chart(fig, 480),
            use_container_width=True
        )

    st.subheader("County and payment type")

    county_payment = pd.crosstab(
        df["County"],
        df["Payment_Type"]
    )

    st.dataframe(
        county_payment,
        use_container_width=True
    )

    fig = px.bar(
        county_payment.reset_index().melt(
            id_vars="County",
            var_name="Payment_Type",
            value_name="Discharges"
        ),
        x="County",
        y="Discharges",
        color="Payment_Type",
        barmode="stack",
        title="Discharges by county and payment type",
        labels={
            "County": "County",
            "Payment_Type": "Payment type"
        }
    )
    st.plotly_chart(
        style_chart(fig, 480),
        use_container_width=True
    )

with tab_quality:
    st.subheader("Data quality and preprocessing report")

    q1, q2, q3 = st.columns(3)

    with q1:
        metric_card(
            "Rows loaded",
            f"{quality['rows_loaded']:,}"
        )

    with q2:
        metric_card(
            "Rows after valid-stay check",
            f"{quality['rows_after_cleaning']:,}"
        )

    with q3:
        metric_card(
            "Potential duplicate rows",
            f"{quality['potential_duplicates']:,}",
            "Flagged for review, not automatically removed"
        )

    quality_table = pd.DataFrame({
        "Check": [
            "Records read",
            "Records remaining after cleaning",
            "Invalid or missing length of stay",
            "Potential exact duplicate rows",
            "Missing or invalid charges",
            "Charge outliers detected by IQR",
            "Length-of-stay outliers detected by IQR",
            "Records with valid discharge dates",
            "Valid dates outside 2021",
            "Original columns detected"
        ],
        "Result": [
            quality["rows_loaded"],
            quality["rows_after_cleaning"],
            quality["invalid_stays_removed"],
            quality["potential_duplicates"],
            quality["missing_charges"],
            quality["charge_outliers"],
            quality["los_outliers"],
            quality["valid_dates"],
            quality["dates_outside_2021"],
            quality["columns_found"]
        ]
    })

    st.dataframe(
        quality_table,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("""
    <div class="section-card">
    <strong>Cleaning decisions</strong>
    <ul>
    <li>Length of stay is converted to numeric values.</li>
    <li>Rows with missing, invalid or non-positive length of stay are excluded.</li>
    <li>Charges are converted to numeric values; missing charges are not imputed.</li>
    <li>Gender and categorical fields are standardized where possible.</li>
    <li>Potential exact duplicates are reported, not automatically deleted.</li>
    <li>Charge and stay outliers are flagged using the IQR method.</li>
    <li>Discharge dates are checked when the source contains a usable date column.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    if quality["valid_dates"] == 0:
        st.warning(
            "No valid discharge dates were found in the analyzed sample. "
            "The year of every record therefore could not be independently verified."
        )
    elif quality["dates_outside_2021"] > 0:
        st.warning(
            f"{quality['dates_outside_2021']:,} records with valid dates "
            "fall outside 2021. Investigate before treating the sample as "
            "strictly limited to the 2021 reporting period."
        )
    else:
        st.success(
            "All valid discharge dates in the analyzed sample fall in 2021."
        )

    st.info(
        "This report describes the analyzed sample, not necessarily the "
        "entire CSV. Exact duplicates can represent legitimate discharges; "
        "confirm the source's record-identification rules before removing them."
    )

with tab_ai:
    st.subheader("AI experiment: predicting a longer hospital stay")

    st.write(
        "This optional Random Forest experiment predicts whether a stay is "
        "longer than the median stay in the currently analyzed sample. "
        "It is a learning demonstration, not a clinical decision tool."
    )

    st.warning(
        "This target is defined from the observed length of stay. The model "
        "does not predict an actual future discharge date, treatment need, "
        "or medical outcome. Results depend on the sample and available fields."
    )

    run_model = st.button(
        "Train and evaluate model",
        key="train_ai_model"
    )

    if run_model:
        model_data = df.copy()
        model_data = model_data[
            model_data["Length_of_Stay"].notna()
            & (model_data["Length_of_Stay"] > 0)
        ]

        model_data["Long_Stay"] = (
            model_data["Length_of_Stay"]
            > model_data["Length_of_Stay"].median()
        ).astype(int)

        numeric_features = []
        categorical_features = [
            "Age_Group",
            "Gender",
            "Race",
            "Severity_Level",
            "Diagnosis_Category",
            "County",
            "Payment_Type"
        ]

        for column in categorical_features:
            if column not in model_data.columns:
                model_data[column] = "Unknown"

        feature_columns = numeric_features + categorical_features
        X = model_data[feature_columns].copy()
        y = model_data["Long_Stay"]

        if len(model_data) < 100 or y.nunique() < 2:
            st.error(
                "There are not enough records or target classes to evaluate "
                "the model reliably. Increase the sample size and check the data."
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.25,
                random_state=42,
                stratify=y
            )

            preprocessing = ColumnTransformer(
                transformers=[
                    (
                        "categorical",
                        Pipeline(steps=[
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="most_frequent"
                                )
                            ),
                            (
                                "encoder",
                                OneHotEncoder(
                                    handle_unknown="ignore"
                                )
                            )
                        ]),
                        categorical_features
                    )
                ],
                remainder="drop"
            )

            model = Pipeline(steps=[
                ("preprocessing", preprocessing),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=120,
                        max_depth=12,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1
                    )
                )
            ])

            try:
                with st.spinner("Training and evaluating the model..."):
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)

                ai1, ai2, ai3, ai4 = st.columns(4)

                with ai1:
                    metric_card(
                        "Accuracy",
                        f"{accuracy_score(y_test, predictions):.3f}"
                    )

                with ai2:
                    metric_card(
                        "Precision",
                        f"{precision_score(y_test, predictions, zero_division=0):.3f}"
                    )

                with ai3:
                    metric_card(
                        "Recall",
                        f"{recall_score(y_test, predictions, zero_division=0):.3f}"
                    )

                with ai4:
                    metric_card(
                        "F1 score",
                        f"{f1_score(y_test, predictions, zero_division=0):.3f}"
                    )

                matrix = confusion_matrix(
                    y_test,
                    predictions,
                    labels=[0, 1]
                )

                matrix_frame = pd.DataFrame(
                    matrix,
                    index=["Actual not-long stay", "Actual long stay"],
                    columns=["Predicted not-long stay", "Predicted long stay"]
                )

                fig = px.imshow(
                    matrix_frame,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="Blues",
                    title="Test-set confusion matrix"
                )

                st.plotly_chart(
                    style_chart(fig, 380),
                    use_container_width=True
                )

                st.caption(
                    f"Training records: {len(X_train):,}. "
                    f"Test records: {len(X_test):,}. "
                    "The test set was held out from model fitting."
                )

                st.write(
                    "Compare the scores with a simple baseline before "
                    "claiming that the model is useful. This experiment "
                    "uses categorical fields and excludes length of stay "
                    "from the predictor features to avoid direct target leakage."
                )

            except Exception as error:
                st.error(
                    f"The model could not be trained: {error}"
                )

with tab_data:
    st.subheader("Filtered data explorer")

    st.write(
        f"Displaying {min(len(df), 5000):,} of {len(df):,} filtered records."
    )

    display_columns = [
        "Facility_Name",
        "County",
        "Diagnosis_Category",
        "Diagnosis_Code",
        "Severity_Level",
        "Age_Group",
        "Gender",
        "Race",
        "Payment_Type",
        "Length_of_Stay",
        "Total_Charges",
        "Discharge_Date"
    ]

    display_columns = [
        column for column in display_columns
        if column in df.columns
    ]

    st.dataframe(
        df[display_columns].head(5000),
        use_container_width=True,
        hide_index=True
    )

    csv_bytes = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download filtered records as CSV",
        data=csv_bytes,
        file_name="mediscope_filtered_discharges.csv",
        mime="text/csv"
    )

    summary_csv = df.groupby(
        ["County", "Severity_Level", "Payment_Type"],
        dropna=False
    ).agg(
        Discharges=("Length_of_Stay", "size"),
        Average_Stay=("Length_of_Stay", "mean"),
        Valid_Charge_Records=("Total_Charges", "count"),
        Average_Charge=("Total_Charges", "mean")
    ).reset_index()

    st.download_button(
        label="Download aggregated summary as CSV",
        data=summary_csv.to_csv(index=False).encode("utf-8"),
        file_name="mediscope_summary.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption(
    "MediScope Health Analytics · SPARCS 2021 · Educational analytics "
    "project. Findings are descriptive and do not establish causation."
)
