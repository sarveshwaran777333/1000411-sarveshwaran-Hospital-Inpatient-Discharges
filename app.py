import os
import streamlit as st
import pandas as pd
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MediScope | Health Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

TARGET_FILE = (
    "Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv"
)


# ============================================================
# NEON ANIMATED THEME
# ============================================================

st.markdown(
    """
    <style>
        :root {
            --neon-cyan: #00f5ff;
            --neon-purple: #bc13fe;
            --neon-pink: #ff2bd6;
            --dark-bg: #0b0f19;
        }

        .stApp {
            background:
                radial-gradient(
                    circle at 10% 10%,
                    rgba(0,245,255,.07),
                    transparent 35%
                ),
                radial-gradient(
                    circle at 90% 80%,
                    rgba(188,19,254,.08),
                    transparent 35%
                ),
                var(--dark-bg);
            color: #f1f5f9;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        .dashboard-header {
            position: relative;
            overflow: hidden;
            padding: 1.8rem;
            border-radius: 18px;
            background: linear-gradient(
                120deg,
                #071a33,
                #064e69,
                #321052,
                #071a33
            );
            background-size: 300% 300%;
            border: 1px solid var(--neon-cyan);
            animation:
                gradientMove 9s ease infinite,
                neonPulse 3s ease-in-out infinite;
            margin-bottom: 1.8rem;
        }

        .dashboard-header h1 {
            color: white;
            margin: 0;
            font-size: 2.2rem;
            text-shadow:
                0 0 8px rgba(0,245,255,.9),
                0 0 20px rgba(0,245,255,.5);
        }

        .dashboard-header p {
            color: #e0faff;
            margin-top: .7rem;
            margin-bottom: 0;
        }

        .dashboard-header::after {
            content: "";
            position: absolute;
            top: 0;
            left: -150%;
            width: 70%;
            height: 100%;
            background: linear-gradient(
                100deg,
                transparent,
                rgba(255,255,255,.13),
                transparent
            );
            transform: skewX(-20deg);
            animation: lightSweep 7s ease-in-out infinite;
            pointer-events: none;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(
                145deg,
                #172238,
                #101827
            );
            color: white;
            padding: 1.1rem;
            border-radius: 14px;
            border: 1px solid rgba(0,245,255,.55);
            box-shadow: 0 0 7px rgba(0,245,255,.16);
            transition: all .3s ease;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-5px);
            border-color: var(--neon-cyan);
            box-shadow:
                0 0 12px rgba(0,245,255,.65),
                0 0 28px rgba(0,245,255,.22);
        }

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] label p,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] [data-testid="stMetricValue"],
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            color: #f1fbff !important;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            text-shadow: 0 0 9px rgba(0,245,255,.35);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #111827,
                #15112b,
                #101827
            );
            border-right: 1px solid rgba(188,19,254,.45);
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: white;
            text-shadow: 0 0 8px rgba(188,19,254,.5);
        }

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label {
            color: #e2e8f0;
        }

        div[data-testid="stPlotlyChart"] {
            border: 1px solid rgba(0,245,255,.23);
            border-radius: 13px;
            padding: .35rem;
            background: rgba(16,24,39,.5);
            transition: all .3s ease;
        }

        div[data-testid="stPlotlyChart"]:hover {
            border-color: rgba(0,245,255,.75);
            box-shadow: 0 0 15px rgba(0,245,255,.12);
        }

        div[data-baseweb="select"] > div {
            border-color: rgba(0,245,255,.35);
        }

        div.stButton > button,
        div.stDownloadButton > button {
            color: white;
            background: linear-gradient(100deg,#073b4c,#321052);
            border: 1px solid var(--neon-cyan);
            border-radius: 9px;
            transition: all .25s ease;
        }

        div.stButton > button:hover,
        div.stDownloadButton > button:hover {
            color: white;
            border-color: var(--neon-pink);
            box-shadow:
                0 0 10px rgba(0,245,255,.45),
                0 0 18px rgba(188,19,254,.25);
            transform: translateY(-2px);
        }

        div[data-testid="stExpander"] {
            border: 1px solid rgba(0,245,255,.25);
            border-radius: 12px;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(0,245,255,.25);
            border-radius: 10px;
        }

        h1, h2, h3 {
            color: #e8faff;
        }

        hr {
            border-color: rgba(0,245,255,.25);
        }

        @keyframes gradientMove {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        @keyframes neonPulse {
            0%,100% {
                box-shadow: 0 0 8px rgba(0,245,255,.22);
            }
            50% {
                box-shadow:
                    0 0 15px rgba(0,245,255,.42),
                    0 0 30px rgba(188,19,254,.16);
            }
        }

        @keyframes lightSweep {
            0%,65% { left: -150%; }
            100% { left: 180%; }
        }

        @media (prefers-reduced-motion: reduce) {
            *,
            *::before,
            *::after {
                animation-duration: .01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: .01ms !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# COLUMN DETECTION HELPERS
# ============================================================

def normalize_column_name(column):
    """Normalize a column label for robust matching."""
    return (
        str(column)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def find_column(dataframe, possible_names):
    """
    Find a source column using exact matches first,
    then carefully ordered partial matches.
    """

    columns = {
        normalize_column_name(column): column
        for column in dataframe.columns
    }

    normalized_candidates = [
        candidate.lower().strip()
        for candidate in possible_names
    ]

    for candidate in normalized_candidates:
        if candidate in columns:
            return columns[candidate]

    for candidate in normalized_candidates:
        for normalized, original in columns.items():
            if candidate in normalized:
                return original

    return None


def first_available_column(dataframe, candidates):
    """Return the first available source column from a candidate list."""
    for candidate in candidates:
        found = find_column(dataframe, [candidate])
        if found is not None:
            return found
    return None


def standardize_text(series, default="Unknown"):
    """Clean text values and replace missing/blank values."""
    return (
        series.astype("string")
        .str.strip()
        .replace("", pd.NA)
        .fillna(default)
        .astype(str)
    )


# ============================================================
# DATA LOADING AND PREPROCESSING
# ============================================================

@st.cache_data(show_spinner=False)
def load_sparcs_data(file_path, max_rows=50000):
    """
    Load and preprocess the fixed SPARCS 2021 dataset.

    max_rows=None loads the entire CSV.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    data = pd.read_csv(
        file_path,
        nrows=max_rows,
        low_memory=False
    )

    # Remove duplicate original column labels.
    data = data.loc[:, ~data.columns.duplicated()].copy()

    original_columns = list(data.columns)

    # Identify diagnosis code and description fields before renaming.
    diagnosis_code_source = first_available_column(
        data,
        [
            "APR MDC Code",
            "APR DRG Code",
            "CCS Diagnosis Code",
            "Diagnosis Code",
            "ICD Diagnosis Code"
        ]
    )

    diagnosis_description_source = first_available_column(
        data,
        [
            "APR MDC Description",
            "APR DRG Description",
            "CCS Diagnosis Description",
            "Diagnosis Description",
            "MDC Description"
        ]
    )

    # Identify demographic and clinical source fields.
    source_fields = {
        "Length_of_Stay": [
            "Length of Stay"
        ],
        "Total_Charges": [
            "Total Charges",
            "Total Charge"
        ],
        "Facility_Name": [
            "Facility Name"
        ],
        "County": [
            "Hospital County",
            "Facility County",
            "County"
        ],
        "Severity_Level": [
            "APR Severity of Illness Description",
            "Severity of Illness Description",
            "Severity of Illness"
        ],
        "Payment_Type": [
            "Payment Typology 1",
            "Payment Type"
        ],
        "Age_Group": [
            "Age Group"
        ],
        "Gender": [
            "Gender"
        ],
        "Race": [
            "Race"
        ],
        "Region": [
            "Hospital Region",
            "Facility Region",
            "Region"
        ]
    }

    rename_mapping = {}

    for standard_name, candidates in source_fields.items():
        source_column = find_column(data, candidates)

        if source_column is not None:
            # Avoid renaming the same source field twice.
            if source_column not in rename_mapping:
                rename_mapping[source_column] = standard_name

    data = data.rename(columns=rename_mapping)

    # Create a diagnosis label using a readable description when possible.
    if diagnosis_description_source is not None:
        if diagnosis_description_source in data.columns:
            data["Diagnosis_Category"] = standardize_text(
                data[diagnosis_description_source],
                "Unknown Diagnosis"
            )
        else:
            data["Diagnosis_Category"] = "Unknown Diagnosis"

    elif diagnosis_code_source is not None:
        if diagnosis_code_source in data.columns:
            data["Diagnosis_Category"] = (
                "Diagnosis Code: "
                + standardize_text(
                    data[diagnosis_code_source],
                    "Unknown"
                )
            )
        else:
            data["Diagnosis_Category"] = "Unknown Diagnosis"

    else:
        data["Diagnosis_Category"] = "Unknown Diagnosis"

    # Keep a diagnosis code field for analysis where available.
    if diagnosis_code_source is not None:
        if diagnosis_code_source in data.columns:
            data["Diagnosis_Code"] = standardize_text(
                data[diagnosis_code_source],
                "Unknown Code"
            )
        else:
            data["Diagnosis_Code"] = "Unknown Code"
    else:
        data["Diagnosis_Code"] = "Not Available"

    # Length of stay is required for this dashboard.
    if "Length_of_Stay" not in data.columns:
        raise ValueError(
            "Could not find the required Length of Stay column. "
            "Check the SPARCS CSV column names."
        )

    # Convert length of stay to numeric.
    data["Length_of_Stay"] = (
        data["Length_of_Stay"]
        .astype(str)
        .str.replace("+", "", regex=False)
        .str.strip()
    )

    data["Length_of_Stay"] = pd.to_numeric(
        data["Length_of_Stay"],
        errors="coerce"
    )

    # Remove missing, invalid, and non-positive stays.
    data = data[
        data["Length_of_Stay"].notna()
        & (data["Length_of_Stay"] > 0)
    ].copy()

    # Convert charges to numeric values.
    if "Total_Charges" in data.columns:
        data["Total_Charges"] = (
            data["Total_Charges"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        data["Total_Charges"] = pd.to_numeric(
            data["Total_Charges"],
            errors="coerce"
        )
    else:
        data["Total_Charges"] = float("nan")

    # Ensure all analysis fields exist.
    fallback_columns = {
        "Facility_Name": "Unknown Facility",
        "County": "Unknown County",
        "Region": "Unknown Region",
        "Severity_Level": "Unknown Severity",
        "Payment_Type": "Unknown Payment Type",
        "Age_Group": "Unknown Age Group",
        "Gender": "Unknown",
        "Race": "Unknown Race"
    }

    for column, default in fallback_columns.items():
        if column not in data.columns:
            data[column] = default
        else:
            data[column] = standardize_text(
                data[column],
                default
            )

    # Standardize gender labels.
    gender_map = {
        "M": "Male",
        "F": "Female",
        "MALE": "Male",
        "FEMALE": "Female",
        "U": "Unknown",
        "UNKNOWN": "Unknown"
    }

    data["Gender"] = (
        data["Gender"]
        .str.strip()
        .str.upper()
        .map(gender_map)
        .fillna(data["Gender"].str.title())
    )

    # Standardize race labels.
    data["Race"] = (
        data["Race"]
        .str.strip()
        .str.title()
        .replace({
            "Black/African American": "Black",
            "White/Caucasian": "White"
        })
    )

    # Standardize severity labels.
    data["Severity_Level"] = (
        data["Severity_Level"]
        .str.strip()
        .str.title()
    )

    # Standardize payment labels.
    data["Payment_Type"] = (
        data["Payment_Type"]
        .str.strip()
        .str.title()
    )

    # Standardize age groups without inventing age categories.
    data["Age_Group"] = (
        data["Age_Group"]
        .str.strip()
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
    )

    # Remove duplicate records.
    data = data.drop_duplicates().copy()

    # Remove completely empty and overwhelmingly null columns,
    # but preserve all standardized fields needed by the dashboard.
    protected_columns = {
        "Length_of_Stay",
        "Total_Charges",
        "Facility_Name",
        "County",
        "Region",
        "Severity_Level",
        "Payment_Type",
        "Age_Group",
        "Gender",
        "Race",
        "Diagnosis_Category",
        "Diagnosis_Code"
    }

    columns_to_drop = []

    for column in data.columns:
        if column in protected_columns:
            continue

        null_fraction = data[column].isna().mean()

        if null_fraction >= 0.90:
            columns_to_drop.append(column)

    # Explicitly remove birth-weight fields when present:
    # they are not relevant to this general discharge analysis.
    for column in data.columns:
        name = normalize_column_name(column)

        if "birth weight" in name and column not in protected_columns:
            columns_to_drop.append(column)

    columns_to_drop = list(set(columns_to_drop))

    if columns_to_drop:
        data = data.drop(
            columns=columns_to_drop,
            errors="ignore"
        )

    # Preprocessing summary for the dashboard.
    preprocessing_summary = {
        "Original columns": len(original_columns),
        "Columns removed": len(columns_to_drop),
        "Valid records after cleaning": len(data),
        "Missing charge values": int(
            data["Total_Charges"].isna().sum()
        ),
        "Duplicate rows removed": 0
    }

    return data, preprocessing_summary


# ============================================================
# CHART HELPERS
# ============================================================

NEON_COLORS = [
    "#00f5ff",
    "#bc13fe",
    "#ff2bd6",
    "#168aad",
    "#7c3aed",
    "#06d6a0",
    "#ffd166",
    "#f97316"
]


def style_chart(fig, height=None):
    """Apply consistent dark styling to Plotly charts."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,15,30,0.65)",
        font_color="#e8faff",
        hoverlabel=dict(
            bgcolor="#101827",
            bordercolor="#00f5ff",
            font_color="#ffffff"
        ),
        margin=dict(l=20, r=20, t=60, b=20)
    )

    if height is not None:
        fig.update_layout(height=height)

    return fig


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="dashboard-header">
        <h1>🏥 MediScope Health Analytics</h1>
        <p>
            Exploring hospital inpatient discharges, length of stay,
            healthcare charges, patient demographics, severity,
            and hospital utilization.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Dashboard Controls")
st.sidebar.markdown(
    "**Dataset:** SPARCS Hospital Inpatient Discharges 2021"
)

st.sidebar.divider()

sample_option = st.sidebar.selectbox(
    "Dataset size",
    [
        "10,000 rows",
        "25,000 rows",
        "50,000 rows",
        "100,000 rows",
        "Full dataset"
    ],
    index=2
)

sample_sizes = {
    "10,000 rows": 10000,
    "25,000 rows": 25000,
    "50,000 rows": 50000,
    "100,000 rows": 100000,
    "Full dataset": None
}

max_rows = sample_sizes[sample_option]


# ============================================================
# LOAD DATA
# ============================================================

try:
    with st.spinner("Loading and cleaning hospital data..."):
        df, preprocessing_summary = load_sparcs_data(
            TARGET_FILE,
            max_rows=max_rows
        )

except FileNotFoundError:
    st.error(
        "The SPARCS dataset could not be found. "
        "Place the CSV file in the same folder as app.py."
    )
    st.code(TARGET_FILE)
    st.stop()

except Exception as error:
    st.error(f"Unable to load the dataset: {error}")
    st.stop()

if df.empty:
    st.warning("No valid discharge records were found.")
    st.stop()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.subheader("Filter Records")

county_options = sorted(df["County"].unique().tolist())

selected_counties = st.sidebar.multiselect(
    "Hospital County",
    options=county_options,
    default=[]
)

severity_options = sorted(df["Severity_Level"].unique().tolist())

selected_severity = st.sidebar.multiselect(
    "Severity Level",
    options=severity_options,
    default=[]
)

payment_options = sorted(df["Payment_Type"].unique().tolist())

selected_payment = st.sidebar.multiselect(
    "Payment Type",
    options=payment_options,
    default=[]
)

age_options = sorted(df["Age_Group"].unique().tolist())

selected_age = st.sidebar.multiselect(
    "Age Group",
    options=age_options,
    default=[]
)

gender_options = sorted(df["Gender"].unique().tolist())

selected_gender = st.sidebar.multiselect(
    "Gender",
    options=gender_options,
    default=[]
)

race_options = sorted(df["Race"].unique().tolist())

selected_race = st.sidebar.multiselect(
    "Race",
    options=race_options,
    default=[]
)

filtered_df = df.copy()

filters = [
    ("County", selected_counties),
    ("Severity_Level", selected_severity),
    ("Payment_Type", selected_payment),
    ("Age_Group", selected_age),
    ("Gender", selected_gender),
    ("Race", selected_race)
]

for column, selected_values in filters:
    if selected_values:
        filtered_df = filtered_df[
            filtered_df[column].isin(selected_values)
        ]

if filtered_df.empty:
    st.warning(
        "No records match the selected filters. "
        "Clear one or more filters to continue."
    )
    st.stop()


# ============================================================
# RESEARCH QUESTIONS
# ============================================================

with st.expander("Research Questions"):
    st.markdown(
        """
        1. Which diagnosis categories are associated with longer hospital stays?
        2. Which age groups have the highest average hospital charges?
        3. How does severity level relate to hospital charges and length of stay?
        4. Which counties have longer average hospital stays?
        5. How do payment types differ across counties?
        6. Which facilities account for the highest number of discharges?
        7. How do charges vary by gender and race?
        """
    )


# ============================================================
# KEY PERFORMANCE INDICATORS
# ============================================================

st.subheader("Hospital Discharge Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Discharges",
    f"{len(filtered_df):,}"
)

col2.metric(
    "Average Length of Stay",
    f"{filtered_df['Length_of_Stay'].mean():.2f} days"
)

total_charges = filtered_df["Total_Charges"].sum(min_count=1)

col3.metric(
    "Total Hospital Charges",
    f"${total_charges:,.2f}" if pd.notna(total_charges) else "N/A"
)

average_charges = filtered_df["Total_Charges"].mean()

col4.metric(
    "Average Charges",
    f"${average_charges:,.2f}" if pd.notna(average_charges) else "N/A"
)

st.caption(
    "All metrics reflect the currently loaded rows and selected filters. "
    "A sample may not represent the complete dataset."
)

st.divider()


# ============================================================
# PREPROCESSING SUMMARY
# ============================================================

with st.expander("Data Preprocessing Summary"):
    st.markdown(
        """
        The data preprocessing workflow includes:
        - Converting length of stay into a numeric field.
        - Removing missing, invalid, and non-positive stay durations.
        - Converting hospital charges to numeric values.
        - Standardizing gender, age group, race, severity, and payment labels.
        - Using diagnosis descriptions where available, with code-based labels
          as a fallback.
        - Removing duplicate records.
        - Removing non-essential columns that are at least 90% null, while
          preserving fields required for dashboard analysis.
        """
    )

    st.dataframe(
        pd.DataFrame(
            {
                "Metric": list(preprocessing_summary.keys()),
                "Value": list(preprocessing_summary.values())
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The duplicate-removal count is not separately tracked in this "
        "summary; the dataset itself is deduplicated before analysis."
    )


# ============================================================
# VISUALIZATION 1
# AVERAGE STAY BY DIAGNOSIS
# ============================================================

st.subheader("1. Average Hospital Stay by Diagnosis")

diagnosis_summary = (
    filtered_df
    .groupby("Diagnosis_Category", as_index=False)
    .agg(
        Average_Stay=("Length_of_Stay", "mean"),
        Discharges=("Length_of_Stay", "size")
    )
    .sort_values("Average_Stay", ascending=False)
)

top_diagnoses = diagnosis_summary.head(15)

fig_diagnosis = px.bar(
    top_diagnoses,
    x="Average_Stay",
    y="Diagnosis_Category",
    orientation="h",
    color="Average_Stay",
    color_continuous_scale=[
        "#10243a",
        "#0077b6",
        "#00f5ff"
    ],
    title="Top Diagnosis Categories by Average Stay",
    labels={
        "Average_Stay": "Average Stay (Days)",
        "Diagnosis_Category": "Diagnosis Category"
    }
)

fig_diagnosis.update_layout(
    yaxis={"categoryorder": "total ascending"}
)

style_chart(fig_diagnosis, 550)

st.plotly_chart(
    fig_diagnosis,
    use_container_width=True
)


# ============================================================
# VISUALIZATION 2
# CHARGES BY SEVERITY
# ============================================================

st.subheader("2. Hospital Charges by Severity Level")

charges_data = filtered_df.dropna(
    subset=["Total_Charges"]
).copy()

if not charges_data.empty:
    fig_charges = px.box(
        charges_data,
        x="Severity_Level",
        y="Total_Charges",
        color="Severity_Level",
        points="outliers",
        color_discrete_sequence=NEON_COLORS,
        title="Distribution of Hospital Charges by Severity",
        labels={
            "Severity_Level": "Severity Level",
            "Total_Charges": "Total Hospital Charges ($)"
        }
    )

    fig_charges.update_layout(showlegend=False)

    style_chart(fig_charges, 500)

    st.plotly_chart(
        fig_charges,
        use_container_width=True
    )
else:
    st.info("No valid hospital charge values are available.")


# ============================================================
# VISUALIZATION 3
# FACILITY AND COUNTY HEATMAP
# ============================================================

st.subheader("3. Average Length of Stay by Facility and County")

facility_summary = (
    filtered_df
    .groupby(["County", "Facility_Name"], as_index=False)
    .agg(
        Average_Stay=("Length_of_Stay", "mean"),
        Discharges=("Length_of_Stay", "size"),
        Average_Charges=("Total_Charges", "mean")
    )
)

if not facility_summary.empty:
    selected_facilities = facility_summary.nlargest(
        20,
        "Average_Stay"
    )

    heatmap_data = selected_facilities.pivot_table(
        index="Facility_Name",
        columns="County",
        values="Average_Stay",
        aggfunc="mean"
    )

    fig_heatmap = px.imshow(
        heatmap_data,
        aspect="auto",
        color_continuous_scale=[
            "#101827",
            "#075985",
            "#00f5ff"
        ],
        title="Average Stay for Selected Facilities",
        labels={
            "x": "County",
            "y": "Facility",
            "color": "Average Stay (Days)"
        }
    )

    style_chart(fig_heatmap, 600)

    st.plotly_chart(
        fig_heatmap,
        use_container_width=True
    )


# ============================================================
# VISUALIZATION 4
# PAYMENT TYPE PIE CHART
# ============================================================

st.subheader("4. Discharges by Payment Type")

payment_summary = (
    filtered_df["Payment_Type"]
    .value_counts()
    .rename_axis("Payment_Type")
    .reset_index(name="Discharges")
)

fig_payment = px.pie(
    payment_summary,
    names="Payment_Type",
    values="Discharges",
    title="Share of Discharges by Payment Type",
    hole=0.4,
    color_discrete_sequence=NEON_COLORS
)

fig_payment.update_traces(
    textposition="inside",
    textinfo="percent+label",
    marker_line_color="#0b0f19",
    marker_line_width=2
)

style_chart(fig_payment)

st.plotly_chart(
    fig_payment,
    use_container_width=True
)


# ============================================================
# VISUALIZATION 5
# LENGTH OF STAY HISTOGRAM
# ============================================================

st.subheader("5. Distribution of Length of Stay")

stay_data = filtered_df[
    filtered_df["Length_of_Stay"] <= 30
].copy()

fig_stay = px.histogram(
    stay_data,
    x="Length_of_Stay",
    nbins=30,
    title="Length of Stay Distribution (Up to 30 Days)",
    labels={
        "Length_of_Stay": "Length of Stay (Days)",
        "count": "Number of Discharges"
    },
    color_discrete_sequence=["#00f5ff"]
)

fig_stay.update_traces(
    marker_line_color="#bc13fe",
    marker_line_width=0.7
)

fig_stay.update_layout(bargap=0.05)

style_chart(fig_stay, 450)

st.plotly_chart(
    fig_stay,
    use_container_width=True
)

st.caption(
    "This histogram focuses on stays of 30 days or fewer. "
    "Longer stays remain included in the other analyses."
)


# ============================================================
# EDA 1
# DIAGNOSIS BY SEVERITY
# ============================================================

st.divider()

st.subheader("6. Average Length of Stay by Diagnosis and Severity")

diagnosis_severity = (
    filtered_df
    .groupby(
        ["Diagnosis_Category", "Severity_Level"],
        as_index=False
    )
    .agg(
        Average_Stay=("Length_of_Stay", "mean"),
        Discharges=("Length_of_Stay", "size")
    )
)

top_diagnosis_names = (
    diagnosis_severity
    .groupby("Diagnosis_Category")["Discharges"]
    .sum()
    .nlargest(15)
    .index
)

diagnosis_severity_top = diagnosis_severity[
    diagnosis_severity["Diagnosis_Category"].isin(
        top_diagnosis_names
    )
]

fig_diagnosis_severity = px.bar(
    diagnosis_severity_top,
    x="Diagnosis_Category",
    y="Average_Stay",
    color="Severity_Level",
    barmode="group",
    title="Average Stay by Diagnosis and Severity",
    labels={
        "Diagnosis_Category": "Diagnosis",
        "Average_Stay": "Average Stay (Days)",
        "Severity_Level": "Severity"
    },
    color_discrete_sequence=NEON_COLORS
)

fig_diagnosis_severity.update_layout(
    xaxis_tickangle=-35
)

style_chart(fig_diagnosis_severity, 550)

st.plotly_chart(
    fig_diagnosis_severity,
    use_container_width=True
)

st.dataframe(
    diagnosis_severity.sort_values(
        "Average_Stay",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# EDA 2
# CHARGES BY AGE GROUP AND GENDER
# ============================================================

st.subheader("7. Average Hospital Charges by Age Group and Gender")

age_gender_summary = (
    filtered_df
    .dropna(subset=["Total_Charges"])
    .groupby(
        ["Age_Group", "Gender"],
        as_index=False
    )
    .agg(
        Average_Charges=("Total_Charges", "mean"),
        Total_Charges=("Total_Charges", "sum"),
        Discharges=("Total_Charges", "size")
    )
)

if not age_gender_summary.empty:
    fig_age_gender = px.bar(
        age_gender_summary,
        x="Age_Group",
        y="Average_Charges",
        color="Gender",
        barmode="group",
        title="Average Charges by Age Group and Gender",
        labels={
            "Age_Group": "Age Group",
            "Average_Charges": "Average Charges ($)",
            "Gender": "Gender"
        },
        color_discrete_sequence=NEON_COLORS
    )

    style_chart(fig_age_gender, 450)

    st.plotly_chart(
        fig_age_gender,
        use_container_width=True
    )

    st.dataframe(
        age_gender_summary.sort_values(
            "Average_Charges",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No charge data is available for this comparison.")


# ============================================================
# EDA 3
# TOP HIGH-UTILIZATION FACILITIES
# ============================================================

st.subheader("8. Highest-Utilization Facilities")

top_facilities = (
    filtered_df
    .groupby(
        ["Facility_Name", "County"],
        as_index=False
    )
    .agg(
        Discharges=("Length_of_Stay", "size"),
        Average_Stay=("Length_of_Stay", "mean"),
        Average_Charges=("Total_Charges", "mean"),
        Total_Charges=("Total_Charges", "sum")
    )
    .sort_values(
        "Discharges",
        ascending=False
    )
    .head(15)
)

fig_facilities = px.bar(
    top_facilities.sort_values("Discharges"),
    x="Discharges",
    y="Facility_Name",
    color="Discharges",
    orientation="h",
    color_continuous_scale=[
        "#10243a",
        "#0077b6",
        "#00f5ff"
    ],
    title="Top 15 Facilities by Discharge Volume",
    hover_data=[
        "County",
        "Average_Stay",
        "Average_Charges"
    ]
)

fig_facilities.update_layout(
    yaxis={"categoryorder": "total ascending"}
)

style_chart(fig_facilities, 550)

st.plotly_chart(
    fig_facilities,
    use_container_width=True
)

st.dataframe(
    top_facilities,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# EDA 4
# COUNTY AND PAYMENT PIVOT TABLE
# ============================================================

st.subheader("9. County and Payment Type Patterns")

county_payment_pivot = pd.pivot_table(
    filtered_df,
    index="County",
    columns="Payment_Type",
    values="Length_of_Stay",
    aggfunc="mean",
    fill_value=0
)

if not county_payment_pivot.empty:
    st.markdown(
        "**Average length of stay by county and payment type.**"
    )

    st.dataframe(
        county_payment_pivot.style.format("{:.2f}"),
        use_container_width=True
    )

    county_payment_counts = pd.pivot_table(
        filtered_df,
        index="County",
        columns="Payment_Type",
        values="Length_of_Stay",
        aggfunc="count",
        fill_value=0
    )

    st.markdown(
        "**Discharge counts by county and payment type.**"
    )

    st.dataframe(
        county_payment_counts,
        use_container_width=True
    )

    county_payment_long = (
        filtered_df
        .groupby(
            ["County", "Payment_Type"],
            as_index=False
        )
        .size()
        .rename(columns={"size": "Discharges"})
    )

    fig_county_payment = px.bar(
        county_payment_long,
        x="County",
        y="Discharges",
        color="Payment_Type",
        barmode="stack",
        title="Discharges by County and Payment Type",
        color_discrete_sequence=NEON_COLORS
    )

    fig_county_payment.update_layout(
        xaxis_tickangle=-35
    )

    style_chart(fig_county_payment, 500)

    st.plotly_chart(
        fig_county_payment,
        use_container_width=True
    )


# ============================================================
# EDA 5
# CHARGES BY GENDER
# ============================================================

st.subheader("10. Hospital Charges by Gender")

gender_charges = (
    filtered_df
    .dropna(subset=["Total_Charges"])
    .groupby("Gender", as_index=False)
    .agg(
        Average_Charges=("Total_Charges", "mean"),
        Median_Charges=("Total_Charges", "median"),
        Total_Charges=("Total_Charges", "sum"),
        Discharges=("Total_Charges", "size")
    )
    .sort_values("Average_Charges", ascending=False)
)

if not gender_charges.empty:
    fig_gender_charges = px.bar(
        gender_charges,
        x="Gender",
        y="Average_Charges",
        color="Gender",
        color_discrete_sequence=NEON_COLORS,
        title="Average Hospital Charges by Gender",
        labels={
            "Gender": "Gender",
            "Average_Charges": "Average Charges ($)"
        }
    )

    fig_gender_charges.update_layout(
        showlegend=False
    )

    style_chart(fig_gender_charges, 400)

    st.plotly_chart(
        fig_gender_charges,
        use_container_width=True
    )

    st.dataframe(
        gender_charges,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# EDA 6
# RACE AND SEVERITY
# ============================================================

st.subheader("11. Discharges by Race and Severity")

race_severity = (
    filtered_df
    .groupby(
        ["Race", "Severity_Level"],
        as_index=False
    )
    .size()
    .rename(columns={"size": "Discharges"})
)

fig_race = px.bar(
    race_severity,
    x="Race",
    y="Discharges",
    color="Severity_Level",
    barmode="group",
    title="Discharges by Race and Severity",
    color_discrete_sequence=NEON_COLORS
)

fig_race.update_layout(
    xaxis_tickangle=-25
)

style_chart(fig_race, 450)

st.plotly_chart(
    fig_race,
    use_container_width=True
)


# ============================================================
# SUMMARY TABLES
# ============================================================

st.divider()

st.subheader("Summary Tables")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Diagnosis",
        "Facilities",
        "Payment",
        "Demographics"
    ]
)

with tab1:
    st.dataframe(
        diagnosis_summary,
        use_container_width=True,
        hide_index=True
    )

with tab2:
    st.dataframe(
        top_facilities,
        use_container_width=True,
        hide_index=True
    )

with tab3:
    payment_table = (
        filtered_df
        .groupby("Payment_Type", as_index=False)
        .agg(
            Discharges=("Length_of_Stay", "size"),
            Average_Stay=("Length_of_Stay", "mean"),
            Average_Charges=("Total_Charges", "mean")
        )
        .sort_values("Discharges", ascending=False)
    )

    st.dataframe(
        payment_table,
        use_container_width=True,
        hide_index=True
    )

with tab4:
    demographics_table = (
        filtered_df
        .groupby(
            ["Age_Group", "Gender", "Race"],
            as_index=False
        )
        .agg(
            Discharges=("Length_of_Stay", "size"),
            Average_Stay=("Length_of_Stay", "mean"),
            Average_Charges=("Total_Charges", "mean")
        )
        .sort_values("Discharges", ascending=False)
    )

    st.dataframe(
        demographics_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DATA EXPLORER AND DOWNLOAD
# ============================================================

st.divider()

with st.expander("Explore Processed Dataset"):
    st.write(
        f"Showing {len(filtered_df):,} filtered records "
        f"from {len(df):,} loaded records."
    )

    st.dataframe(
        filtered_df.head(1000),
        use_container_width=True
    )

    csv_data = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv_data,
        file_name="mediscope_filtered_discharges.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        padding:1rem;
        border:1px solid rgba(0,245,255,.25);
        border-radius:12px;
        background:rgba(16,24,39,.5);
    ">
        <p style="
            color:#00f5ff;
            margin-bottom:.4rem;
            text-shadow:0 0 8px rgba(0,245,255,.3);
        ">
            <strong>MediScope Health Analytics</strong>
        </p>
        <p style="color:#cbd5e1;margin:0;">
            CRS: Artificial Intelligence |
            Course: Mathematics for AI-II |
            Dataset: SPARCS 2021
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "This dashboard supports exploratory analysis. "
    "Associations do not establish causation. Results from a loaded "
    "sample may not represent the complete dataset."
)
