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
# NEON CSS THEME
# ============================================================

st.markdown(
    """
    <style>
        :root {
            --neon-cyan: #00f5ff;
            --neon-blue: #168aad;
            --neon-purple: #bc13fe;
            --neon-pink: #ff2bd6;
            --dark-bg: #0b0f19;
            --card-bg: #141b2d;
            --text-color: #f1f5f9;
        }

        /* Main application background */

        .stApp {
            background:
                radial-gradient(
                    circle at 10% 10%,
                    rgba(0, 245, 255, 0.07),
                    transparent 35%
                ),
                radial-gradient(
                    circle at 90% 80%,
                    rgba(188, 19, 254, 0.08),
                    transparent 35%
                ),
                var(--dark-bg);
            color: var(--text-color);
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        /* Animated neon header */

        .dashboard-header {
            position: relative;
            overflow: hidden;
            padding: 1.8rem;
            border-radius: 18px;
            color: white;
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
            color: #ffffff;
            margin: 0;
            font-size: 2.2rem;
            text-shadow:
                0 0 8px rgba(0, 245, 255, 0.9),
                0 0 20px rgba(0, 245, 255, 0.5);
        }

        .dashboard-header p {
            color: #e0faff;
            margin-top: 0.7rem;
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
                rgba(255, 255, 255, 0.13),
                transparent
            );
            transform: skewX(-20deg);
            animation: lightSweep 7s ease-in-out infinite;
            pointer-events: none;
        }

        /* Neon KPI cards */

        div[data-testid="stMetric"] {
            background: linear-gradient(
                145deg,
                #172238,
                #101827
            );
            color: #ffffff;
            padding: 1.2rem;
            border-radius: 14px;
            border: 1px solid rgba(0, 245, 255, 0.55);
            box-shadow:
                0 0 7px rgba(0, 245, 255, 0.16),
                inset 0 0 10px rgba(0, 245, 255, 0.035);
            transition:
                transform 0.3s ease,
                box-shadow 0.3s ease,
                border-color 0.3s ease;
            animation: cardAppear 0.8s ease both;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-5px);
            border-color: var(--neon-cyan);
            box-shadow:
                0 0 12px rgba(0, 245, 255, 0.65),
                0 0 28px rgba(0, 245, 255, 0.22),
                inset 0 0 12px rgba(0, 245, 255, 0.08);
        }

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] label p,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] [data-testid="stMetricValue"],
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            color: #f1fbff !important;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            text-shadow: 0 0 9px rgba(0, 245, 255, 0.35);
        }

        /* Sidebar styling */

        section[data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #111827,
                #15112b,
                #101827
            );
            border-right: 1px solid rgba(188, 19, 254, 0.45);
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #ffffff;
            text-shadow: 0 0 8px rgba(188, 19, 254, 0.5);
        }

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label {
            color: #e2e8f0;
        }

        /* Input controls */

        div[data-baseweb="select"] > div {
            border-color: rgba(0, 245, 255, 0.35);
            transition:
                border-color 0.25s ease,
                box-shadow 0.25s ease;
        }

        div[data-baseweb="select"] > div:hover {
            border-color: var(--neon-cyan);
            box-shadow: 0 0 9px rgba(0, 245, 255, 0.25);
        }

        div[data-testid="stSlider"] [role="slider"] {
            box-shadow: 0 0 10px rgba(0, 245, 255, 0.6);
        }

        /* Headings */

        h1, h2, h3 {
            color: #e8faff;
        }

        div[data-testid="stMarkdownContainer"] h2 {
            border-bottom: 1px solid rgba(0, 245, 255, 0.25);
            padding-bottom: 0.45rem;
        }

        /* Chart borders and hover effects */

        div[data-testid="stPlotlyChart"] {
            border: 1px solid rgba(0, 245, 255, 0.23);
            border-radius: 13px;
            padding: 0.35rem;
            background: rgba(16, 24, 39, 0.5);
            transition:
                border-color 0.3s ease,
                box-shadow 0.3s ease;
        }

        div[data-testid="stPlotlyChart"]:hover {
            border-color: rgba(0, 245, 255, 0.75);
            box-shadow: 0 0 15px rgba(0, 245, 255, 0.12);
        }

        /* Buttons */

        div.stButton > button,
        div.stDownloadButton > button {
            color: #ffffff;
            background: linear-gradient(
                100deg,
                #073b4c,
                #321052
            );
            border: 1px solid var(--neon-cyan);
            border-radius: 9px;
            transition: all 0.25s ease;
        }

        div.stButton > button:hover,
        div.stDownloadButton > button:hover {
            color: #ffffff;
            border-color: var(--neon-pink);
            box-shadow:
                0 0 10px rgba(0, 245, 255, 0.45),
                0 0 18px rgba(188, 19, 254, 0.25);
            transform: translateY(-2px);
        }

        /* Tabs */

        button[data-baseweb="tab"] {
            transition: all 0.25s ease;
        }

        button[data-baseweb="tab"]:hover {
            color: var(--neon-cyan);
            text-shadow: 0 0 8px rgba(0, 245, 255, 0.6);
        }

        /* Expanders */

        div[data-testid="stExpander"] {
            border: 1px solid rgba(0, 245, 255, 0.25);
            border-radius: 12px;
            transition: box-shadow 0.3s ease;
        }

        div[data-testid="stExpander"]:hover {
            box-shadow: 0 0 12px rgba(0, 245, 255, 0.12);
        }

        /* Dataframes */

        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(0, 245, 255, 0.25);
            border-radius: 10px;
            overflow: hidden;
        }

        /* Neon divider */

        hr {
            border-color: rgba(0, 245, 255, 0.25);
        }

        /* Animations */

        @keyframes gradientMove {
            0%   { background-position: 0% 50%; }
            50%  { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        @keyframes neonPulse {
            0%, 100% {
                box-shadow:
                    0 0 8px rgba(0, 245, 255, 0.22),
                    0 0 18px rgba(0, 245, 255, 0.08);
            }
            50% {
                box-shadow:
                    0 0 15px rgba(0, 245, 255, 0.42),
                    0 0 30px rgba(188, 19, 254, 0.16);
            }
        }

        @keyframes lightSweep {
            0%, 65% { left: -150%; }
            100% { left: 180%; }
        }

        @keyframes cardAppear {
            from {
                opacity: 0;
                transform: translateY(12px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Accessibility */

        @media (prefers-reduced-motion: reduce) {
            *,
            *::before,
            *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATASET COLUMN DETECTION
# ============================================================

def find_column(dataframe, possible_names):
    """Find one matching source column."""

    normalized_columns = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    # Exact matches first
    for possible_name in possible_names:
        possible_name = possible_name.lower()

        if possible_name in normalized_columns:
            return normalized_columns[possible_name]

    # Partial matches second
    for possible_name in possible_names:
        possible_name = possible_name.lower()

        for normalized_name, original_name in normalized_columns.items():
            if possible_name in normalized_name:
                return original_name

    return None


# ============================================================
# LOAD AND PREPROCESS DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_sparcs_data(file_path, max_rows=50000):
    """Load and clean the fixed SPARCS 2021 dataset."""

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    dataframe = pd.read_csv(
        file_path,
        nrows=max_rows,
        low_memory=False
    )

    # Remove duplicate original column labels
    dataframe = dataframe.loc[
        :, ~dataframe.columns.duplicated()
    ].copy()

    column_mapping = {
        "Length_of_Stay": [
            "length of stay"
        ],
        "Total_Charges": [
            "total charges",
            "total charge"
        ],
        "Facility_Name": [
            "facility name"
        ],
        "County": [
            "hospital county",
            "facility county",
            "county"
        ],
        "Diagnosis_Category": [
            "apr mdc description",
            "mdc description",
            "apr drg description"
        ],
        "Severity_Level": [
            "apr severity of illness description",
            "severity of illness"
        ],
        "Payment_Type": [
            "payment typology 1",
            "payment type"
        ],
        "Age_Group": [
            "age group"
        ],
        "Gender": [
            "gender"
        ]
    }

    rename_mapping = {}

    for standard_name, possible_names in column_mapping.items():
        source_column = find_column(
            dataframe,
            possible_names
        )

        if source_column is not None:
            rename_mapping[source_column] = standard_name

    dataframe = dataframe.rename(
        columns=rename_mapping
    )

    if "Length_of_Stay" not in dataframe.columns:
        raise ValueError(
            "Could not find a recognizable Length of Stay column "
            "in the SPARCS dataset."
        )

    # Clean length of stay
    dataframe["Length_of_Stay"] = (
        dataframe["Length_of_Stay"]
        .astype(str)
        .str.replace("+", "", regex=False)
        .str.strip()
    )

    dataframe["Length_of_Stay"] = pd.to_numeric(
        dataframe["Length_of_Stay"],
        errors="coerce"
    )

    # Remove invalid and non-positive stays
    dataframe = dataframe[
        dataframe["Length_of_Stay"].notna()
        & (dataframe["Length_of_Stay"] > 0)
    ].copy()

    # Convert hospital charges to numeric
    if "Total_Charges" in dataframe.columns:
        dataframe["Total_Charges"] = (
            dataframe["Total_Charges"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        dataframe["Total_Charges"] = pd.to_numeric(
            dataframe["Total_Charges"],
            errors="coerce"
        )
    else:
        dataframe["Total_Charges"] = float("nan")

    # Fallback values for unavailable columns
    fallback_columns = {
        "Facility_Name": "Unknown Facility",
        "County": "Unknown County",
        "Diagnosis_Category": "Unknown Diagnosis",
        "Severity_Level": "Unknown Severity",
        "Payment_Type": "Unknown Payment Type",
        "Age_Group": "Unknown Age Group",
        "Gender": "Unknown"
    }

    for column, default_value in fallback_columns.items():
        if column not in dataframe.columns:
            dataframe[column] = default_value

        dataframe[column] = (
            dataframe[column]
            .fillna(default_value)
            .astype(str)
            .str.strip()
            .replace("", default_value)
        )

    # Standardize gender
    dataframe["Gender"] = (
        dataframe["Gender"]
        .str.strip()
        .str.title()
        .replace({
            "M": "Male",
            "F": "Female"
        })
    )

    # Standardize severity and payment labels
    dataframe["Severity_Level"] = (
        dataframe["Severity_Level"].str.strip().str.title()
    )

    dataframe["Payment_Type"] = (
        dataframe["Payment_Type"].str.strip().str.title()
    )

    # Remove duplicate records
    dataframe = dataframe.drop_duplicates().copy()

    return dataframe


# ============================================================
# NEON DASHBOARD HEADER
# ============================================================

st.markdown(
    """
    <div class="dashboard-header">
        <h1>🏥 MediScope Health Analytics</h1>
        <p>
            Exploring hospital inpatient discharges, length of stay,
            healthcare charges, patient demographics, and hospital trends.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.title("🏥 Dashboard Controls")

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
# LOAD DATASET
# ============================================================

try:
    with st.spinner("Loading and preparing hospital data..."):
        df = load_sparcs_data(
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
    st.warning("No valid hospital discharge records were found.")
    st.stop()


# ============================================================
# FILTERS
# ============================================================

st.sidebar.subheader("Filter Records")

county_options = sorted(
    df["County"].dropna().unique().tolist()
)

selected_counties = st.sidebar.multiselect(
    "Hospital County",
    options=county_options,
    default=[]
)

severity_options = sorted(
    df["Severity_Level"].dropna().unique().tolist()
)

selected_severity = st.sidebar.multiselect(
    "Severity Level",
    options=severity_options,
    default=[]
)

payment_options = sorted(
    df["Payment_Type"].dropna().unique().tolist()
)

selected_payment = st.sidebar.multiselect(
    "Payment Type",
    options=payment_options,
    default=[]
)

filtered_df = df.copy()

if selected_counties:
    filtered_df = filtered_df[
        filtered_df["County"].isin(selected_counties)
    ]

if selected_severity:
    filtered_df = filtered_df[
        filtered_df["Severity_Level"].isin(selected_severity)
    ]

if selected_payment:
    filtered_df = filtered_df[
        filtered_df["Payment_Type"].isin(selected_payment)
    ]

if filtered_df.empty:
    st.warning(
        "No records match your selected filters. "
        "Change or clear your filters."
    )
    st.stop()


# ============================================================
# KEY PERFORMANCE INDICATORS
# ============================================================

st.subheader("Hospital Discharge Overview")

total_discharges = len(filtered_df)

average_stay = filtered_df["Length_of_Stay"].mean()

total_charges = filtered_df["Total_Charges"].sum(min_count=1)

average_charges = filtered_df["Total_Charges"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Discharges",
    f"{total_discharges:,}"
)

col2.metric(
    "Average Length of Stay",
    f"{average_stay:.2f} days"
)

col3.metric(
    "Total Hospital Charges",
    (
        f"${total_charges:,.2f}"
        if pd.notna(total_charges)
        else "N/A"
    )
)

col4.metric(
    "Average Charges",
    (
        f"${average_charges:,.2f}"
        if pd.notna(average_charges)
        else "N/A"
    )
)

st.caption(
    "KPIs reflect the loaded rows and selected filters. "
    "A sample may not represent the complete dataset."
)

st.divider()


# ============================================================
# RESEARCH QUESTIONS
# ============================================================

with st.expander("Research Questions"):
    st.markdown(
        """
        1. Which diagnosis categories have the longest average hospital stays?
        2. How do hospital charges vary across severity levels?
        3. How does average length of stay vary by hospital and county?
        4. Which payment types account for the largest share of discharges?
        5. What is the distribution of inpatient length of stay?
        6. How do hospital charges vary between age groups?
        7. How do discharge records differ by gender and severity level?
        """
    )


# ============================================================
# VISUALIZATION 1: AVERAGE STAY BY DIAGNOSIS
# ============================================================

st.subheader("1. Average Length of Stay by Diagnosis")

diagnosis_summary = (
    filtered_df
    .groupby("Diagnosis_Category", as_index=False)
    ["Length_of_Stay"]
    .mean()
    .rename(
        columns={
            "Length_of_Stay": "Average_Stay"
        }
    )
    .sort_values("Average_Stay", ascending=False)
)

top_diagnoses = diagnosis_summary.head(15)

fig_diagnosis = px.bar(
    top_diagnoses,
    x="Average_Stay",
    y="Diagnosis_Category",
    orientation="h",
    title="Top 15 Diagnosis Categories by Average Stay",
    labels={
        "Average_Stay": "Average Stay (Days)",
        "Diagnosis_Category": "Diagnosis Category"
    },
    color="Average_Stay",
    color_continuous_scale=[
        "#10243a",
        "#0077b6",
        "#00f5ff"
    ],
    template="plotly_dark"
)

fig_diagnosis.update_layout(
    yaxis={"categoryorder": "total ascending"},
    height=550,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,15,30,0.65)",
    font_color="#e8faff"
)

st.plotly_chart(fig_diagnosis, use_container_width=True)

if not diagnosis_summary.empty:
    longest_diagnosis = diagnosis_summary.iloc[0]

    st.info(
        f"Within the currently loaded and filtered records, "
        f"'{longest_diagnosis['Diagnosis_Category']}' has the "
        f"highest average length of stay "
        f"({longest_diagnosis['Average_Stay']:.2f} days)."
    )


# ============================================================
# VISUALIZATION 2: CHARGES BY SEVERITY
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
        title="Distribution of Hospital Charges by Severity",
        labels={
            "Severity_Level": "Severity Level",
            "Total_Charges": "Total Hospital Charges ($)"
        },
        points="outliers",
        template="plotly_dark",
        color_discrete_sequence=[
            "#00f5ff",
            "#bc13fe",
            "#ff2bd6",
            "#168aad"
        ]
    )

    fig_charges.update_layout(
        showlegend=False,
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,15,30,0.65)",
        font_color="#e8faff"
    )

    st.plotly_chart(fig_charges, use_container_width=True)

    st.caption(
        "Box plots display medians, spread, and potential outliers. "
        "Extreme charges may affect how differences appear."
    )
else:
    st.warning(
        "No valid numeric hospital charge values are available."
    )


# ============================================================
# VISUALIZATION 3: FACILITY AND COUNTY HEATMAP
# ============================================================

st.subheader("3. Average Length of Stay by Facility and County")

facility_summary = (
    filtered_df
    .groupby(
        ["County", "Facility_Name"],
        as_index=False
    )
    ["Length_of_Stay"]
    .mean()
    .rename(
        columns={
            "Length_of_Stay": "Average_Stay"
        }
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
        },
        template="plotly_dark"
    )

    fig_heatmap.update_layout(
        height=600,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,15,30,0.65)",
        font_color="#e8faff"
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)
else:
    st.info("Facility and county information is unavailable.")


# ============================================================
# VISUALIZATION 4: PAYMENT TYPE
# ============================================================

st.subheader("4. Discharges by Payment Type")

payment_summary = (
    filtered_df["Payment_Type"]
    .value_counts()
    .reset_index()
)

payment_summary.columns = [
    "Payment_Type",
    "Discharges"
]

fig_payment = px.pie(
    payment_summary,
    names="Payment_Type",
    values="Discharges",
    title="Share of Discharges by Payment Type",
    hole=0.4,
    template="plotly_dark",
    color_discrete_sequence=[
        "#00f5ff",
        "#bc13fe",
        "#ff2bd6",
        "#168aad",
        "#7c3aed",
        "#06d6a0",
        "#ffd166"
    ]
)

fig_payment.update_traces(
    textposition="inside",
    textinfo="percent+label",
    marker_line_color="#0b0f19",
    marker_line_width=2
)

fig_payment.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#e8faff"
)

st.plotly_chart(fig_payment, use_container_width=True)


# ============================================================
# VISUALIZATION 5: LENGTH OF STAY DISTRIBUTION
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
    color_discrete_sequence=["#00f5ff"],
    template="plotly_dark"
)

fig_stay.update_layout(
    height=450,
    bargap=0.05,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,15,30,0.65)",
    font_color="#e8faff"
)

fig_stay.update_traces(
    marker_line_color="#bc13fe",
    marker_line_width=0.7
)

st.plotly_chart(fig_stay, use_container_width=True)

st.caption(
    "This histogram focuses on stays of 30 days or fewer. "
    "Longer stays remain included in the other calculations."
)


# ============================================================
# VISUALIZATION 6: CHARGES BY AGE GROUP
# ============================================================

st.divider()

st.subheader("6. Average Hospital Charges by Age Group")

age_summary = (
    filtered_df
    .groupby("Age_Group", as_index=False)
    ["Total_Charges"]
    .mean()
    .rename(
        columns={
            "Total_Charges": "Average_Charges"
        }
    )
    .sort_values("Average_Charges", ascending=False)
    .dropna(subset=["Average_Charges"])
)

if not age_summary.empty:
    fig_age = px.bar(
        age_summary,
        x="Age_Group",
        y="Average_Charges",
        color="Average_Charges",
        color_continuous_scale=[
            "#10243a",
            "#0077b6",
            "#00f5ff"
        ],
        title="Average Hospital Charges by Age Group",
        labels={
            "Age_Group": "Age Group",
            "Average_Charges": "Average Charges ($)"
        },
        template="plotly_dark"
    )

    fig_age.update_layout(
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,15,30,0.65)",
        font_color="#e8faff"
    )

    st.plotly_chart(fig_age, use_container_width=True)
else:
    st.info("No valid charge values are available for age analysis.")


# ============================================================
# VISUALIZATION 7: GENDER AND SEVERITY
# ============================================================

st.subheader("7. Discharges by Gender and Severity")

gender_severity = pd.crosstab(
    filtered_df["Gender"],
    filtered_df["Severity_Level"]
)

if not gender_severity.empty:
    gender_plot_data = (
        gender_severity
        .reset_index()
        .melt(
            id_vars="Gender",
            var_name="Severity Level",
            value_name="Discharges"
        )
    )

    fig_gender = px.bar(
        gender_plot_data,
        x="Gender",
        y="Discharges",
        color="Severity Level",
        barmode="group",
        title="Discharges by Gender and Severity Level",
        template="plotly_dark",
        color_discrete_sequence=[
            "#00f5ff",
            "#bc13fe",
            "#ff2bd6",
            "#168aad"
        ]
    )

    fig_gender.update_layout(
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,15,30,0.65)",
        font_color="#e8faff"
    )

    st.plotly_chart(fig_gender, use_container_width=True)


# ============================================================
# SUMMARY TABLES
# ============================================================

st.divider()

st.subheader("Summary Tables")

tab1, tab2, tab3 = st.tabs(
    [
        "Diagnosis Summary",
        "Facility Summary",
        "Payment Summary"
    ]
)

with tab1:
    st.dataframe(
        diagnosis_summary,
        use_container_width=True,
        hide_index=True
    )

with tab2:
    facility_table = (
        filtered_df
        .groupby(
            ["County", "Facility_Name"],
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
        facility_table,
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


# ============================================================
# DATA EXPLORER AND CSV DOWNLOAD
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
        text-align: center;
        padding: 1rem;
        border: 1px solid rgba(0,245,255,0.25);
        border-radius: 12px;
        background: rgba(16,24,39,0.5);
    ">
        <p style="
            color: #00f5ff;
            margin-bottom: 0.4rem;
            text-shadow: 0 0 8px rgba(0,245,255,0.3);
        ">
            <strong>MediScope Health Analytics</strong>
        </p>
        <p style="color: #cbd5e1; margin: 0;">
            CRS: Artificial Intelligence |
            SPARCS Hospital Inpatient Discharges, 2021
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "This dashboard supports exploratory analysis. "
    "Associations do not establish causation, and sampled records "
    "may not represent the complete dataset."
)
