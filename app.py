import os
import streamlit as st
import pandas as pd
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MediScope | Hospital Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONSTANTS
# ============================================================

TARGET_FILE = (
    "Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv"
)

APP_TITLE = "MediScope Health Analytics"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main {
            background-color: #f5f7fb;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        .dashboard-header {
            padding: 1.5rem;
            border-radius: 16px;
            background: linear-gradient(120deg, #123b67, #168aad);
            color: white;
            margin-bottom: 1.5rem;
        }

        .dashboard-header h1 {
            margin: 0;
            font-size: 2.2rem;
        }

        .dashboard-header p {
            margin-top: 0.5rem;
            margin-bottom: 0;
            font-size: 1rem;
        }
        
        div[data-testid="stMetric"] {
        background: #1e293b;
        color: #f8fafc;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15);
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] label p,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #f8fafc !important;
    }
        h2, h3 {
            color: #123b67;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATASET COLUMN DETECTION
# ============================================================

def find_column(dataframe, possible_names):
    """
    Find one matching column without renaming multiple columns
    to the same standardized name.
    """

    normalized_columns = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    # First look for exact matches
    for possible_name in possible_names:
        possible_name = possible_name.lower()

        if possible_name in normalized_columns:
            return normalized_columns[possible_name]

    # Then look for partial matches
    for possible_name in possible_names:
        possible_name = possible_name.lower()

        for normalized_name, original_name in normalized_columns.items():
            if possible_name in normalized_name:
                return original_name

    return None


# ============================================================
# DATA LOADING AND PREPROCESSING
# ============================================================

@st.cache_data(show_spinner=False)
def load_sparcs_data(file_path, max_rows=50000):
    """
    Load and preprocess the fixed SPARCS 2021 dataset.

    max_rows=None loads the complete dataset.
    No external dataset upload is supported.
    """

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

    # Map source columns to standardized column names
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

    # Required field validation
    if "Length_of_Stay" not in dataframe.columns:
        raise ValueError(
            "The dataset does not contain a recognizable "
            "'Length of Stay' column."
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

    # Remove missing, invalid, and non-positive stays
    dataframe = dataframe[
        dataframe["Length_of_Stay"].notna()
        & (dataframe["Length_of_Stay"] > 0)
    ].copy()

    # Clean total charges
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

    # Create fallback columns if source fields are unavailable
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

    # Standardize gender labels
    dataframe["Gender"] = (
        dataframe["Gender"]
        .str.strip()
        .str.title()
        .replace({
            "M": "Male",
            "F": "Female"
        })
    )

    # Standardize severity labels
    dataframe["Severity_Level"] = (
        dataframe["Severity_Level"]
        .str.strip()
        .str.title()
    )

    # Standardize payment labels
    dataframe["Payment_Type"] = (
        dataframe["Payment_Type"]
        .str.strip()
        .str.title()
    )

    # Remove duplicate rows
    dataframe = dataframe.drop_duplicates().copy()

    return dataframe


# ============================================================
# HEADER
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
# SIDEBAR
# ============================================================

st.sidebar.title("Dashboard Controls")

st.sidebar.markdown(
    """
    **Dataset:** SPARCS Hospital Inpatient Discharges 2021

    This dashboard uses the fixed project dataset.
    Uploading other datasets is disabled.
    """
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
    st.error(
        f"Unable to load the dataset: {error}"
    )
    st.stop()


if df.empty:
    st.warning(
        "No valid hospital discharge records were found."
    )
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
        "No records match the selected filters. "
        "Change or clear your filters."
    )
    st.stop()


# ============================================================
# KEY PERFORMANCE INDICATORS
# ============================================================

st.subheader("Hospital Discharge Overview")

total_discharges = len(filtered_df)

average_stay = filtered_df[
    "Length_of_Stay"
].mean()

total_charges = filtered_df[
    "Total_Charges"
].sum(min_count=1)

average_charges = filtered_df[
    "Total_Charges"
].mean()

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
    "KPIs reflect the currently loaded rows and selected filters. "
    "When a sample is selected, results describe that sample rather "
    "than the complete dataset."
)

st.divider()


# ============================================================
# RESEARCH QUESTIONS
# ============================================================

with st.expander("Research Questions"):
    st.markdown(
        """
        This dashboard investigates the following questions:

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
    .sort_values(
        "Average_Stay",
        ascending=False
    )
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
    color_continuous_scale="Blues"
)

fig_diagnosis.update_layout(
    yaxis={"categoryorder": "total ascending"},
    height=550
)

st.plotly_chart(
    fig_diagnosis,
    use_container_width=True
)

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
        points="outliers"
    )

    fig_charges.update_layout(
        showlegend=False,
        height=500
    )

    st.plotly_chart(
        fig_charges,
        use_container_width=True
    )

    st.caption(
        "Box plots show the median, spread, and potential outliers. "
        "Extreme charges can make differences between groups harder "
        "to see, so interpret the distribution as well as the median."
    )
else:
    st.warning(
        "No valid numeric hospital charge values are available "
        "for the selected records."
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
    # Limit the heatmap to a manageable number of facilities
    # with the highest average length of stay.
    selected_facilities = (
        facility_summary
        .nlargest(20, "Average_Stay")
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
        color_continuous_scale="Blues",
        title="Average Stay for Selected Facilities",
        labels={
            "x": "County",
            "y": "Facility",
            "color": "Average Stay (Days)"
        }
    )

    fig_heatmap.update_layout(
        height=600
    )

    st.plotly_chart(
        fig_heatmap,
        use_container_width=True
    )
else:
    st.info(
        "Facility and county information is unavailable "
        "for the selected records."
    )


# ============================================================
# VISUALIZATION 4: PAYMENT TYPE PIE CHART
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
    hole=0.35
)

fig_payment.update_traces(
    textposition="inside",
    textinfo="percent+label"
)

st.plotly_chart(
    fig_payment,
    use_container_width=True
)


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
    color_discrete_sequence=["#168aad"]
)

fig_stay.update_layout(
    height=450,
    bargap=0.05
)

st.plotly_chart(
    fig_stay,
    use_container_width=True
)

st.caption(
    "This histogram focuses on stays of 30 days or fewer. "
    "Longer stays are excluded from this chart only, not from "
    "the other dashboard calculations."
)


# ============================================================
# ADDITIONAL ANALYSIS: CHARGES BY AGE GROUP
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
    .sort_values(
        "Average_Charges",
        ascending=False
    )
)

age_summary = age_summary.dropna(
    subset=["Average_Charges"]
)

if not age_summary.empty:
    fig_age = px.bar(
        age_summary,
        x="Age_Group",
        y="Average_Charges",
        color="Average_Charges",
        color_continuous_scale="Teal",
        title="Average Hospital Charges by Age Group",
        labels={
            "Age_Group": "Age Group",
            "Average_Charges": "Average Charges ($)"
        }
    )

    fig_age.update_layout(
        height=450
    )

    st.plotly_chart(
        fig_age,
        use_container_width=True
    )
else:
    st.info(
        "There are no valid charge values for the age-group analysis."
    )


# ============================================================
# ADDITIONAL ANALYSIS: GENDER AND SEVERITY
# ============================================================

st.subheader("7. Discharges by Gender and Severity")

gender_severity = pd.crosstab(
    filtered_df["Gender"],
    filtered_df["Severity_Level"]
)

if not gender_severity.empty:
    fig_gender = px.bar(
        gender_severity.reset_index().melt(
            id_vars="Gender",
            var_name="Severity Level",
            value_name="Discharges"
        ),
        x="Gender",
        y="Discharges",
        color="Severity Level",
        barmode="group",
        title="Discharges by Gender and Severity Level"
    )

    fig_gender.update_layout(
        height=450
    )

    st.plotly_chart(
        fig_gender,
        use_container_width=True
    )


# ============================================================
# DATA TABLES
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
        .sort_values(
            "Discharges",
            ascending=False
        )
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
        .sort_values(
            "Discharges",
            ascending=False
        )
    )

    st.dataframe(
        payment_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DATA EXPLORER
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

st.caption(
    "MediScope Health Analytics | CRS: Artificial Intelligence | "
    "Dataset: SPARCS Hospital Inpatient Discharges, 2021"
)

st.caption(
    "This dashboard supports exploratory analysis. "
    "Observed associations do not establish causation, and "
    "the loaded sample may not represent the complete dataset."
)
