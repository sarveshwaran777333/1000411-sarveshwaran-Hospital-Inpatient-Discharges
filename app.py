import os
import glob
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MediScope Health Analytics - SPARCS 2021 Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; color: #1E3A8A; font-weight: 700; }
    .sub-header { font-size: 1.1rem; color: #4B5563; }
    </style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# DATA LOADING & SPARCS PREPROCESSING
# -----------------------------------------------------------------------------
TARGET_FILE = "Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv"

@st.cache_data(ttl=3600, show_spinner="Loading and cleaning SPARCS dataset...")
def load_sparcs_data(file_source, max_rows=50000):
    """
    Reads and preprocesses the NY State SPARCS Inpatient Discharge dataset.
    """
    # 1. Read CSV
    if isinstance(file_source, str):
        if max_rows and max_rows > 0:
            df = pd.read_csv(file_source, nrows=max_rows, low_memory=False)
        else:
            df = pd.read_csv(file_source, low_memory=False)
    else:
        df = pd.read_csv(file_source, nrows=max_rows if max_rows else None, low_memory=False)

    # Clean Column Names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # 2. SPARCS Specific Column Standardizer
    col_map = {}
    for col in df.columns:
        c_lower = col.lower()
        if 'length of stay' in c_lower or 'los' in c_lower:
            col_map[col] = 'Length_of_Stay'
        elif 'total charges' in c_lower or 'charges' in c_lower:
            col_map[col] = 'Total_Charges'
        elif 'facility name' in c_lower:
            col_map[col] = 'Facility_Name'
        elif 'hospital county' in c_lower or 'county' in c_lower:
            col_map[col] = 'County'
        elif 'apr mdc description' in c_lower or 'mdc description' in c_lower:
            col_map[col] = 'Diagnosis_Category'
        elif 'apr drg description' in c_lower and 'Diagnosis_Category' not in col_map.values():
            col_map[col] = 'Diagnosis_Category'
        elif 'apr severity of illness description' in c_lower or 'severity' in c_lower:
            col_map[col] = 'Severity_Level'
        elif 'payment typology 1' in c_lower or 'payment' in c_lower:
            col_map[col] = 'Payment_Type'
        elif 'age group' in c_lower:
            col_map[col] = 'Age_Group'
        elif 'gender' in c_lower:
            col_map[col] = 'Gender'

    df = df.rename(columns=col_map)

    # 3. Handle Fallbacks if missing
    if 'Facility_Name' not in df.columns:
        df['Facility_Name'] = 'Unknown Facility'
    if 'County' not in df.columns:
        df['County'] = 'Unknown County'
    if 'Severity_Level' not in df.columns:
        df['Severity_Level'] = 'Moderate'
    if 'Payment_Type' not in df.columns:
        df['Payment_Type'] = 'Insurance'
    if 'Diagnosis_Category' not in df.columns:
        df['Diagnosis_Category'] = 'General Medical'

    # 4. Clean 'Length_of_Stay' (SPARCS represents >120 days as "120+")
    if 'Length_of_Stay' in df.columns:
        df['Length_of_Stay'] = df['Length_of_Stay'].astype(str).str.replace('+', '', regex=False)
        df['Length_of_Stay'] = pd.to_numeric(df['Length_of_Stay'], errors='coerce')
        df = df[df['Length_of_Stay'] > 0]

    # 5. Clean 'Total_Charges'
    if 'Total_Charges' in df.columns:
        df['Total_Charges'] = df['Total_Charges'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
        df['Total_Charges'] = pd.to_numeric(df['Total_Charges'], errors='coerce')
        df['Total_Charges'] = df['Total_Charges'].fillna(df['Total_Charges'].median())

    # 6. Simplify Payment Types for readability
    if 'Payment_Type' in df.columns:
        df['Payment_Type'] = df['Payment_Type'].replace({
            'Medicare': 'Medicare',
            'Medicaid': 'Medicaid',
            'Private Health Insurance': 'Private Insurance',
            'Blue Cross': 'Private Insurance',
            'Self-Pay': 'Self-Pay'
        })

    return df


# -----------------------------------------------------------------------------
# MAIN APP INTERFACE
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">MediScope Health Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">NY State SPARCS Hospital Inpatient Discharges & Performance Metrics</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar - Dataset Detection & Filters
st.sidebar.header("⚙️ Data Settings & Filters")

uploaded_file = st.sidebar.file_uploader("Upload Different CSV", type=["csv"])

# Determine Data File Location
data_source = None
if uploaded_file is not None:
    data_source = uploaded_file
    source_label = f"Uploaded: {uploaded_file.name}"
elif os.path.exists(TARGET_FILE):
    data_source = TARGET_FILE
    source_label = f"Local SPARCS Dataset (`{TARGET_FILE}`)"
else:
    # Search for any .csv file in current folder
    csv_list = glob.glob("*.csv")
    if csv_list:
        data_source = csv_list[0]
        source_label = f"Local File (`{csv_list[0]}`)"
    else:
        st.error(f"Dataset `{TARGET_FILE}` not found in the current folder. Please place the CSV in the same folder as `app.py` or upload it via sidebar.")
        st.stop()

# Performance Control: Row Limit Slider
sample_size = st.sidebar.select_slider(
    "Data Sample Size (for optimal speed):",
    options=[10000, 25000, 50000, 100000, "Full Dataset"],
    value=50000
)

max_rows_val = None if sample_size == "Full Dataset" else sample_size

df_raw = load_sparcs_data(data_source, max_rows=max_rows_val)
st.sidebar.success(f"Loaded: **{source_label}** ({len(df_raw):,} records)")

# Filter Controls
selected_counties = st.sidebar.multiselect(
    "Filter County:",
    options=sorted(df_raw['County'].dropna().unique()),
    default=sorted(df_raw['County'].dropna().unique())[:5] if len(df_raw['County'].dropna().unique()) > 5 else sorted(df_raw['County'].dropna().unique())
)

selected_severity = st.sidebar.multiselect(
    "Filter Severity Level:",
    options=df_raw['Severity_Level'].dropna().unique(),
    default=df_raw['Severity_Level'].dropna().unique()
)

selected_payer = st.sidebar.multiselect(
    "Filter Payment Type:",
    options=df_raw['Payment_Type'].dropna().unique(),
    default=df_raw['Payment_Type'].dropna().unique()
)

# Apply Active Filters
df_filtered = df_raw[
    (df_raw['County'].isin(selected_counties)) &
    (df_raw['Severity_Level'].isin(selected_severity)) &
    (df_raw['Payment_Type'].isin(selected_payer))
]

if df_filtered.empty:
    st.warning("No records match your selected sidebar filters. Please expand your selection.")
    st.stop()

# -----------------------------------------------------------------------------
# EXECUTIVE KPI CARDS
# -----------------------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(label="Total Discharges Analyzed", value=f"{len(df_filtered):,}")
with kpi2:
    st.metric(label="Avg Length of Stay (ALOS)", value=f"{df_filtered['Length_of_Stay'].mean():.1f} Days")
with kpi3:
    st.metric(label="Total Hospital Billing", value=f"${df_filtered['Total_Charges'].sum():,.2f}")
with kpi4:
    st.metric(label="Avg Charge per Stay", value=f"${df_filtered['Total_Charges'].mean():,.2f}")

st.markdown("---")

# -----------------------------------------------------------------------------
# INTERACTIVE VISUALIZATIONS (STEP 4)
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Primary Visualizations", "📋 Diagnostic Summary Tables", "📄 Data Explorer"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        # CHART 1: Average Length of Stay per Diagnosis Category
        st.subheader("1. Avg Length of Stay (Days) by Major Diagnosis")
        df_diag = df_filtered.groupby('Diagnosis_Category')['Length_of_Stay'].mean().reset_index()
        df_diag = df_diag.sort_values(by='Length_of_Stay', ascending=True).tail(10)  # Top 10
        fig1 = px.bar(
            df_diag,
            x='Length_of_Stay',
            y='Diagnosis_Category',
            orientation='h',
            color='Length_of_Stay',
            color_continuous_scale='Blues',
            labels={'Length_of_Stay': 'Avg Stay (Days)', 'Diagnosis_Category': 'Diagnosis Category'},
            text_auto='.1f'
        )
        fig1.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # CHART 2: Financial Charges Distribution by Severity Level
        st.subheader("2. Total Hospital Charges by Severity Level")
        fig2 = px.box(
            df_filtered,
            x='Severity_Level',
            y='Total_Charges',
            color='Severity_Level',
            labels={'Total_Charges': 'Total Charges ($)', 'Severity_Level': 'Severity Level'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig2.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # CHART 3: Heatmap - Stay Duration by Top Facilities & Counties
        st.subheader("3. Heatmap: Average Stay (Days) by Facility & County")
        top_facilities = df_filtered['Facility_Name'].value_counts().head(8).index
        df_sub = df_filtered[df_filtered['Facility_Name'].isin(top_facilities)]
        heatmap_data = df_sub.pivot_table(
            index='Facility_Name',
            columns='County',
            values='Length_of_Stay',
            aggfunc='mean'
        ).fillna(0)
        
        fig3 = px.imshow(
            heatmap_data,
            labels=dict(x="County", y="Facility", color="Avg Stay (Days)"),
            color_continuous_scale='Viridis',
            text_auto='.1f'
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # CHART 4: Patient Volume by Payment Typology
        st.subheader("4. Patient Volume Breakdown by Payment Type")
        df_payer = df_filtered['Payment_Type'].value_counts().head(6).reset_index()
        df_payer.columns = ['Payment_Type', 'Count']
        fig4 = px.pie(
            df_payer,
            names='Payment_Type',
            values='Count',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)

    # CHART 5: Histogram - Length of Stay Distribution
    st.subheader("5. Distribution Profile of Length of Stay (LOS)")
    fig5 = px.histogram(
        df_filtered[df_filtered['Length_of_Stay'] <= 30],  # Filter extreme outliers for clean view
        x='Length_of_Stay',
        nbins=30,
        color_discrete_sequence=['#2563EB'],
        marginal='rug',
        labels={'Length_of_Stay': 'Length of Stay (Days)'}
    )
    fig5.update_layout(height=350, yaxis_title="Discharge Volume")
    st.plotly_chart(fig5, use_container_width=True)

with tab2:
    st.subheader("Exploratory Cross-Tabulation Summaries")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**1. Metrics by Age Group**")
        if 'Age_Group' in df_filtered.columns:
            age_summary = df_filtered.groupby('Age_Group')[['Length_of_Stay', 'Total_Charges']].agg(['count', 'mean']).round(2)
            st.dataframe(age_summary, use_container_width=True)
        else:
            st.info("Age_Group field not present.")

    with col_b:
        st.markdown("**2. Severity vs Payment Type Breakdown**")
        pivot_sev = df_filtered.pivot_table(
            index='Severity_Level',
            columns='Payment_Type',
            values='Length_of_Stay',
            aggfunc='count',
            fill_value=0
        )
        st.dataframe(pivot_sev, use_container_width=True)

with tab3:
    st.subheader("SPARCS Dataset Explorer")
    st.dataframe(df_filtered.head(500), use_container_width=True)
    st.download_button(
        label="📥 Download Filtered SPARCS Dataset (CSV)",
        data=df_filtered.to_csv(index=False).encode('utf-8'),
        file_name="SPARCS_2021_Filtered_Discharges.csv",
        mime="text/csv"
    )
