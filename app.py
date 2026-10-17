
import os
import re
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="MediScope | SPARCS Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_FILE = os.path.join(
    BASE_DIR,
    "Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv",
)

NEON_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 0%, rgba(0, 245, 212, 0.09), transparent 27%),
        radial-gradient(circle at 90% 10%, rgba(123, 97, 255, 0.12), transparent 28%),
        linear-gradient(135deg, #070b16 0%, #0b1020 50%, #080d19 100%);
    color: #e5edff;
}

[data-testid="stHeader"] {
    background: rgba(7, 11, 22, 0.78);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1222 0%, #10172b 100%);
    border-right: 1px solid rgba(0, 245, 212, 0.22);
}

[data-testid="stSidebar"] * {
    color: #e5edff;
}

.block-container {
    padding-top: 1.6rem;
    padding-bottom: 2.5rem;
    max-width: 1600px;
}

.hero {
    padding: 1.7rem 1.9rem;
    border-radius: 22px;
    background: linear-gradient(120deg, rgba(0, 245, 212, 0.10), rgba(123, 97, 255, 0.13));
    border: 1px solid rgba(0, 245, 212, 0.28);
    box-shadow: 0 0 28px rgba(0, 245, 212, 0.07);
    animation: appear 0.8s ease-out;
}

.hero-title {
    color: #ffffff;
    font-size: clamp(1.8rem, 3vw, 2.7rem);
    font-weight: 800;
    letter-spacing: -1px;
    margin: 0;
}

.hero-title span {
    color: #00f5d4;
    text-shadow: 0 0 18px rgba(0, 245, 212, 0.35);
}

.hero-subtitle {
    color: #a8b7d8;
    font-size: 1rem;
    margin-top: 0.55rem;
}

.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #e5edff;
    margin-top: 0.3rem;
}

.metric-card {
    background: linear-gradient(145deg, rgba(20, 31, 55, 0.97), rgba(12, 19, 36, 0.97));
    border: 1px solid rgba(105, 130, 190, 0.23);
    border-radius: 17px;
    padding: 1.05rem 1.1rem;
    min-height: 112px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.13);
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    animation: appear 0.65s ease-out;
}

.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(0, 245, 212, 0.55);
    box-shadow: 0 0 22px rgba(0, 245, 212, 0.09);
}

.metric-label {
    color: #9eafd2;
    font-size: 0.83rem;
    font-weight: 500;
}

.metric-value {
    color: #ffffff;
    font-size: clamp(1.3rem, 2vw, 1.85rem);
    font-weight: 800;
    margin-top: 0.45rem;
    overflow-wrap: anywhere;
}

.metric-accent {
    color: #00f5d4;
}

div[data-testid="stTabs"] button {
    border-radius: 10px 10px 0 0;
    font-weight: 600;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00f5d4;
    border-bottom-color: #00f5d4;
}

div[data-testid="stPlotlyChart"] {
    background: rgba(13, 20, 37, 0.72);
    border: 1px solid rgba(105, 130, 190, 0.16);
    border-radius: 15px;
    padding: 0.3rem;
}

div[data-testid="stDataFrame"] {
    border: 1px solid rgba(105, 130, 190, 0.2);
    border-radius: 12px;
    overflow: hidden;
}

.stButton button, .stDownloadButton button {
    border-radius: 10px;
    border: 1px solid rgba(0, 245, 212, 0.4);
    background: linear-gradient(110deg, rgba(0, 245, 212, 0.13), rgba(123, 97, 255, 0.15));
    color: #efffff;
    transition: all 0.2s ease;
}

.stButton button:hover, .stDownloadButton button:hover {
    border-color: #00f5d4;
    box-shadow: 0 0 15px rgba(0, 245, 212, 0.12);
    color: #ffffff;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

hr {
    border-color: rgba(120, 145, 195, 0.18);
}

@keyframes appear {
    from { opacity: 0; transform: translateY(7px); }
    to { opacity: 1; transform: translateY(0); }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}
</style>
"""

st.markdown(NEON_CSS, unsafe_allow_html=True)

PLOT_TEMPLATE = "plotly_dark"
PLOT_BG = "rgba(13, 20, 37, 0.0)"
PLOT_PAPER = "rgba(13, 20, 37, 0.0)"
NEON_PALETTE = [
    "#00F5D4",
    "#7B61FF",
    "#FF6B9D",
    "#FFD166",
    "#5AA9FF",
    "#B8F35B",
    "#FF9F43",
    "#C792EA",
]


def normalize_name(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def match_exact_column(columns, aliases):
    normalized = {normalize_name(column): column for column in columns}
    for alias in aliases:
        found = normalized.get(normalize_name(alias))
        if found is not None:
            return found
    return None


def clean_numeric(series):
    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def apply_standard_categories(df):
    category_columns = [
        "Facility_Name",
        "County",
        "Diagnosis_Category",
        "Severity_Level",
        "Payment_Type",
        "Age_Group",
        "Gender",
        "Race",
    ]

    for column in category_columns:
        if column not in df.columns:
            df[column] = "Unknown"
        else:
            values = df[column].astype("string").str.strip()
            values = values.replace("", pd.NA)
            df[column] = values.fillna("Unknown").astype(str)

    df["Gender"] = df["Gender"].replace(
        {
            "M": "Male",
            "F": "Female",
            "MALE": "Male",
            "FEMALE": "Female",
        }
    )

    df["Payment_Type"] = df["Payment_Type"].replace(
        {
            "Private Health Insurance": "Private Insurance",
            "Blue Cross": "Private Insurance",
        }
    )

    return df


def iqr_outlier_count(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 4:
        return 0
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((values < lower) | (values > upper)).sum())


@st.cache_data(ttl=3600, show_spinner="Reading and preprocessing the SPARCS dataset...")
def load_sparcs_data(file_path, mode, sample_limit):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Dataset not found: {file_path}. Place the CSV in the same folder as app.py."
        )

    chunk_size = 100_000

    if mode == "Sample":
        raw = pd.read_csv(
            file_path,
            nrows=int(sample_limit),
            low_memory=False,
        )
        rows_read = len(raw)
        total_file_rows_read = False
    else:
        chunks = []
        rows_read = 0
        for chunk in pd.read_csv(
            file_path,
            chunksize=chunk_size,
            low_memory=False,
        ):
            chunks.append(chunk)
            rows_read += len(chunk)
        if not chunks:
            raise ValueError("The CSV file contains no rows.")
        raw = pd.concat(chunks, ignore_index=True)
        total_file_rows_read = True

    raw.columns = [str(column).strip() for column in raw.columns]
    raw = raw.loc[:, ~pd.Index(raw.columns).duplicated()].copy()

    original_columns = list(raw.columns)

    aliases = {
        "Length_of_Stay": [
            "Length of Stay",
            "Length_of_Stay",
        ],
        "Total_Charges": [
            "Total Charges",
            "Total_Charges",
        ],
        "Facility_Name": [
            "Facility Name",
            "Facility_Name",
        ],
        "County": [
            "Hospital County",
            "Facility County",
            "County",
        ],
        "Diagnosis_Category": [
            "APR MDC Description",
            "MDC Description",
            "APR DRG Description",
            "CCS Diagnosis Description",
            "Diagnosis Category",
        ],
        "Severity_Level": [
            "APR Severity of Illness Description",
            "Severity of Illness Description",
            "Severity Level",
        ],
        "Payment_Type": [
            "Payment Typology 1",
            "Payment Type",
        ],
        "Age_Group": [
            "Age Group",
            "Age_Group",
        ],
        "Gender": [
            "Gender",
        ],
        "Race": [
            "Race",
        ],
        "Discharge_Year": [
            "Discharge Year",
            "Discharge_Year",
        ],
        "Discharge_Date": [
            "Discharge Date",
            "Discharge_Date",
        ],
    }

    mapping = {}
    used_columns = set()

    for target, possible_names in aliases.items():
        source = match_exact_column(
            [column for column in original_columns if column not in used_columns],
            possible_names,
        )
        if source is not None:
            mapping[source] = target
            used_columns.add(source)

    df = raw.rename(columns=mapping).copy()

    if "Length_of_Stay" not in df.columns:
        raise ValueError(
            "The required Length of Stay column could not be found. "
            "Check the exact column headers in the source CSV."
        )

    report = {
        "rows_read": int(rows_read),
        "full_dataset_mode": bool(total_file_rows_read),
        "raw_rows": int(len(df)),
        "invalid_los_removed": 0,
        "missing_charge_count": 0,
        "missing_year_count": 0,
        "outside_2021_count": 0,
        "year_filter_applied": False,
        "duplicate_rows": int(df.duplicated().sum()),
        "available_source_columns": original_columns,
        "matched_columns": mapping,
        "year_validation": "No year or discharge date column was detected.",
    }

    df["Length_of_Stay"] = clean_numeric(df["Length_of_Stay"])
    valid_los = df["Length_of_Stay"].notna() & (df["Length_of_Stay"] > 0)
    report["invalid_los_removed"] = int((~valid_los).sum())
    df = df.loc[valid_los].copy()

    if "Total_Charges" in df.columns:
        df["Total_Charges"] = clean_numeric(df["Total_Charges"])
    else:
        df["Total_Charges"] = np.nan

    report["missing_charge_count"] = int(df["Total_Charges"].isna().sum())

    if "Discharge_Year" in df.columns:
        year_values = clean_numeric(df["Discharge_Year"])
        missing_year = year_values.isna()
        outside_year = year_values.notna() & (year_values != 2021)

        report["missing_year_count"] = int(missing_year.sum())
        report["outside_2021_count"] = int(outside_year.sum())
        report["year_filter_applied"] = True

        keep_year = year_values == 2021
        df = df.loc[keep_year].copy()
        report["year_validation"] = (
            "Discharge Year was detected. Rows with missing years or years other than "
            "2021 were excluded from the analysis."
        )

    elif "Discharge_Date" in df.columns:
        parsed_dates = pd.to_datetime(df["Discharge_Date"], errors="coerce")
        missing_year = parsed_dates.isna()
        outside_year = parsed_dates.notna() & (parsed_dates.dt.year != 2021)

        report["missing_year_count"] = int(missing_year.sum())
        report["outside_2021_count"] = int(outside_year.sum())
        report["year_filter_applied"] = True

        keep_year = parsed_dates.notna() & (parsed_dates.dt.year == 2021)
        df = df.loc[keep_year].copy()
        report["year_validation"] = (
            "Discharge Date was detected. Missing or unparseable dates and dates outside "
            "2021 were excluded from the analysis."
        )

    else:
        report["year_validation"] = (
            "No exact Discharge Year or Discharge Date column was found. "
            "The file name indicates 2021, but the year could not be independently "
            "validated from a date field."
        )

    df = apply_standard_categories(df)
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No valid rows remain after preprocessing. Check the source columns and values."
        )

    report["clean_rows"] = int(len(df))
    report["rows_removed_total"] = int(report["raw_rows"] - report["clean_rows"])

    return df, report


def make_plot(fig, height=420):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor=PLOT_PAPER,
        plot_bgcolor=PLOT_BG,
        font=dict(color="#DFE8FF", family="Inter, sans-serif"),
        margin=dict(l=15, r=15, t=55, b=15),
        height=height,
        hoverlabel=dict(
            bgcolor="#111A2E",
            font_color="#FFFFFF",
        ),
    )
    return fig


def metric_card(label, value, accent=False):
    accent_class = " metric-accent" if accent else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value{accent_class}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_money(value):
    if pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"


def get_filtered_data(df):
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dashboard Filters")

    county_values = sorted(df["County"].dropna().unique().tolist())
    severity_values = sorted(df["Severity_Level"].dropna().unique().tolist())
    payer_values = sorted(df["Payment_Type"].dropna().unique().tolist())
    gender_values = sorted(df["Gender"].dropna().unique().tolist())
    age_values = sorted(df["Age_Group"].dropna().unique().tolist())

    selected_counties = st.sidebar.multiselect(
        "Hospital County",
        county_values,
        default=county_values,
    )
    selected_severity = st.sidebar.multiselect(
        "Severity Level",
        severity_values,
        default=severity_values,
    )
    selected_payers = st.sidebar.multiselect(
        "Payment Type",
        payer_values,
        default=payer_values,
    )
    selected_gender = st.sidebar.multiselect(
        "Gender",
        gender_values,
        default=gender_values,
    )
    selected_age = st.sidebar.multiselect(
        "Age Group",
        age_values,
        default=age_values,
    )

    filtered = df[
        df["County"].isin(selected_counties)
        & df["Severity_Level"].isin(selected_severity)
        & df["Payment_Type"].isin(selected_payers)
        & df["Gender"].isin(selected_gender)
        & df["Age_Group"].isin(selected_age)
    ].copy()

    return filtered


def show_quality_report(df_raw, df_filtered, report):
    st.subheader("Data Quality and Preprocessing")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Rows Read", f"{report['rows_read']:,}")
    with c2:
        metric_card("Cleaned Rows", f"{len(df_raw):,}", True)
    with c3:
        metric_card("Invalid LOS Removed", f"{report['invalid_los_removed']:,}")
    with c4:
        metric_card("Exact Duplicate Rows", f"{report['duplicate_rows']:,}")

    st.markdown("#### Year validation")
    st.info(report["year_validation"])

    year_col1, year_col2 = st.columns(2)
    with year_col1:
        st.metric("Rows with missing or invalid year/date", f"{report['missing_year_count']:,}")
    with year_col2:
        st.metric("Rows outside 2021", f"{report['outside_2021_count']:,}")

    if not report["year_filter_applied"]:
        st.warning(
            "Year filtering was not applied because the source did not contain a recognized "
            "year or date field. Confirm the dataset's reporting period before presenting "
            "the results as independently year-validated."
        )

    st.markdown("#### Missing values after preprocessing")

    missing = (
        df_raw.isna()
        .sum()
        .rename("Missing Values")
        .to_frame()
        .reset_index(names="Column")
    )
    missing = missing[missing["Missing Values"] > 0].sort_values(
        "Missing Values", ascending=False
    )

    if missing.empty:
        st.success("No missing values were detected in the loaded analysis columns.")
    else:
        st.dataframe(missing, use_container_width=True, hide_index=True)

    st.markdown("#### Charge outliers")

    raw_outliers = iqr_outlier_count(df_raw["Total_Charges"])
    filtered_outliers = iqr_outlier_count(df_filtered["Total_Charges"])

    outlier_col1, outlier_col2 = st.columns(2)
    with outlier_col1:
        st.metric("Loaded cleaned data", f"{raw_outliers:,}")
    with outlier_col2:
        st.metric("Currently filtered data", f"{filtered_outliers:,}")

    st.caption(
        "Outliers are identified with the 1.5 × IQR rule. They are counted for reporting "
        "and are not automatically deleted from the dataset."
    )

    st.markdown("#### Source-to-dashboard column mapping")

    mapping_rows = [
        {"Dashboard field": target, "Matched source field": source}
        for source, target in report["matched_columns"].items()
    ]
    mapping_table = pd.DataFrame(mapping_rows)

    if mapping_table.empty:
        st.warning("No source columns were mapped.")
    else:
        st.dataframe(mapping_table, use_container_width=True, hide_index=True)

    with st.expander("View source column headers"):
        st.write(report["available_source_columns"])


def run_ai_experiment(df):
    st.subheader("AI Experiment: Longer-than-Typical Stay Classification")

    st.write(
        "This exploratory model predicts whether a discharge record's length of stay is "
        "above the median length of stay calculated from the training split. The threshold "
        "is calculated using training records only, then applied unchanged to the test set. "
        "This is a demonstration of classification, not a clinical prediction tool."
    )

    candidate_features = [
        "Age_Group",
        "Gender",
        "Race",
        "County",
        "Severity_Level",
        "Payment_Type",
        "Diagnosis_Category",
    ]
    features = [
        column
        for column in candidate_features
        if column in df.columns and df[column].nunique(dropna=True) > 1
    ]

    if len(df) < 50:
        st.warning("At least 50 valid filtered records are recommended for this experiment.")
        return

    if len(features) < 2:
        st.warning(
            "There are not enough usable categorical predictors in the filtered data "
            "to run the experiment."
        )
        return

    model_data = df[features + ["Length_of_Stay"]].copy()
    model_data = model_data.dropna(subset=["Length_of_Stay"])

    if len(model_data) < 50:
        st.warning("There are not enough usable records after removing missing target values.")
        return

    X = model_data[features].astype("string").fillna("Unknown")
    stay_values = model_data["Length_of_Stay"].astype(float)

    indices = np.arange(len(model_data))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.25,
        random_state=42,
    )

    train_stay = stay_values.iloc[train_indices]
    test_stay = stay_values.iloc[test_indices]
    threshold = float(train_stay.median())

    y_train = (train_stay > threshold).astype(int)
    y_test = (test_stay > threshold).astype(int)

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        st.warning(
            "The train/test split contains only one target class. Try a larger or less "
            "restricted dataset to evaluate the model."
        )
        return

    X_train = X.iloc[train_indices].copy()
    X_test = X.iloc[test_indices].copy()

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, features),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=160,
                    max_depth=12,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    baseline = DummyClassifier(strategy="most_frequent")

    try:
        model.fit(X_train, y_train)
        baseline.fit(X_train, y_train)
        predictions = model.predict(X_test)
        baseline_predictions = baseline.predict(X_test)
    except Exception as error:
        st.error(f"The AI experiment could not be completed: {error}")
        return

    model_accuracy = accuracy_score(y_test, predictions)
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)

    model_col1, model_col2, model_col3, model_col4 = st.columns(4)

    with model_col1:
        metric_card("Training Records", f"{len(train_indices):,}")
    with model_col2:
        metric_card("Test Records", f"{len(test_indices):,}")
    with model_col3:
        metric_card("Training Median LOS", f"{threshold:.1f} days", True)
    with model_col4:
        metric_card("Model Accuracy", f"{model_accuracy:.1%}", True)

    metrics = pd.DataFrame(
        [
            {
                "Model": "Random Forest",
                "Accuracy": model_accuracy,
                "Precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "Recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "F1 Score": f1_score(
                    y_test, predictions, zero_division=0
                ),
            },
            {
                "Model": "Majority-Class Baseline",
                "Accuracy": baseline_accuracy,
                "Precision": precision_score(
                    y_test, baseline_predictions, zero_division=0
                ),
                "Recall": recall_score(
                    y_test, baseline_predictions, zero_division=0
                ),
                "F1 Score": f1_score(
                    y_test, baseline_predictions, zero_division=0
                ),
            },
        ]
    )

    st.markdown("#### Model compared with a baseline")
    st.dataframe(
        metrics.style.format(
            {
                "Accuracy": "{:.1%}",
                "Precision": "{:.1%}",
                "Recall": "{:.1%}",
                "F1 Score": "{:.1%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "The majority-class baseline always predicts the most frequent class in the training "
        "data. Compare the Random Forest results with this baseline rather than judging "
        "accuracy alone."
    )

    class_counts = pd.DataFrame(
        {
            "Class": ["Typical stay", "Longer-than-typical stay"],
            "Training records": [
                int((y_train == 0).sum()),
                int((y_train == 1).sum()),
            ],
            "Test records": [
                int((y_test == 0).sum()),
                int((y_test == 1).sum()),
            ],
        }
    )

    st.markdown("#### Target class distribution")
    st.dataframe(class_counts, use_container_width=True, hide_index=True)

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    cm_fig = px.imshow(
        cm,
        x=["Predicted typical", "Predicted longer"],
        y=["Actual typical", "Actual longer"],
        text_auto=True,
        color_continuous_scale="Teal",
        labels={"x": "Predicted class", "y": "Actual class", "color": "Records"},
    )
    cm_fig.update_layout(title="Random Forest Confusion Matrix")
    st.plotly_chart(make_plot(cm_fig, 380), use_container_width=True)

    st.markdown("#### Classification report")
    report_text = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Typical stay", "Longer-than-typical stay"],
        zero_division=0,
    )
    st.code(report_text, language="text")

    st.caption(
        "The model uses age group, gender, race, county, severity, payment type, and diagnosis "
        "category where available. Length of stay is used only to construct the target and is "
        "not supplied as a predictor. No causal or clinical conclusions should be drawn."
    )


st.markdown(
    """
    <div class="hero">
        <div class="hero-title">MediScope <span>Health Analytics</span></div>
        <div class="hero-subtitle">
            SPARCS hospital inpatient discharges · Exploratory analysis · Interactive reporting
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

st.sidebar.markdown("## MediScope")
st.sidebar.caption("SPARCS 2021 Inpatient Discharges")
st.sidebar.markdown("---")

if not os.path.isfile(TARGET_FILE):
    st.error(
        "The required dataset was not found. Place this CSV in the same folder as app.py:"
    )
    st.code(os.path.basename(TARGET_FILE), language="text")
    st.stop()

st.sidebar.subheader("Data Settings")

analysis_mode = st.sidebar.radio(
    "Dataset coverage",
    options=["Sample", "Full Dataset"],
    index=0,
    help=(
        "Sample mode reads the first selected number of rows. Full Dataset reads all CSV "
        "chunks and combines them in memory; it can require several gigabytes of RAM."
    ),
)

sample_size = st.sidebar.select_slider(
    "Rows to read in sample mode",
    options=[10_000, 25_000, 50_000, 100_000],
    value=50_000,
    format_func=lambda value: f"{value:,}",
    disabled=analysis_mode == "Full Dataset",
)

if analysis_mode == "Full Dataset":
    st.sidebar.warning(
        "Full Dataset mode may use several gigabytes of memory. If the app stops or runs "
        "out of memory, return to Sample mode."
    )

try:
    with st.spinner("Preparing the dataset..."):
        df_raw, data_report = load_sparcs_data(
            TARGET_FILE,
            analysis_mode,
            int(sample_size),
        )
except Exception as error:
    st.error(f"Could not load or preprocess the dataset: {error}")
    st.stop()

st.sidebar.success(f"{len(df_raw):,} valid records ready")

if analysis_mode == "Sample":
    st.sidebar.caption(
        "Sample mode reads the first selected rows from the CSV. It is not a random sample "
        "and may not represent the entire dataset."
    )
else:
    st.sidebar.caption("All rows were read before preprocessing.")

df_filtered = get_filtered_data(df_raw)

if df_filtered.empty:
    st.warning("No records match the current filters. Change the sidebar selections.")
    st.stop()

st.markdown("")

k1, k2, k3, k4 = st.columns(4)

with k1:
    metric_card("Filtered Discharges", f"{len(df_filtered):,}", True)
with k2:
    avg_los = df_filtered["Length_of_Stay"].mean()
    metric_card(
        "Average Length of Stay",
        f"{avg_los:.1f} days" if pd.notna(avg_los) else "N/A",
    )
with k3:
    total_charges = df_filtered["Total_Charges"].sum(min_count=1)
    metric_card("Total Hospital Charges", format_money(total_charges))
with k4:
    average_charge = df_filtered["Total_Charges"].mean()
    metric_card("Average Charge per Stay", format_money(average_charge))

st.markdown("---")

overview_tab, diagnosis_tab, demographics_tab, facilities_tab, quality_tab, ai_tab, explorer_tab = st.tabs(
    [
        "Overview",
        "Diagnosis & Stay",
        "Charges & Demographics",
        "Facilities & Counties",
        "Data Quality",
        "AI Experiment",
        "Data Explorer",
    ]
)

with overview_tab:
    st.markdown('<div class="section-title">Discharge Overview</div>', unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.subheader("Patient volume by payment type")
        payer_summary = (
            df_filtered["Payment_Type"]
            .value_counts()
            .head(10)
            .rename_axis("Payment Type")
            .reset_index(name="Discharges")
        )

        payer_fig = px.pie(
            payer_summary,
            names="Payment Type",
            values="Discharges",
            hole=0.55,
            color_discrete_sequence=NEON_PALETTE,
        )
        payer_fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
        )
        st.plotly_chart(make_plot(payer_fig, 430), use_container_width=True)

    with right:
        st.subheader("Length-of-stay distribution")
        max_days = st.slider(
            "Maximum stay displayed (days)",
            min_value=5,
            max_value=90,
            value=30,
            step=5,
            key="overview_max_los",
        )
        hist_data = df_filtered[
            df_filtered["Length_of_Stay"] <= max_days
        ]

        if hist_data.empty:
            st.info("No stays fall within the selected range.")
        else:
            hist_fig = px.histogram(
                hist_data,
                x="Length_of_Stay",
                nbins=min(max_days, 40),
                color_discrete_sequence=["#00F5D4"],
                labels={
                    "Length_of_Stay": "Length of Stay (Days)",
                    "count": "Discharges",
                },
            )
            hist_fig.update_layout(bargap=0.08)
            st.plotly_chart(make_plot(hist_fig, 430), use_container_width=True)

    st.subheader("Quick insights")

    longest_diagnosis = (
        df_filtered.groupby("Diagnosis_Category")["Length_of_Stay"]
        .mean()
        .sort_values(ascending=False)
    )
    highest_volume_county = df_filtered["County"].value_counts()
    highest_payment = df_filtered["Payment_Type"].value_counts()

    insight_cols = st.columns(3)

    with insight_cols[0]:
        if not longest_diagnosis.empty:
            st.markdown("**Highest average stay**")
            st.write(longest_diagnosis.index[0])
            st.caption(f"{longest_diagnosis.iloc[0]:.1f} days on average")

    with insight_cols[1]:
        if not highest_volume_county.empty:
            st.markdown("**Highest discharge volume by county**")
            st.write(highest_volume_county.index[0])
            st.caption(f"{highest_volume_county.iloc[0]:,} discharges in the filtered data")

    with insight_cols[2]:
        if not highest_payment.empty:
            st.markdown("**Most common payment type**")
            st.write(highest_payment.index[0])
            st.caption(f"{highest_payment.iloc[0]:,} discharges in the filtered data")

    st.caption(
        "These insights describe the loaded and filtered records only. In Sample mode, "
        "they are based on the first selected rows rather than the complete CSV."
    )

with diagnosis_tab:
    st.markdown('<div class="section-title">Diagnosis and Length of Stay</div>', unsafe_allow_html=True)

    top_n = st.slider(
        "Number of diagnosis categories to display",
        min_value=5,
        max_value=25,
        value=10,
        key="diagnosis_top_n",
    )

    diagnosis_summary = (
        df_filtered.groupby("Diagnosis_Category", dropna=False)
        .agg(
            Discharges=("Length_of_Stay", "size"),
            Average_Stay=("Length_of_Stay", "mean"),
            Median_Stay=("Length_of_Stay", "median"),
        )
        .reset_index()
        .sort_values("Average_Stay", ascending=False)
        .head(top_n)
        .sort_values("Average_Stay")
    )

    diag_fig = px.bar(
        diagnosis_summary,
        x="Average_Stay",
        y="Diagnosis_Category",
        orientation="h",
        color="Average_Stay",
        color_continuous_scale=["#27345B", "#7B61FF", "#00F5D4"],
        hover_data=["Discharges", "Median_Stay"],
        labels={
            "Average_Stay": "Average Stay (Days)",
            "Diagnosis_Category": "Diagnosis Category",
            "Discharges": "Discharges",
            "Median_Stay": "Median Stay (Days)",
        },
        text_auto=".1f",
    )
    diag_fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(make_plot(diag_fig, 520), use_container_width=True)

    st.subheader("Diagnosis summary table")
    st.dataframe(
        diagnosis_summary.round(2),
        use_container_width=True,
        hide_index=True,
    )

    severity_stay = (
        df_filtered.groupby("Severity_Level")
        .agg(
            Discharges=("Length_of_Stay", "size"),
            Average_Stay=("Length_of_Stay", "mean"),
            Median_Stay=("Length_of_Stay", "median"),
        )
        .reset_index()
        .sort_values("Average_Stay", ascending=False)
    )

    st.subheader("Length of stay by severity")
    severity_fig = px.bar(
        severity_stay,
        x="Severity_Level",
        y="Average_Stay",
        color="Severity_Level",
        color_discrete_sequence=NEON_PALETTE,
        hover_data=["Discharges", "Median_Stay"],
        labels={
            "Severity_Level": "Severity Level",
            "Average_Stay": "Average Stay (Days)",
        },
    )
    st.plotly_chart(make_plot(severity_fig, 390), use_container_width=True)

with demographics_tab:
    st.markdown('<div class="section-title">Charges and Demographics</div>', unsafe_allow_html=True)

    hide_charge_outliers = st.checkbox(
        "Hide extreme charge values in charge charts",
        value=True,
        help=(
            "This affects the charge box plot and average-charge-by-demographic chart. "
            "It does not delete rows from the dataset or change other dashboard metrics."
        ),
    )

    charge_data = df_filtered.dropna(subset=["Total_Charges"]).copy()

    if hide_charge_outliers and not charge_data.empty:
        q1 = charge_data["Total_Charges"].quantile(0.25)
        q3 = charge_data["Total_Charges"].quantile(0.75)
        iqr = q3 - q1
        lower_limit = max(0, q1 - 1.5 * iqr)
        upper_limit = q3 + 1.5 * iqr
        charge_chart_data = charge_data[
            charge_data["Total_Charges"].between(lower_limit, upper_limit)
        ].copy()
    else:
        charge_chart_data = charge_data.copy()

    if not charge_data.empty and not charge_chart_data.empty:
        st.caption(
            f"{len(charge_chart_data):,} of {len(charge_data):,} records with valid charges "
            "are displayed in the charge charts."
        )

    left, right = st.columns(2)

    with left:
        st.subheader("Hospital charges by severity")
        if charge_chart_data.empty:
            st.info("No valid charge values are available for the selected filters.")
        else:
            box_fig = px.box(
                charge_chart_data,
                x="Severity_Level",
                y="Total_Charges",
                color="Severity_Level",
                color_discrete_sequence=NEON_PALETTE,
                points="outliers" if not hide_charge_outliers else False,
                labels={
                    "Severity_Level": "Severity Level",
                    "Total_Charges": "Total Charges ($)",
                },
            )
            st.plotly_chart(make_plot(box_fig, 440), use_container_width=True)

    with right:
        st.subheader("Average charge by age group")
        age_charge = (
            charge_chart_data.groupby("Age_Group")
            .agg(
                Average_Charges=("Total_Charges", "mean"),
                Discharges=("Total_Charges", "size"),
            )
            .reset_index()
            .sort_values("Average_Charges", ascending=False)
        )

        if age_charge.empty:
            st.info("No valid charges are available for the selected filters.")
        else:
            age_fig = px.bar(
                age_charge,
                x="Age_Group",
                y="Average_Charges",
                color="Average_Charges",
                color_continuous_scale=["#27345B", "#7B61FF", "#00F5D4"],
                hover_data=["Discharges"],
                labels={
                    "Age_Group": "Age Group",
                    "Average_Charges": "Average Charges ($)",
                },
            )
            age_fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(make_plot(age_fig, 440), use_container_width=True)

    st.subheader("Discharges and average stay by gender")
    gender_summary = (
        df_filtered.groupby("Gender")
        .agg(
            Discharges=("Length_of_Stay", "size"),
            Average_Stay=("Length_of_Stay", "mean"),
            Average_Charges=("Total_Charges", "mean"),
        )
        .reset_index()
        .sort_values("Discharges", ascending=False)
    )

    gender_col1, gender_col2 = st.columns(2)

    with gender_col1:
        gender_volume_fig = px.bar(
            gender_summary,
            x="Gender",
            y="Discharges",
            color="Gender",
            color_discrete_sequence=NEON_PALETTE,
        )
        st.plotly_chart(make_plot(gender_volume_fig, 360), use_container_width=True)

    with gender_col2:
        gender_stay_fig = px.bar(
            gender_summary,
            x="Gender",
            y="Average_Stay",
            color="Gender",
            color_discrete_sequence=NEON_PALETTE,
            labels={"Average_Stay": "Average Stay (Days)"},
        )
        st.plotly_chart(make_plot(gender_stay_fig, 360), use_container_width=True)

    st.dataframe(gender_summary.round(2), use_container_width=True, hide_index=True)

with facilities_tab:
    st.markdown('<div class="section-title">Facility and County Performance</div>', unsafe_allow_html=True)

    facility_count = (
        df_filtered.groupby(["Facility_Name", "County"])
        .agg(
            Discharges=("Length_of_Stay", "size"),
            Average_Stay=("Length_of_Stay", "mean"),
            Average_Charges=("Total_Charges", "mean"),
        )
        .reset_index()
    )

    top_facility_count = st.slider(
        "Number of facilities to display",
        min_value=5,
        max_value=30,
        value=10,
        key="facility_top_n",
    )

    top_facilities = (
        facility_count.sort_values("Discharges", ascending=False)
        .head(top_facility_count)
    )

    st.subheader("Facilities with the highest discharge volume")
    if top_facilities.empty:
        st.info("No facility data is available for the current filters.")
    else:
        facility_fig = px.bar(
            top_facilities.sort_values("Discharges"),
            x="Discharges",
            y="Facility_Name",
            color="Average_Stay",
            orientation="h",
            color_continuous_scale=["#27345B", "#7B61FF", "#00F5D4"],
            hover_data=["County", "Average_Charges"],
            labels={
                "Facility_Name": "Facility",
                "Discharges": "Discharges",
                "Average_Stay": "Average Stay (Days)",
                "Average_Charges": "Average Charges ($)",
            },
        )
        st.plotly_chart(make_plot(facility_fig, 520), use_container_width=True)

    st.subheader("Average stay by facility and county heatmap")
    facility_heatmap_rows = (
        df_filtered.groupby(["Facility_Name", "County"])
        .agg(
            Average_Stay=("Length_of_Stay", "mean"),
            Discharges=("Length_of_Stay", "size"),
        )
        .reset_index()
        .sort_values("Discharges", ascending=False)
        .head(3000)
    )

    if facility_heatmap_rows.empty:
        st.info("There is not enough facility and county data to create a heatmap.")
    else:
        heatmap_facilities = (
            facility_heatmap_rows.groupby("Facility_Name")["Discharges"]
            .sum()
            .nlargest(min(12, facility_heatmap_rows["Facility_Name"].nunique()))
            .index
        )

        heatmap_data = facility_heatmap_rows[
            facility_heatmap_rows["Facility_Name"].isin(heatmap_facilities)
        ].pivot(
            index="Facility_Name",
            columns="County",
            values="Average_Stay",
        )

        heat_fig = px.imshow(
            heatmap_data,
            labels={
                "x": "County",
                "y": "Facility",
                "color": "Average Stay (Days)",
            },
            color_continuous_scale=["#17213C", "#7B61FF", "#00F5D4"],
            aspect="auto",
        )
        st.plotly_chart(make_plot(heat_fig, 500), use_container_width=True)

    st.subheader("County-level summary")
    county_summary = (
        df_filtered.groupby("County")
        .agg(
            Discharges=("Length_of_Stay", "size"),
            Average_Stay=("Length_of_Stay", "mean"),
            Median_Stay=("Length_of_Stay", "median"),
            Average_Charges=("Total_Charges", "mean"),
        )
        .reset_index()
        .sort_values("Discharges", ascending=False)
    )

    st.dataframe(
        county_summary.round(2),
        use_container_width=True,
        hide_index=True,
    )

with quality_tab:
    show_quality_report(df_raw, df_filtered, data_report)

with ai_tab:
    run_ai_experiment(df_filtered)

with explorer_tab:
    st.markdown('<div class="section-title">SPARCS Data Explorer</div>', unsafe_allow_html=True)

    st.caption(
        "Previewing up to 500 filtered rows. The download button exports the filtered "
        "records from the current analysis mode."
    )

    st.dataframe(
        df_filtered.head(500),
        use_container_width=True,
        hide_index=True,
    )

    summary_download = pd.DataFrame(
        [
            {
                "Metric": "Rows read",
                "Value": data_report["rows_read"],
            },
            {
                "Metric": "Valid rows after preprocessing",
                "Value": len(df_raw),
            },
            {
                "Metric": "Rows in current filters",
                "Value": len(df_filtered),
            },
            {
                "Metric": "Average length of stay",
                "Value": df_filtered["Length_of_Stay"].mean(),
            },
            {
                "Metric": "Average total charges",
                "Value": df_filtered["Total_Charges"].mean(),
            },
            {
                "Metric": "Invalid LOS rows removed",
                "Value": data_report["invalid_los_removed"],
            },
            {
                "Metric": "Duplicate rows detected",
                "Value": data_report["duplicate_rows"],
            },
            {
                "Metric": "Missing year/date rows",
                "Value": data_report["missing_year_count"],
            },
            {
                "Metric": "Rows outside 2021",
                "Value": data_report["outside_2021_count"],
            },
        ]
    )

    download_col1, download_col2 = st.columns(2)

    with download_col1:
        st.download_button(
            "Download Filtered Records",
            data=df_filtered.to_csv(index=False).encode("utf-8"),
            file_name="SPARCS_2021_Filtered_Discharges.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with download_col2:
        st.download_button(
            "Download Analysis Summary",
            data=summary_download.to_csv(index=False).encode("utf-8"),
            file_name="SPARCS_2021_Analysis_Summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.markdown("---")
st.caption(
    "MediScope Health Analytics · SPARCS inpatient discharge exploration. "
    "Results describe the data loaded into this dashboard and should not be interpreted "
    "as clinical recommendations or causal findings."
)
