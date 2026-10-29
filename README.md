## 1000411-sarveshwaran-Hospital-Inpatient-Discharges
# MediScope – Hospital Inpatient Discharges Dashboard

## Project Overview

MediScope is an interactive data analytics dashboard developed for the CRS (Artificial Intelligence) Summative Assessment. It analyses the **2021 Hospital Inpatient Discharges dataset from the Statewide Planning and Research Cooperative System (SPARCS), New York State**.

The dashboard transforms hospital discharge data into interactive visualizations and summaries to help users explore hospital stays, healthcare charges, patient demographics, diagnoses, and hospital performance.

## Problem Statement

Hospital inpatient discharge datasets contain large amounts of information that can be difficult to interpret using raw tables alone. Without effective visualization and filtering tools, it can be challenging to identify patterns in length of stay, healthcare costs, patient demographics, and hospital utilization.

MediScope addresses this problem by providing an interactive, user-centered dashboard that makes hospital discharge data easier to explore and understand.

## Project Objectives

- Clean and preprocess hospital discharge data.
- Explore relationships between patient demographics, diagnoses, length of stay, and healthcare charges.
- Identify patterns in hospital utilization and discharge records.
- Present findings through interactive charts and visualizations.
- Provide filters and data exploration tools for user-centered analysis.
- Experiment with machine learning to classify hospital discharge severity.

## Dataset

**Dataset name:** Hospital Inpatient Discharges – 2021

**Source:** New York State Department of Health, SPARCS.

The dataset contains hospital inpatient discharge records and associated information, such as patient demographics, diagnoses, length of stay, hospital charges, severity, payment information, and facility details.

The original dataset is large. To reduce memory usage and improve dashboard performance, the application loads a sample of up to 250,000 rows.

### Dataset Setup

The original CSV file is not included in this repository by default because of its large size.

1. Obtain the 2021 Hospital Inpatient Discharges dataset from the official New York State Health Department website.
2. Place the CSV file in the same directory as `app.py`, or inside a `data/` or `dataset/` folder.
3. Use the following filename:

`Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv`

Ensure that the filename matches the one expected by the application.

## Technologies Used

- **Python** – Data processing and application logic.
- **Streamlit** – Interactive web dashboard.
- **Pandas** – Data loading, cleaning, and analysis.
- **NumPy** – Numerical operations.
- **Plotly Express** – Interactive charts and visualizations.
- **Scikit-learn** – Machine learning classification experiment.
- **Git and GitHub** – Version control and project hosting.
- **Streamlit Community Cloud** – Dashboard deployment, if configured.

## Data Preprocessing

The application performs preprocessing steps to improve data quality and consistency.

- Standardizes column names for easier processing.
- Converts length-of-stay and healthcare-charge fields into numeric values.
- Removes records with invalid or non-positive lengths of stay.
- Excludes records with negative healthcare charges.
- Standardizes patient demographic categories.
- Handles missing values in relevant fields.
- Groups diagnosis, severity, payment, facility, and county information for analysis.
- Limits extreme length-of-stay values to reduce the effect of unusually large values on visualizations.

## Dashboard Features

### 1. Overview
- Summary metrics for discharge records.
- Patient age distribution.
- Payment-type distribution.
- Length-of-stay distribution.
- Interactive filtering options.

### 2. Diagnosis and Length of Stay
- Average length of stay by diagnosis.
- Length-of-stay comparisons across severity categories.
- Interactive charts for examining hospitalization patterns.

### 3. Healthcare Charges and Demographics
- Healthcare charges by patient age group and gender.
- Charge comparisons across severity categories.
- Distribution of charges using box plots and violin plots.

### 4. Facilities and Counties
- Hospital facility comparisons.
- Discharge volume by facility.
- Average length of stay by facility.
- Facility and county heatmap analysis.

### 5. Data Quality
- Missing-value analysis.
- Overview of data quality and selected preprocessing results.

### 6. AI Experiment
- Machine learning experiment for predicting discharge severity categories.
- Classification performance evaluation.
- Confusion matrix visualization.

The AI experiment is exploratory and should not be used for clinical decisions or patient treatment.

### 7. Data Explorer
- Search and filter discharge records.
- Inspect selected data fields.
- Download filtered records as a CSV file.

## Research Questions

The dashboard supports investigation of the following questions:

1. How does the average length of hospital stay vary across diagnoses?
2. How does length of stay differ between severity categories?
3. How do healthcare charges vary across severity categories?
4. What patterns exist between patient age groups, gender, and healthcare charges?
5. Which facilities have the highest discharge volumes?
6. How does average length of stay vary across facilities and counties?
7. What is the distribution of hospital discharges across payment types?
8. Can a machine learning model classify discharge severity using selected patient and hospital features?

## Installation and Local Setup

### Prerequisites

- Python 3.10 or later.
- Git.
- The hospital discharge CSV dataset.
- A compatible web browser.

### Step 1: Clone the Repository

```bash
git clone https://github.com/sarveshwaran777333/1000411-sarveshwaran-Hospital-Inpatient-Discharges.git
cd 1000411-sarveshwaran-Hospital-Inpatient-Discharges
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Add the Dataset

Place the CSV dataset beside `app.py` or in the supported `data/` or `dataset/` directory.

### Step 5: Run the Application

```bash
streamlit run app.py
```

Streamlit will provide a local URL, usually:

`http://localhost:8501`

Open this URL in your browser to use the dashboard.

## Deployment

The application can be deployed using Streamlit Community Cloud, provided the repository and dataset are accessible to the deployed application.

1. Push the project code to GitHub.
2. Confirm that `app.py` and `requirements.txt` are in the repository.
3. Open [Streamlit Community Cloud](https://share.streamlit.io/).
4. Sign in and create a new application.
5. Select the GitHub repository and the `app.py` entry point.
6. Configure any required secrets or settings.
7. Deploy and test the application.

**Important:** The dataset is large, so deployment requires an appropriate data-hosting strategy. Do not assume the app will work on the cloud unless the CSV is available at runtime and the hosting environment has sufficient memory and storage. Avoid committing large datasets directly to GitHub unless the repository's storage and hosting arrangements support them.

## Repository Structure

```text
1000411-sarveshwaran-Hospital-Inpatient-Discharges/
│
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv
```

The `data/` folder and CSV shown above are optional examples of the local arrangement. The CSV may be stored elsewhere if the application can locate it.

## Limitations

- The dashboard uses a sample of up to 250,000 rows rather than necessarily analysing every record in the original dataset.
- Results may differ from full-dataset statistics.
- Missing or inconsistent source data may affect the accuracy of some analyses.
- Machine learning results depend on the selected features, data quality, and class distribution.
- The severity classification experiment is educational and is not a validated medical prediction system.
- The dashboard is intended for data exploration and educational analysis, not clinical decision-making.

## Future Improvements

- Add support for processing the complete dataset efficiently.
- Improve data validation and preprocessing reports.
- Add more diagnosis and hospital performance comparisons.
- Evaluate machine learning models using additional metrics and validation techniques.
- Improve deployment reliability for large datasets.
- Add clearer explanations of key findings for non-technical users.

## Conclusion

MediScope demonstrates how Python-based data processing, interactive visualization, and machine learning can be combined to explore a large healthcare dataset. By organizing hospital discharge records into meaningful charts and filters, the dashboard helps users investigate patterns in hospital stays, healthcare charges, demographics, diagnoses, and facility utilization.

## Project Information

- **Project title:** Developing User-Centered Dashboards to Solve Real-World Problems
- **Dashboard name:** MediScope
- **CRS:** Artificial Intelligence
- **Course:** IBCP Career-related Studies (CRS)
- **GitHub repository:** https://github.com/sarveshwaran777333/1000411-sarveshwaran-Hospital-Inpatient-Discharges
- **Live dashboard:** Add the deployed Streamlit URL here after deployment.
- **Student name:** Add your full name.
- **Candidate registration number:** Add your candidate registration number.
- **School name:** Add your official school name.

---

*Developed as part of the IBCP Artificial Intelligence Summative Assessment.*
