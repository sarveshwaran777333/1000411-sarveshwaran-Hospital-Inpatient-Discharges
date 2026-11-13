## 1000411-sarveshwaran-Hospital-Inpatient-Discharges

# 🏥 MediScope

## Smart Hospital Inpatient Discharges Analytics Dashboard

MediScope is an interactive data analytics dashboard designed to analyze hospital inpatient discharge records.

The system uses **Python, Pandas, NumPy, Plotly, Scikit-learn, and Streamlit** to process hospital discharge data and present meaningful insights through interactive visualizations, statistical analysis, filters, and downloadable datasets.

The dashboard helps users understand how factors such as **patient age, gender, race, diagnosis, severity, payment type, hospital facility, and county** are associated with hospital length of stay and healthcare charges.

---

## 📌 Project Overview

Hospital inpatient care generates large amounts of data related to patient demographics, diagnoses, hospital stays, healthcare charges, and discharge information.

Analyzing this information using raw datasets can be difficult because of the volume of records and the relationships between different variables.

MediScope addresses this problem by transforming hospital discharge records into an interactive dashboard.

The system provides:

- Total number of inpatient discharge records
- Average length of hospital stay
- Average healthcare charges
- Patient age distribution
- Patient distribution by payment type
- Length-of-stay distribution
- Average length of stay by diagnosis
- Length-of-stay comparison by severity
- Healthcare charges by severity
- Healthcare charges by age group and gender
- Hospital facility discharge analysis
- Facility and county heatmap analysis
- Statistical data exploration
- Interactive filters
- Data quality analysis
- Machine learning classification experiment
- Raw dataset viewing
- Filtered dataset download

---

## 🎯 Project Objective

The main objective of MediScope is to develop an interactive, user-centered data analytics dashboard that can be used to investigate patterns in hospital inpatient discharge records.

The project focuses on:

1. Cleaning and preparing the hospital discharge dataset.
2. Exploring patient demographics and hospital discharge information.
3. Investigating relationships between diagnoses and length of stay.
4. Comparing healthcare charges across different patient groups and severity categories.
5. Identifying hospital facilities with high discharge volumes.
6. Visualizing hospital discharge patterns through interactive charts.
7. Providing an easy-to-use dashboard for healthcare data exploration.
8. Experimenting with machine learning to classify discharge severity.
9. Allowing users to download filtered data for further analysis.

---

## 📊 Dataset

The application uses the following dataset:

**Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv**

The dataset is based on the 2021 Hospital Inpatient Discharges data published through the Statewide Planning and Research Cooperative System (SPARCS), New York State.

The dataset contains hospital inpatient discharge information, including:

- Patient demographic information
- Age group
- Gender
- Race
- Diagnosis information
- Severity of illness
- Length of hospital stay
- Total healthcare charges
- Payment type
- Hospital facility
- County
- Discharge information

The original dataset is large, containing hundreds of megabytes of data.

To improve application performance and reduce memory usage, the dashboard loads a sample of up to **250,000 records**.

The sample size can be adjusted in the application code.

### Missing Data

The dataset may contain missing values in different columns.

The application handles missing values during data preprocessing.

Categorical information is standardized, and missing values in relevant categories are represented using appropriate placeholder values.

Numerical fields are converted into numeric formats before analysis.

---

## 🧹 Data Processing

Before analysis, the dataset goes through a preprocessing stage.

The application:

- Loads the CSV dataset using Pandas.
- Standardizes column names for easier processing.
- Converts length-of-stay values into numeric data types.
- Converts healthcare charges into numeric values.
- Removes records with invalid or non-positive lengths of stay.
- Removes records containing negative healthcare charges.
- Standardizes patient demographic categories.
- Handles missing categorical information.
- Processes diagnosis and severity information.
- Organizes payment, facility, county, and region information for analysis.
- Limits extreme length-of-stay values to reduce their effect on visualizations.
- Applies the selected dashboard filters.

These steps improve data consistency and make the dataset more suitable for statistical analysis and visualization.

---

## 📐 Statistical Analysis

MediScope uses statistical analysis to investigate patterns in hospital discharge records.

### Average Length of Stay

The average length of stay represents the mean number of days patients spend in the hospital.

Mean Length of Stay:

Mean = Σx / n

Where:

- x represents an individual patient's length of stay.
- n represents the total number of valid discharge records.

### Average Healthcare Charges

The average healthcare charges represent the mean recorded hospital charges across the selected discharge records.

Mean Charges = Σx / n

Where:

- x represents an individual discharge record's healthcare charges.
- n represents the total number of valid records.

### Severity Comparison

The application compares length of stay and healthcare charges across different severity categories.

This helps investigate whether records with different severity classifications show differences in hospitalization duration and recorded charges.

### Data Distribution

Histograms, boxplots, and violin plots are used to examine the distribution and variation of hospital stays and healthcare charges.

These visualizations help identify patterns, differences, and potential outliers in the dataset.

---

## 📊 Dashboard Visualizations

MediScope contains multiple interactive visualizations for exploring hospital discharge information.

### 01 · Patient Age Distribution

The Patient Age Distribution visualization shows the distribution of hospital discharge records across different patient age groups.

The visualization uses a bar chart.

Purpose

This visualization helps users understand which age groups account for a larger number of inpatient discharge records.

### 02 · Payment Type Distribution

The Payment Type visualization displays the distribution of patients across different payment categories.

The visualization uses a pie chart.

Purpose

This helps users understand the distribution of hospital discharge records across different payment types.

### 03 · Length-of-Stay Distribution

The Length-of-Stay visualization displays how hospital stay durations are distributed across discharge records.

The visualization uses a histogram.

Purpose

This visualization helps identify common hospitalization durations and examine the variation in length of stay.

### 04 · Diagnosis and Average Length of Stay

This visualization compares average hospital length of stay across different diagnosis categories.

The visualization uses a bar chart.

Purpose

This helps investigate which diagnosis categories are associated with longer average hospital stays.

### 05 · Severity and Length of Stay

This visualization compares the distribution of hospital stay durations across severity categories.

The visualization uses a boxplot.

The boxplot displays:

- Median length of stay
- Middle portion of the data
- Spread of hospitalization durations
- Potential outliers

Purpose

This visualization helps compare hospitalization duration across different severity classifications.

### 06 · Healthcare Charges by Severity

This visualization compares healthcare charges across different severity categories.

The visualization uses a boxplot.

Purpose

This helps investigate how recorded healthcare charges vary between severity groups.

### 07 · Healthcare Charges by Age and Gender

This visualization examines healthcare charges across patient age groups and gender categories.

Purpose

This helps users explore differences in recorded hospital charges between demographic groups.

### 08 · Facility and County Heatmap

The Facility and County Heatmap displays average length of stay across hospital facilities and counties.

The visualization uses a heatmap.

Colour intensity represents the average length of hospital stay.

Purpose

This visualization makes it easier to compare hospitalization patterns across different facilities and geographical areas.

### 09 · Hospital Facility Analysis

This visualization compares hospital facilities using discharge volume and average length of stay.

Purpose

This helps identify facilities with higher discharge volumes and investigate differences in average hospitalization duration.

### 10 · Healthcare Charge Distribution

This visualization uses a violin plot to display the distribution of healthcare charges.

Purpose

This helps users examine the spread and variation of recorded hospital charges.

---

## 🔎 Interactive Filters

The dashboard provides interactive filters through the sidebar.

Users can filter the dataset using available categories such as:

- Patient age group
- Gender
- Severity
- Payment type

The selected filters are applied dynamically to the dashboard.

When filters are changed:

- KPI values update.
- Graphs update.
- Statistical information updates.
- The displayed dataset updates.
- The downloadable dataset updates.

This allows users to explore specific subsets of hospital discharge records.

---

## 📈 Key Performance Indicators

The dashboard displays important summary metrics.

### Total Discharge Records

Shows the number of hospital discharge records currently included after applying the selected filters.

### Average Length of Stay

Shows the average number of days patients spent in the hospital within the currently filtered dataset.

### Average Healthcare Charges

Shows the average recorded healthcare charges within the selected discharge records.

### Additional Metrics

Other metrics can be explored through the dashboard visualizations and data analysis sections.

---

## 📋 Raw Dataset

The Raw Dataset section allows users to inspect hospital discharge records directly inside the application.

The application provides:

- Number of filtered records
- Filtered dataset viewing
- Search and filtering options
- CSV download option

Users can download the currently filtered dataset as a CSV file.

This allows users to continue working with the selected hospital discharge data offline.

---

## 🤖 AI Experiment

MediScope includes a machine learning experiment for classifying hospital discharge severity.

The experiment uses Scikit-learn to train a classification model using selected dataset features.

The application uses a Random Forest classification model when the required machine learning dependencies are available.

The experiment includes:

- Data preparation for model training
- Feature selection
- Training and testing data separation
- Severity classification
- Classification performance evaluation
- Confusion matrix visualization

Purpose

This experiment demonstrates how machine learning can be applied to structured hospital discharge data.

The model is intended for educational exploration only. Its predictions are not validated medical assessments and must not be used for clinical decisions.

---

## 🧪 Data Quality Analysis

The Data Quality section provides an overview of the dataset after preprocessing.

It helps users investigate:

- Missing values
- Data availability
- Invalid records removed during preprocessing
- Data consistency
- Selected dataset characteristics

Purpose

This section helps users understand the quality of the source dataset and the importance of data preprocessing before performing analysis.

---

## 🛠️ Technologies Used

### Programming Language

**Python 3.13**

Python is used for application development, data processing, statistical calculations, and machine learning.

### Data Processing

**Pandas**

Pandas is used for:

- Loading the dataset
- Cleaning data
- Handling missing values
- Filtering records
- Grouping data
- Calculating statistical measures
- Preparing data for visualization

**NumPy**

NumPy is used for numerical operations and data calculations.

### Data Visualization

**Plotly**

Plotly is used to create interactive:

- Bar charts
- Pie charts
- Histograms
- Boxplots
- Violin plots
- Heatmaps

### Machine Learning

**Scikit-learn**

Scikit-learn is used for:

- Classification model development
- Random Forest classification
- Training and testing data preparation
- Model evaluation
- Confusion matrix generation

### Web Application

**Streamlit**

Streamlit is used to build the interactive dashboard and provide:

- Sidebar filters
- Navigation sections
- KPI cards
- Interactive graphs
- Dataset viewing
- CSV download
- Data analysis
- Machine learning experiment

---

## 📦 Project Structure

```text
1000411-sarveshwaran-Hospital-Inpatient-Discharges/
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
└── Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv
```

The CSV dataset may also be placed inside a `data/` or `dataset/` folder.

The application searches the supported locations for the dataset.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/sarveshwaran777333/1000411-sarveshwaran-Hospital-Inpatient-Discharges.git
```

Move into the project folder:

```bash
cd 1000411-sarveshwaran-Hospital-Inpatient-Discharges
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

The project requires the following Python libraries:

- streamlit
- pandas
- numpy
- plotly
- scikit-learn

These dependencies are also provided in:

`requirements.txt`

---

## ▶️ Running the Application

Run the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in a web browser.

After opening the dashboard:

- Explore the Overview section.
- Use the sidebar filters.
- Explore the diagnosis and length-of-stay visualizations.
- Compare healthcare charges across severity categories.
- Open the Facilities and Counties section.
- Inspect the Data Quality section.
- Explore the AI Experiment.
- Open the Data Explorer section.
- Download filtered records if required.

---

## 🔬 Testing

The dashboard can be tested using different filter combinations.

Examples include:

- Different patient age groups
- Different gender categories
- Different severity categories
- Different payment types
- Multiple filters at the same time

Testing should verify that:

- KPI values update correctly.
- Graphs respond to selected filters.
- Average length-of-stay calculations work correctly.
- Healthcare charge calculations work correctly.
- Diagnosis comparisons display the correct data.
- Severity comparisons update correctly.
- Facility and county visualizations display correctly.
- The raw dataset matches the selected filters.
- The download option exports the filtered records.
- Missing-value analysis works correctly.
- The AI experiment handles the available data correctly.
- Empty filter results are handled correctly.
- Dashboard navigation works correctly.

---

## ⚠️ Limitations

MediScope has several limitations.

### Dataset Dependency

The quality of the analysis depends on the quality and completeness of the original hospital discharge dataset.

### Sample-Based Analysis

The application loads up to 250,000 records from the original dataset to reduce memory usage and improve performance.

Therefore, the displayed results may differ from calculations performed using the complete dataset.

### Correlation Does Not Prove Causation

The visualizations show patterns and relationships between variables.

They do not prove that a particular patient characteristic, diagnosis, or hospital factor directly causes a change in length of stay or healthcare charges.

### Missing Information

Some potentially important healthcare factors may not be available in the dataset, such as:

- Detailed patient medical history
- Individual treatment information
- Specific clinical procedures
- Real-time hospital occupancy
- Detailed treatment outcomes
- Individual patient recovery information

### Static Dataset

The dashboard analyzes the provided 2021 dataset and does not automatically receive live hospital discharge information.

### Machine Learning Limitations

The severity classification experiment is based on the available dataset and selected features.

Its performance depends on data quality, feature selection, class distribution, and model evaluation.

The experiment is educational and is not a validated clinical prediction system.

---

## 🚀 Future Improvements

Possible future improvements include:

- Adding support for processing the complete dataset efficiently.
- Adding more detailed diagnosis analysis.
- Adding additional patient demographic comparisons.
- Adding discharge outcome analysis.
- Adding hospital occupancy analysis.
- Adding more advanced statistical analysis.
- Improving machine learning model evaluation.
- Comparing multiple classification algorithms.
- Adding more interactive geographical visualizations.
- Adding year-wise hospital discharge comparisons.
- Adding more detailed data quality reports.
- Adding automated analytical reports.
- Improving dashboard performance for larger datasets.

---

## 📚 Research and Learning

This project demonstrates the application of several data analytics and artificial intelligence concepts:

- Data Cleaning
- Data Preprocessing
- Missing Value Handling
- Descriptive Statistics
- Mean
- Data Filtering
- Grouped Data Analysis
- Healthcare Data Analysis
- Data Visualization
- Histogram Analysis
- Boxplot Analysis
- Violin Plot Analysis
- Heatmap Analysis
- Interactive Dashboards
- CSV Data Processing
- Data Export
- Machine Learning Classification
- Random Forest
- Model Evaluation
- Confusion Matrix

The project demonstrates how raw hospital discharge data can be transformed into an interactive analytical dashboard that helps users investigate healthcare-related patterns.

---

## 🌐 Deployment

The application is designed to run using Streamlit.

The project can be hosted using a Streamlit-compatible deployment platform with the following files:

```text
app.py
requirements.txt
README.md
```

The hospital discharge CSV must also be accessible to the deployed application.

Because the original dataset is large, deployment requires an appropriate data-hosting strategy and sufficient memory and storage.

**GitHub Repository:**

https://github.com/sarveshwaran777333/1000411-sarveshwaran-Hospital-Inpatient-Discharges

**Live Demo:**

https://1000411-sarveshwaran-hospital-inpatient-discharges-cnptg9nvq8j.streamlit.app/
---

## 👨‍💻 Project Information

Project: MediScope

Title: Smart Hospital Inpatient Discharges Analytics Dashboard

Course: Data Analytics

CRS: Artificial Intelligence

Programming Language: Python 3.13

Data Processing: Pandas, NumPy

Visualization: Plotly

Machine Learning: Scikit-learn

Application Framework: Streamlit

Dataset: Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv

GitHub Repository:

https://github.com/sarveshwaran777333/1000411-sarveshwaran-Hospital-Inpatient-Discharges

Testing:

To validate usability and overall user experience, the Streamlit dashboard should be tested using different filter combinations and dataset selections. Testing should evaluate navigation, chart clarity, filtering behaviour, data accuracy, and dashboard responsiveness. Particular attention should be given to initial loading time because of the large dataset size.

---

## 📌 Conclusion

MediScope demonstrates how Python-based data analytics can be used to explore hospital inpatient discharge information.

By combining data preprocessing, statistical analysis, interactive filtering, data visualization, and machine learning, the dashboard provides a simple way to investigate relationships between patient demographics, diagnoses, hospital stays, healthcare charges, and hospital facilities.

The visualizations provide different perspectives on hospital discharge patterns, while the interactive filtering system allows users to explore specific subsets of the dataset.

The Data Quality section helps users understand the importance of preprocessing, and the AI Experiment demonstrates how machine learning classification can be applied to structured healthcare data.

Overall, MediScope demonstrates how raw hospital discharge records can be transformed into a useful and interactive data analytics solution using Python, Pandas, NumPy, Plotly, Scikit-learn, and Streamlit.

**Credits:**

Name: Sarveshwaran K

No: 1000411

School Name: Jain Vidyalaya IB World School

Mentor's Name: Add mentor's name
