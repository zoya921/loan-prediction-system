# Loan Approval Prediction System

## Project Overview

The **Loan Approval Prediction System** is a machine learning project that predicts whether a loan application is likely to be approved or rejected based on applicant information and financial details.

The project combines data analysis, data preprocessing, machine learning model comparison, risk scoring, and an interactive Streamlit dashboard to demonstrate an end-to-end machine learning workflow.

## Objectives

- Analyze loan applicant data and identify patterns.
- Handle missing values and preprocess categorical and numerical features.
- Train and compare multiple machine learning models.
- Predict loan approval status for new applicants.
- Calculate model-based approval probability and risk scores.
- Present insights through an interactive dashboard.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook

## Project Structure

```text
loan-approval-prediction-system/
│
├── app.py
├── Loan_Approval_Prediction_System.ipynb
├── loan_approval_model.pkl
├── loan_data_set.csv
├── requirements.txt
└── README.md
```

## Dataset

The project uses a loan prediction dataset containing applicant information such as:

- Gender and marital status
- Number of dependents
- Education and self-employment status
- Applicant and co-applicant income
- Loan amount and loan term
- Credit history
- Property area
- Loan approval status

The dataset contains **614 rows and 13 columns**.

## Data Preprocessing

The following preprocessing steps were performed:

- Examined missing values and dataset characteristics.
- Imputed missing numerical values using the median.
- Imputed missing categorical values using the most frequent value.
- Encoded categorical features using One-Hot Encoding.
- Standardized numerical features using StandardScaler.
- Split the data into training and testing sets.

## Machine Learning Models

Three classification algorithms were trained and compared:

1. Logistic Regression
2. Random Forest Classifier
3. Decision Tree Classifier

### Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic
