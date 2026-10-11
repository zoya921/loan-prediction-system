import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide"
)

# Load the saved machine learning model
model = joblib.load("loan_approval_model.pkl")

# Dashboard heading
st.title("🏦 Loan Approval Prediction System")
st.write(
    "An AI-powered dashboard to explore loan data, "
    "predict approval status, and estimate model-based risk."
)

st.info(
    "Educational prediction only. Results are not a guarantee "
    "of actual bank approval."
)

st.divider()

st.header("📂 Upload Loan Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        st.subheader("Dataset Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Rows", uploaded_df.shape[0])

        with col2:
            st.metric("Total Columns", uploaded_df.shape[1])

        with col3:
            st.metric(
                "Missing Values",
                int(uploaded_df.isnull().sum().sum())
            )

        st.subheader("Dataset Preview")
        st.dataframe(uploaded_df.head(10), use_container_width=True)

        st.subheader("Missing Values by Column")
        missing_values = uploaded_df.isnull().sum()
        missing_values = missing_values[missing_values > 0]

        if not missing_values.empty:
            st.bar_chart(missing_values)
        else:
            st.success("No missing values found!")

    except Exception as e:
        st.error(f"Could not read the CSV file: {e}")
else:
    st.caption("Upload a CSV file to view its summary and preview.")


st.divider()
st.header("🧮 Loan Eligibility Checker")
st.write("Enter applicant details to generate a model-based prediction.")

with st.form("loan_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox(
            "Education", ["Graduate", "Not Graduate"]
        )
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        applicant_income = st.number_input(
            "Applicant Income", min_value=0, value=5000, step=500
        )

    with col2:
        coapplicant_income = st.number_input(
            "Coapplicant Income", min_value=0, value=0, step=500
        )
        loan_amount = st.number_input(
            "Loan Amount (in dataset units)",
            min_value=1, value=150, step=10
        )
        loan_term = st.selectbox(
            "Loan Term (months)", [360, 180, 120, 60, 36, 12]
        )
        credit_history = st.selectbox(
            "Credit History", [1.0, 0.0],
            format_func=lambda x: "Good / available (1)" if x == 1.0
            else "Poor / unavailable (0)"
        )
        property_area = st.selectbox(
            "Property Area", ["Urban", "Semiurban", "Rural"]
        )

    submitted = st.form_submit_button("Predict Loan Status")

if submitted:
    applicant_details = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }

    applicant_df = pd.DataFrame([applicant_details])

    prediction = model.predict(applicant_df)[0]
    probabilities = model.predict_proba(applicant_df)[0]
    approved_index = list(model.classes_).index("Y")

    approval_probability = float(probabilities[approved_index] * 100)
    risk_score = 100 - approval_probability

    if risk_score < 20:
        risk_category = "Low Risk"
    elif risk_score < 50:
        risk_category = "Medium Risk"
    else:
        risk_category = "High Risk"

    st.subheader("Prediction Results")

    result1, result2, result3 = st.columns(3)

    with result1:
        status_text = "Approved (Y)" if prediction == "Y" else "Rejected (N)"
        st.metric("Predicted Status", status_text)

    with result2:
        st.metric("Approval Probability", f"{approval_probability:.2f}%")

    with result3:
        st.metric("Model-Based Risk Score", f"{risk_score:.2f}%")

    st.write("**Risk Category:**", risk_category)
    st.progress(int(round(approval_probability)))
    st.caption(
        "These are model estimates, not a guarantee of actual loan approval."
    )

st.divider()
st.header("📊 Machine Learning Model Comparison")

comparison_data = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "Decision Tree"
    ],
    "Accuracy": [0.862, 0.854, 0.821],
    "Precision": [0.840, 0.832, 0.825],
    "Recall": [0.988, 0.988, 0.941],
    "F1 Score": [0.908, 0.903, 0.879]
})

st.dataframe(
    comparison_data.set_index("Model").style.format("{:.3f}"),
    use_container_width=True
)

st.bar_chart(
    comparison_data.set_index("Model")[
        ["Accuracy", "Precision", "Recall", "F1 Score"]
    ]
)

st.divider()
st.header("🔍 Important Factors Behind Predictions")

classifier = model.named_steps["classifier"]
preprocessor = model.named_steps["preprocessor"]

feature_names = preprocessor.get_feature_names_out()
coefficients = classifier.coef_[0]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Importance": np.abs(coefficients)
})

importance_df = importance_df.sort_values(
    "Importance", ascending=False
).head(10)

st.write(
    "These are the 10 features with the largest absolute coefficients "
    "in the Logistic Regression model."
)

st.dataframe(importance_df.round(3), use_container_width=True)

chart_df = importance_df.sort_values("Coefficient").set_index("Feature")

st.bar_chart(chart_df["Coefficient"])
st.caption(
    "Positive coefficients push predictions toward Y; negative coefficients "
    "push them toward N. These are learned model relationships, not proof "
    "of real-world causes."
)


st.divider()
st.header("💡 Automated Data Insights")

if "uploaded_df" in locals():
    insight_df = uploaded_df.copy()

    st.subheader("Dataset Summary")

    st.write(f"- Dataset contains **{insight_df.shape[0]} rows** and **{insight_df.shape[1]} columns**.")

    missing_total = int(insight_df.isnull().sum().sum())
    st.write(f"- Total missing values: **{missing_total}**.")

    duplicate_total = int(insight_df.duplicated().sum())
    st.write(f"- Duplicate rows: **{duplicate_total}**.")

    if "Loan_Status" in insight_df.columns:
        status_counts = insight_df["Loan_Status"].value_counts(dropna=True)
        st.subheader("Loan Status Distribution")
        st.bar_chart(status_counts)

        if len(status_counts) > 0:
            common_status = status_counts.idxmax()
            st.write(
                f"- Most common loan status in this dataset: **{common_status}**."
            )

    numeric_columns = insight_df.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_columns:
        st.subheader("Numeric Data Summary")
        st.dataframe(
            insight_df[numeric_columns].describe().round(2),
            use_container_width=True
        )
else:
    st.info("Upload a CSV dataset above to generate automated insights.")

