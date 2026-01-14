import streamlit as st
import pandas as pd
import pickle

# ---------------- LOAD PIPELINE ----------------
@st.cache_resource
def load_pipeline():
    return pickle.load(open("loan_pipeline.pkl", "rb"))

pipeline = load_pipeline()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# ---------------- TITLE ----------------
st.title("🏦 Loan Approval Prediction")
st.markdown("Predict whether a loan will be **Approved or Rejected** using Machine Learning.")
st.divider()

# ---------------- INPUT FORM ----------------
with st.form("loan_form"):
    st.subheader("Applicant Financial Details")

    col1, col2 = st.columns(2)

    with col1:
        applicant_income = st.number_input("Applicant Income", min_value=0, value=60000)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=10000)
        savings = st.number_input("Savings", min_value=0, value=200000)
        collateral = st.number_input("Collateral Value", min_value=0, value=300000)
        loan_amount = st.number_input("Loan Amount", min_value=0, value=150000)

    with col2:
        credit_score = st.number_input("Credit Score", 300, 900, 750)
        dti = st.slider("DTI Ratio", 0.0, 1.0, 0.25)
        existing_loans = st.number_input("Existing Loans", 0, 10, 1)
        loan_term = st.number_input("Loan Term (months)", 6, 360, 60)
        dependents = st.number_input("Dependents", 0, 10, 1)

    st.subheader("Personal & Employment Information")

    col3, col4 = st.columns(2)

    with col3:
        age = st.number_input("Age", 18, 100, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])

    with col4:
        employment_status = st.selectbox(
            "Employment Status",
            ["Salaried", "Self-employed", "Unemployed"]
        )
        employer_category = st.selectbox(
            "Employer Category",
            ["Private", "Government", "MNC", "Unemployed"]
        )
        loan_purpose = st.selectbox(
            "Loan Purpose",
            ["Home", "Car", "Education", "Personal", "Business"]
        )
        property_area = st.selectbox(
            "Property Area",
            ["Urban", "Semiurban", "Rural"]
        )

    submit = st.form_submit_button("🔍 Predict Loan Approval")

# ---------------- PREDICTION ----------------
if submit:
    input_df = pd.DataFrame({
        "Applicant_ID": [1],   # 👈 REQUIRED by pipeline (dummy value)
        "Applicant_Income": [applicant_income],
        "Coapplicant_Income": [coapplicant_income],
        "Age": [age],
        "Dependents": [dependents],
        "Credit_Score": [credit_score],
        "Existing_Loans": [existing_loans],
        "DTI_Ratio": [dti],
        "Savings": [savings],
        "Collateral_Value": [collateral],
        "Loan_Amount": [loan_amount],
        "Loan_Term": [loan_term],
        "Employment_Status": [employment_status],
        "Marital_Status": [marital_status],
        "Loan_Purpose": [loan_purpose],
        "Property_Area": [property_area],
        "Education_Level": [education],
        "Gender": [gender],
        "Employer_Category": [employer_category]
    })

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    st.divider()

    if prediction == 1:
        st.success(f"✅ Loan Approved\n\n**Confidence:** {probability:.2%}")
    else:
        st.error(f"❌ Loan Rejected\n\n**Confidence:** {(1 - probability):.2%}")

