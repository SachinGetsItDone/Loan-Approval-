import streamlit as st
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

st.title("🏦 Loan Approval Prediction App")
st.write("Enter applicant details to predict loan approval status")

age = st.number_input("Age", min_value=18, max_value=100)
income = st.number_input("Applicant Income", min_value=0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900)
dti = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0)
savings = st.number_input("Savings", min_value=0)

gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married"])
employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Self-employed"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

input_data = pd.DataFrame({
    "Age": [age],
    "Applicant_Income": [income],
    "Credit_Score": [credit_score],
    "DTI_Ratio": [dti],
    "Savings": [savings],
    "Gender": [gender],
    "Marital_Status": [marital_status],
    "Employment_Status": [employment_status],
    "Property_Area": [property_area]
})

input_data = pd.DataFrame({
    "Age": [age],
    "Applicant_Income": [income],
    "Credit_Score": [credit_score],
    "DTI_Ratio": [dti],
    "Savings": [savings],
    "Gender": [gender],
    "Marital_Status": [marital_status],
    "Employment_Status": [employment_status],
    "Property_Area": [property_area]
})

if st.button("Predict Loan Approval"):
    prediction = model.predict(final_input)[0]

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
