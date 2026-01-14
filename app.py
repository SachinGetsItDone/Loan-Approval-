import streamlit as st
import pickle
import pandas as pd

# ---------------- LOAD MODELS ----------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))  # IMPORTANT

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# ---------------- TITLE ----------------
st.title("🏦 Loan Approval Prediction")
st.markdown("Predict whether a loan will be **approved or rejected** using ML.")

st.divider()

# ---------------- INPUT FORM ----------------
with st.form("loan_form"):
    st.subheader("Applicant Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 30)
        income = st.number_input("Applicant Income", min_value=0, value=60000)
        savings = st.number_input("Savings", min_value=0, value=200000)

    with col2:
        credit_score = st.number_input("Credit Score", 300, 900, 750)
        dti = st.slider("DTI Ratio", 0.0, 1.0, 0.25)

    st.subheader("Personal Information")

    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    employment_status = st.selectbox(
        "Employment Status",
        ["Employed", "Self-employed", "Unemployed"]
    )
    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )

    submit = st.form_submit_button("🔍 Predict Loan Approval")

# ---------------- PREDICTION ----------------
if submit:
    input_df = pd.DataFrame({
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

    # Numerical scaling
    num_cols = list(scaler.feature_names_in_)
    scaled_num = scaler.transform(input_df[num_cols])
    scaled_df = pd.DataFrame(scaled_num, columns=num_cols)

    # Categorical encoding
    cat_cols = encoder.feature_names_in_
    encoded_cat = encoder.transform(input_df[cat_cols])
    encoded_df = pd.DataFrame(
        encoded_cat,
        columns=encoder.get_feature_names_out(cat_cols)
    )

    # Final input
    final_input = pd.concat([scaled_df, encoded_df], axis=1)
    final_input = final_input.reindex(columns=features, fill_value=0)

    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0][1]

    st.divider()

    if prediction == 1:
        st.success(f"✅ Loan Approved (Confidence: {probability:.2%})")
    else:
        st.error(f"❌ Loan Rejected (Confidence: {1 - probability:.2%})")

