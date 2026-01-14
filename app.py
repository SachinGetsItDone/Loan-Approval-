import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================================
# Load the pre-trained pipeline
# =========================================

@st.cache_resource
def load_pipeline():
    try:
        with open('loan_pipeline.pkl', 'rb') as f:
            pipeline = pickle.load(f)
        return pipeline
    except FileNotFoundError:
        st.error("⚠️ loan_pipeline.pkl not found.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading pipeline: {str(e)}")
        st.stop()

# =========================================
# Feature Engineering
# =========================================

def prepare_input(input_data, expected_columns):
    input_df = pd.DataFrame([input_data])

    numeric_features = [
        "Applicant_ID", "Applicant_Income", "Coapplicant_Income",
        "Age", "Dependents", "Credit_Score", "Existing_Loans",
        "DTI_Ratio", "Savings", "Collateral_Value",
        "Loan_Amount", "Loan_Term"
    ]

    categorical_features = [
        "Employment_Status", "Marital_Status", "Loan_Purpose",
        "Property_Area", "Education_Level", "Gender",
        "Employer_Category"
    ]

    numeric_df = input_df[numeric_features].copy()
    categorical_df = pd.get_dummies(
        input_df[categorical_features], drop_first=False
    )

    processed_df = pd.concat([numeric_df, categorical_df], axis=1)

    for col in expected_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0

    return processed_df[expected_columns]

# =========================================
# Prediction
# =========================================

def predict_loan_approval(pipeline, input_data):
    try:
        if hasattr(pipeline, 'feature_names_in_'):
            expected_columns = pipeline.feature_names_in_.tolist()
            input_df = prepare_input(input_data, expected_columns)
        else:
            input_df = pd.DataFrame([input_data])

        prediction = pipeline.predict(input_df)

        if hasattr(pipeline, 'predict_proba'):
            probability = pipeline.predict_proba(input_df)
            return prediction[0], probability[0]

        return prediction[0], None

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# =========================================
# UI
# =========================================

def main():

    st.set_page_config(
        page_title="Loan Approval Predictor",
        page_icon="💰",
        layout="wide"
    )

    # ===================== CSS =====================
    st.markdown("""
    <style>

    html, body {
        background-color: #fffaf5;
        font-family: 'Segoe UI', sans-serif;
    }

    .main {
        padding: 2.5rem 3rem;
    }

    h1 {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff9966, #ff5e62);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }

    .subtitle {
        text-align: center;
        font-size: 1.15rem;
        color: #7a6c5d;
        margin-bottom: 2.2rem;
    }

    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ff7a18;
        margin: 1.8rem 0 1rem;
        border-bottom: 2px solid #ffe0c3;
        padding-bottom: 0.4rem;
    }

    .info-box {
        background: #fff2e6;
        border-left: 5px solid #ff9f68;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: #5c3d2e;
    }

    .stButton > button {
        height: 3.6rem;
        font-size: 1.15rem;
        font-weight: 700;
        border-radius: 14px;
        background: linear-gradient(90deg, #ff9966, #ff5e62);
        color: white;
        border: none;
    }

    .success-card {
        background: linear-gradient(135deg, #56ab2f, #a8e063);
        padding: 2rem;
        border-radius: 18px;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }

    .danger-card {
        background: linear-gradient(135deg, #ff512f, #f09819);
        padding: 2rem;
        border-radius: 18px;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }

    </style>
    """, unsafe_allow_html=True)

    # ===================== HEADER =====================
    st.markdown("<h1>💰 Loan Approval Predictor</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>Friendly AI assistance to evaluate loan eligibility</p>",
        unsafe_allow_html=True
    )

    pipeline = load_pipeline()

    st.markdown("""
    <div class='info-box'>
        ✅ <b>Model loaded successfully.</b> Fill the form to analyze eligibility.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📝 Application", "ℹ️ About"])

    # ===================== FORM =====================
    with tab1:
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("<div class='section-header'>👤 Personal</div>", unsafe_allow_html=True)

            applicant_id = st.number_input("Applicant ID", 1, step=1)
            age = st.number_input("Age", 18, 100, 30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
            dependents = st.number_input("Dependents", 0, 10, 0)
            education_level = st.selectbox("Education", ["Graduate", "Not Graduate"])

            st.markdown("<div class='section-header'>💼 Employment</div>", unsafe_allow_html=True)
            employment_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])
            employer_category = st.selectbox("Employer Type", ["Private", "Government", "MNC", "Unemployed"])
            applicant_income = st.number_input("Monthly Income", 0.0, value=5000.0)
            coapplicant_income = st.number_input("Co-applicant Income", 0.0, value=0.0)

        with col2:
            st.markdown("<div class='section-header'>📊 Financial</div>", unsafe_allow_html=True)

            credit_score = st.number_input("Credit Score", 300, 850, 650)
            existing_loans = st.number_input("Existing Loans", 0, 20, 0)
            dti_ratio = st.number_input("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
            savings = st.number_input("Savings", 0.0, value=10000.0)

            st.markdown("<div class='section-header'>🏠 Loan</div>", unsafe_allow_html=True)
            loan_amount = st.number_input("Loan Amount", 0.0, value=10000.0)
            loan_term = st.number_input("Loan Term (months)", 1, 480, 36)
            loan_purpose = st.selectbox("Purpose", ["Personal", "Business", "Car", "Home", "Education"])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            collateral_value = st.number_input("Collateral Value", 0.0, value=20000.0)

        input_data = {
            "Applicant_ID": applicant_id,
            "Applicant_Income": applicant_income,
            "Coapplicant_Income": coapplicant_income,
            "Employment_Status": employment_status,
            "Age": age,
            "Marital_Status": marital_status,
            "Dependents": dependents,
            "Credit_Score": credit_score,
            "Existing_Loans": existing_loans,
            "DTI_Ratio": dti_ratio,
            "Savings": savings,
            "Collateral_Value": collateral_value,
            "Loan_Amount": loan_amount,
            "Loan_Term": loan_term,
            "Loan_Purpose": loan_purpose,
            "Property_Area": property_area,
            "Education_Level": education_level,
            "Gender": gender,
            "Employer_Category": employer_category
        }

        if st.button("🔮 Analyze Application"):
            with st.spinner("Analyzing..."):
                pred, prob = predict_loan_approval(pipeline, input_data)

                if prob is not None:
                    score = prob[1] * 100

                    if score >= 70:
                        st.markdown(f"""
                        <div class='success-card'>
                            <h2>✅ LOAN APPROVED</h2>
                            <h3>{score:.1f}% Confidence</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='danger-card'>
                            <h2>❌ NOT APPROVED</h2>
                            <h3>{score:.1f}% Confidence</h3>
                        </div>
                        """, unsafe_allow_html=True)

                    st.progress(prob[1])

    # ===================== ABOUT =====================
    with tab2:
        st.markdown("""
        ### About This App
        - Chef Sachin cooked this app
        - AI-powered loan approval predictor
        - 70% confidence threshold
        - Built using Streamlit + ML Pipeline
        - Designed for demo & production use
        """)

if __name__ == "__main__":
    main()
