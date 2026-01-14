import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Try to import plotly, use fallback if not available
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly not installed. Charts will use simplified visualizations.")

# =========================================
# Load the pre-trained pipeline
# =========================================
@st.cache_resource
def load_pipeline():
    try:
        with open('loan_pipeline.pkl', 'rb') as f:
            pipeline = pickle.load(f)
        return pipeline
    except Exception as e:
        st.error(f"Pipeline load error: {e}")
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

    num_df = input_df[numeric_features]
    cat_df = pd.get_dummies(input_df[categorical_features], drop_first=False)
    df = pd.concat([num_df, cat_df], axis=1)

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    return df[expected_columns]

# =========================================
# Prediction
# =========================================
def predict_loan_approval(pipeline, input_data):
    expected_columns = pipeline.feature_names_in_.tolist()
    X = prepare_input(input_data, expected_columns)
    pred = pipeline.predict(X)[0]
    prob = pipeline.predict_proba(X)[0]
    return pred, prob

# =========================================
# Streamlit UI
# =========================================
def main():
    st.set_page_config(
        page_title="Loan Approval Predictor",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # ================= GLASS CSS (ONLY VISUAL) =================
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    .main {
        background: linear-gradient(180deg, #f9fafb, #eef2ff);
    }

    .card,
    .metric-square,
    .result-card-success,
    .result-card-danger,
    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.65) !important;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255,255,255,0.45);
        box-shadow: 0 8px 30px rgba(0,0,0,0.06);
    }

    .result-card-success {
        background: rgba(240,253,244,0.7) !important;
    }

    .result-card-danger {
        background: rgba(254,242,242,0.7) !important;
    }

    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] {
        background: rgba(255,255,255,0.7) !important;
        backdrop-filter: blur(10px);
    }

    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.6);
        backdrop-filter: blur(12px);
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("<h1>Loan Approval System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>AI-powered credit decision platform</p>", unsafe_allow_html=True)

    pipeline = load_pipeline()

    tab1, tab2 = st.tabs(["Application", "Information"])

    with tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            applicant_id = st.number_input("Applicant ID", 1, value=1)
            age = st.number_input("Age", 18, 100, 30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
            dependents = st.number_input("Dependents", 0, 10, 0)
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            income = st.number_input("Applicant Income", 0.0, value=5000.0)
            co_income = st.number_input("Coapplicant Income", 0.0, value=0.0)
            credit = st.number_input("Credit Score", 300, 850, 650)
            loans = st.number_input("Existing Loans", 0, 20, 0)
            dti = st.slider("DTI Ratio", 0.0, 1.0, 0.3)
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            savings = st.number_input("Savings", 0.0, value=10000.0)
            collateral = st.number_input("Collateral Value", 0.0, value=20000.0)
            amount = st.number_input("Loan Amount", 0.0, value=10000.0)
            term = st.number_input("Loan Term", 1, 480, 36)
            purpose = st.selectbox("Loan Purpose", ["Personal", "Business", "Car", "Home", "Education"])
            area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            employment = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])
            employer = st.selectbox("Employer Type", ["Private", "Government", "MNC", "Unemployed"])
            st.markdown("</div>", unsafe_allow_html=True)

        input_data = {
            "Applicant_ID": applicant_id,
            "Applicant_Income": income,
            "Coapplicant_Income": co_income,
            "Employment_Status": employment,
            "Age": age,
            "Marital_Status": marital_status,
            "Dependents": dependents,
            "Credit_Score": credit,
            "Existing_Loans": loans,
            "DTI_Ratio": dti,
            "Savings": savings,
            "Collateral_Value": collateral,
            "Loan_Amount": amount,
            "Loan_Term": term,
            "Loan_Purpose": purpose,
            "Property_Area": area,
            "Education_Level": education,
            "Gender": gender,
            "Employer_Category": employer
        }

        if st.button("Analyze Application"):
            pred, prob = predict_loan_approval(pipeline, input_data)
            prob_pct = prob[1] * 100

            if prob_pct >= 70:
                st.success(f"Approved — {prob_pct:.1f}%")
            else:
                st.error(f"Not Approved — {prob_pct:.1f}%")

    with tab2:
        st.info("This system provides AI-assisted credit recommendations only.")

    # ================= FOOTER =================
    st.markdown("""
    <div style="text-align:center; margin-top:2.5rem; color:#6b7280; font-size:0.9rem;">
        🍽️ Cooked by <strong>Chef Sachin</strong> 👨‍🍳
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
