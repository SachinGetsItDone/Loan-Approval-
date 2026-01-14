import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================================
# Optional Plotly Support
# =========================================
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly not installed. Install with: pip install plotly")

# =========================================
# Load Pipeline
# =========================================
@st.cache_resource
def load_pipeline():
    with open("loan_pipeline.pkl", "rb") as f:
        return pickle.load(f)

# =========================================
# Feature Engineering
# =========================================
def prepare_input(input_data, expected_columns):
    df = pd.DataFrame([input_data])

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

    num_df = df[numeric_features]
    cat_df = pd.get_dummies(df[categorical_features], drop_first=False)

    final_df = pd.concat([num_df, cat_df], axis=1)

    for col in expected_columns:
        if col not in final_df.columns:
            final_df[col] = 0

    return final_df[expected_columns]

# =========================================
# Prediction
# =========================================
def predict_loan_approval(pipeline, input_data):
    expected_cols = pipeline.feature_names_in_.tolist()
    X = prepare_input(input_data, expected_cols)
    pred = pipeline.predict(X)[0]
    prob = pipeline.predict_proba(X)[0][1]
    return pred, prob

# =========================================
# Gauge Chart
# =========================================
def create_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#38bdf8" if prob >= 0.7 else "#ef4444"},
            "steps": [
                {"range": [0, 55], "color": "#450a0a"},
                {"range": [55, 70], "color": "#78350f"},
                {"range": [70, 100], "color": "#064e3b"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 4},
                "value": 70
            }
        }
    ))
    fig.update_layout(height=280, margin=dict(t=20, b=20))
    return fig

# =========================================
# MAIN APP
# =========================================
def main():
    st.set_page_config(
        page_title="Loan Approval System",
        page_icon="💰",
        layout="wide"
    )

    # ================== GLASS CSS ==================
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    * { font-family: Inter; }

    .main {
        background: linear-gradient(135deg, #020617, #0f172a);
        padding: 1.5rem;
    }

    .card, .metric-square {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(18px);
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.15);
        padding: 1.5rem;
        box-shadow: 0 10px 35px rgba(0,0,0,0.4);
        color: #e5e7eb;
    }

    h1 { color: #f8fafc; font-size: 2.7rem; font-weight: 800; }
    .subtitle { color: #94a3b8; }

    .stButton>button {
        background: linear-gradient(135deg, #38bdf8, #6366f1);
        border-radius: 14px;
        height: 3.5rem;
        font-weight: 600;
        box-shadow: 0 15px 45px rgba(99,102,241,0.5);
    }

    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #94a3b8;
        font-size: 0.9rem;
        letter-spacing: 0.06em;
    }
    </style>
    """, unsafe_allow_html=True)

    # ================== HEADER ==================
    st.markdown("<h1>Loan Approval System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>AI-powered credit decision platform</p>", unsafe_allow_html=True)

    pipeline = load_pipeline()

    # ================== FORM ==================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        applicant_id = st.number_input("Applicant ID", 1, 9999, 1)
        age = st.number_input("Age", 18, 100, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        dependents = st.number_input("Dependents", 0, 10, 0)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        income = st.number_input("Applicant Income", 0.0, 1e7, 5000.0)
        co_income = st.number_input("Coapplicant Income", 0.0, 1e7, 0.0)
        credit = st.number_input("Credit Score", 300, 850, 650)
        loans = st.number_input("Existing Loans", 0, 20, 0)
        dti = st.slider("DTI Ratio", 0.0, 1.0, 0.3)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        savings = st.number_input("Savings", 0.0, 1e7, 10000.0)
        collateral = st.number_input("Collateral Value", 0.0, 1e7, 20000.0)
        amount = st.number_input("Loan Amount", 0.0, 1e7, 10000.0)
        term = st.number_input("Loan Term (months)", 1, 480, 36)
        purpose = st.selectbox("Loan Purpose", ["Personal", "Business", "Car", "Home", "Education"])
        area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        employment = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])
        employer = st.selectbox("Employer Type", ["Private", "Government", "MNC", "Unemployed"])
        st.markdown("</div>", unsafe_allow_html=True)

    # ================== INPUT ==================
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
        st.markdown("---")

        if prob >= 0.7:
            st.success(f"✅ Approved — Confidence: {prob*100:.1f}%")
        else:
            st.error(f"❌ Not Approved — Confidence: {prob*100:.1f}%")

        if PLOTLY_AVAILABLE:
            st.plotly_chart(create_gauge(prob), use_container_width=True)

    # ================== FOOTER ==================
    st.markdown("""
    <div class="footer">
        🍽️ Cooked by <strong>Chef Sachin</strong> 👨‍🍳
    </div>
    """, unsafe_allow_html=True)

# =========================================
if __name__ == "__main__":
    main()
