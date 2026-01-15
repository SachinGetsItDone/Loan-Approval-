import streamlit as st
import pandas as pd
import numpy as np
import pickle

# load model
@st.cache_resource
def load_pipeline():
    try:
        with open('loan_pipeline.pkl', 'rb') as f:
            pipeline = pickle.load(f)
        return pipeline
    except FileNotFoundError:
        st.error("⚠️ Model file not found.")
        st.stop()

# prepare data
def prepare_input(user_data, model_columns):
    df = pd.DataFrame([user_data])
    
    nums = [
        "Applicant_ID", "Applicant_Income", "Coapplicant_Income",
        "Age", "Dependents", "Credit_Score", "Existing_Loans",
        "DTI_Ratio", "Savings", "Collateral_Value", "Loan_Amount", "Loan_Term"
    ]
    
    cats = [
        "Employment_Status", "Marital_Status", "Loan_Purpose",
        "Property_Area", "Education_Level", "Gender", "Employer_Category"
    ]
    
    num_data = df[nums]
    cat_data = df[cats]
    
    cat_encoded = pd.get_dummies(cat_data, drop_first=False)
    
    processed = pd.concat([num_data, cat_encoded], axis=1)
    
    for col in model_columns:
        if col not in processed.columns:
            processed[col] = 0
            
    processed = processed[model_columns]
    
    return processed

# make prediction
def predict_loan_approval(pipeline, user_data):
    try:
        cols = pipeline.feature_names_in_
    except:
        try:
            cols = pipeline.get_feature_names_out()
        except:
            cols = None

    if cols:
        input_df = prepare_input(user_data, cols)
    else:
        input_df = pd.DataFrame([user_data])
    
    prediction = pipeline.predict(input_df)
    
    prob = None
    try:
        prob = pipeline.predict_proba(input_df)
    except:
        pass
            
    return prediction[0], prob[0]

# app interface
def main():
    st.set_page_config(
        page_title="Loan Approval Predictor",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.markdown("""
        <style>
        .main { padding: 2rem; }
        .stButton>button {
            width: 100%; height: 3.5rem; font-size: 1.2rem; font-weight: 600;
            border-radius: 10px; margin-top: 1rem;
        }
        .success-card {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 2rem; border-radius: 15px; color: white; text-align: center;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2); margin: 2rem 0;
        }
        .danger-card {
            background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
            padding: 2rem; border-radius: 15px; color: white; text-align: center;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2); margin: 2rem 0;
        }
        .info-box {
            background: #6c87bd; padding: 1rem; border-radius: 10px;
            border-left: 4px solid #667eea; margin: 1rem 0;
        }
        h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-size: 3rem; font-weight: 800; text-align: center; margin-bottom: 0.5rem;
        }
        .subtitle { text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem; }
        .section-header {
            color: #667eea; font-size: 1.3rem; font-weight: 600;
            margin-top: 1.5rem; margin-bottom: 1rem; border-bottom: 2px solid #667eea;
            padding-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1>💰 Loan Approval Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>AI-powered loan decision system with 70% confidence threshold</p>", unsafe_allow_html=True)
    
    pipeline = load_pipeline()
    
    st.markdown("""
        <div class='info-box'>
            ✅ <strong>Status:</strong> Ready
        </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📝 Application Form", "ℹ️ About"])
    
    with tab1:
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("<div class='section-header'>👤 Personal Information</div>", unsafe_allow_html=True)
            
            col1a, col1b = st.columns(2)
            with col1a:
                applicant_id = st.number_input("📋 Applicant ID", min_value=1, value=1, step=1)
                age = st.number_input("🎂 Age", min_value=18, max_value=100, value=30)
                gender = st.selectbox("⚧ Gender", ["Male", "Female"])
            with col1b:
                marital_status = st.selectbox("💑 Marital Status", ["Single", "Married"])
                dependents = st.number_input("👨‍👩‍👧‍👦 Dependents", min_value=0, max_value=10, value=0)
                education_level = st.selectbox("🎓 Education", ["Graduate", "Not Graduate"])
            
            st.markdown("<div class='section-header'>💼 Employment Details</div>", unsafe_allow_html=True)
            
            employment_status = st.selectbox("👔 Employment Status", ["Salaried", "Self-employed", "Unemployed"])
            employer_category = st.selectbox("🏢 Employer Type", ["Private", "Government", "MNC", "Unemployed"])
            
            col1c, col1d = st.columns(2)
            with col1c:
                applicant_income = st.number_input("💵 Monthly Income ($)", min_value=0.0, value=5000.0, step=100.0)
            with col1d:
                coapplicant_income = st.number_input("💰 Co-applicant Income ($)", min_value=0.0, value=0.0, step=100.0)
        
        with col2:
            st.markdown("<div class='section-header'>📊 Financial Profile</div>", unsafe_allow_html=True)
            
            col2a, col2b = st.columns(2)
            with col2a:
                credit_score = st.number_input("⭐ Credit Score", min_value=300, max_value=850, value=650)
                existing_loans = st.number_input("📑 Existing Loans", min_value=0, max_value=20, value=0)
            with col2b:
                dti_ratio = st.number_input("📈 Debt-to-Income", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
                savings = st.number_input("🏦 Savings ($)", min_value=0.0, value=10000.0, step=500.0)
            
            st.markdown("<div class='section-header'>🏠 Loan Details</div>", unsafe_allow_html=True)
            
            col2c, col2d = st.columns(2)
            with col2c:
                loan_amount = st.number_input("💳 Loan Amount ($)", min_value=0.0, value=10000.0, step=500.0)
                loan_term = st.number_input("📅 Term (months)", min_value=1, max_value=480, value=36)
            with col2d:
                loan_purpose = st.selectbox("🎯 Purpose", ["Personal", "Business", "Car", "Home", "Education"])
                property_area = st.selectbox("🌍 Property Area", ["Urban", "Semiurban", "Rural"])
            
            collateral_value = st.number_input("💎 Collateral Value ($)", min_value=0.0, value=20000.0, step=500.0)
        
        input_data = {
            "Applicant_ID": applicant_id, "Applicant_Income": applicant_income, "Coapplicant_Income": coapplicant_income,
            "Employment_Status": employment_status, "Age": age, "Marital_Status": marital_status, "Dependents": dependents,
            "Credit_Score": credit_score, "Existing_Loans": existing_loans, "DTI_Ratio": dti_ratio, "Savings": savings,
            "Collateral_Value": collateral_value, "Loan_Amount": loan_amount, "Loan_Term": loan_term, "Loan_Purpose": loan_purpose,
            "Property_Area": property_area, "Education_Level": education_level, "Gender": gender, "Employer_Category": employer_category,
        }
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🔮 Analyze Application", type="primary", use_container_width=True)
        
        if predict_button:
            with st.spinner("Analyzing..."):
                prediction, probability = predict_loan_approval(pipeline, input_data)
                
                if prediction is not None:
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    if probability is not None:
                        prob_approved = probability[1] if len(probability) > 1 else probability[0]
                        prob_percentage = prob_approved * 100
                        
                        if prob_percentage >= 70:
                            st.markdown(f"""
                                <div class='success-card'>
                                    <h2 style='color: white; margin: 0;'>✅ LOAN APPROVED</h2>
                                    <p style='font-size: 1.5rem; margin: 1rem 0;'>{prob_percentage:.1f}% Confidence</p>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div class='danger-card'>
                                    <h2 style='color: white; margin: 0;'>❌ LOAN NOT APPROVED</h2>
                                    <p style='font-size: 1.5rem; margin: 1rem 0;'>{prob_percentage:.1f}% Confidence</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            if prob_percentage >= 55:
                                st.warning("⚠️ **Borderline Case**: Try improving credit score or collateral.")
                        
                        col_m1, col_m2, col_m3 = st.columns(3)
                        
                        with col_m1:
                            delta_color = "normal" if prob_percentage >= 70 else "inverse"
                            st.metric("📊 Probability", f"{prob_percentage:.1f}%", delta=f"{prob_percentage - 70:.1f}%", delta_color=delta_color)
                        
                        with col_m2:
                            if prob_percentage >= 75: risk = "🟢 Low Risk"
                            elif prob_percentage >= 55: risk = "🟡 Moderate"
                            else: risk = "🔴 High Risk"
                            st.metric("🎯 Risk", risk)
                        
                        with col_m3:
                            st.metric("⚖️ Threshold", "70%")
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.progress(float(prob_approved))
                        st.caption(f"Confidence: {prob_percentage:.1f}%")
                        
                        with st.expander("📋 View Details"):
                            s1, s2 = st.columns(2)
                            with s1:
                                st.markdown("**Personal**")
                                p_data = {k:v for k,v in input_data.items() if k in ["Applicant_ID","Age","Gender","Marital_Status","Dependents","Education_Level","Employment_Status","Employer_Category"]}
                                st.dataframe(pd.DataFrame([p_data]).T)
                            with s2:
                                st.markdown("**Financial**")
                                f_data = {k:v for k,v in input_data.items() if k not in ["Applicant_ID","Age","Gender","Marital_Status","Dependents","Education_Level","Employment_Status","Employer_Category"]}
                                st.dataframe(pd.DataFrame([f_data]).T)
                    
                    else:
                        if prediction == 1 or prediction == "Yes": st.success("✅ Approved")
                        else: st.error("❌ Not Approved")
    
    with tab2:
        st.markdown("### ℹ️ About")
        st.write("This tool uses AI to predict loan approval.")
        st.write("- **Approved**: > 70%")
        st.write("- **Borderline**: 55-70%")
        st.write("- **Rejected**: < 55%")

if __name__ == "__main__":
    main()
