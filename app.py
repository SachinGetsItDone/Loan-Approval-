import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================================
# Load the pre-trained pipeline
# =========================================

@st.cache_resource
def load_pipeline():
    """Load the pre-trained loan approval pipeline from pickle file"""
    try:
        with open('loan_pipeline.pkl', 'rb') as f:
            pipeline = pickle.load(f)
        return pipeline
    except FileNotFoundError:
        st.error("⚠️ loan_pipeline.pkl not found. Please ensure the file is in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading pipeline: {str(e)}")
        st.stop()

# =========================================
# Feature Engineering
# =========================================

def prepare_input(input_data, expected_columns):
    """Prepare input data with one-hot encoding to match training format"""
    
    # Create base DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Separate numeric and categorical features
    numeric_features = [
        "Applicant_ID", "Applicant_Income", "Coapplicant_Income",
        "Age", "Dependents", "Credit_Score", "Existing_Loans",
        "DTI_Ratio", "Savings", "Collateral_Value", "Loan_Amount", "Loan_Term"
    ]
    
    categorical_features = [
        "Employment_Status", "Marital_Status", "Loan_Purpose",
        "Property_Area", "Education_Level", "Gender", "Employer_Category"
    ]
    
    # Extract numeric features
    numeric_df = input_df[numeric_features].copy()
    
    # One-hot encode categorical features
    categorical_df = input_df[categorical_features].copy()
    categorical_encoded = pd.get_dummies(categorical_df, drop_first=False)
    
    # Combine numeric and encoded categorical
    processed_df = pd.concat([numeric_df, categorical_encoded], axis=1)
    
    # Ensure all expected columns are present (add missing ones with 0)
    for col in expected_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0
    
    # Keep only expected columns in the correct order
    processed_df = processed_df[expected_columns]
    
    return processed_df

# =========================================
# Prediction function
# =========================================

def predict_loan_approval(pipeline, input_data):
    """Make prediction using the loaded pipeline"""
    try:
        # Get expected feature names from the pipeline
        if hasattr(pipeline, 'feature_names_in_'):
            expected_columns = pipeline.feature_names_in_.tolist()
        elif hasattr(pipeline, 'get_feature_names_out'):
            expected_columns = pipeline.get_feature_names_out().tolist()
        else:
            # Try to extract from the model within the pipeline
            if hasattr(pipeline, 'named_steps'):
                model = pipeline.named_steps.get('model') or pipeline.named_steps.get('classifier')
                if model and hasattr(model, 'feature_names_in_'):
                    expected_columns = model.feature_names_in_.tolist()
                else:
                    expected_columns = None
            else:
                expected_columns = None
        
        # Prepare input with proper encoding
        if expected_columns:
            input_df = prepare_input(input_data, expected_columns)
        else:
            # Fallback: create DataFrame and let pipeline handle it
            input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = pipeline.predict(input_df)
        
        # Get probability if available
        if hasattr(pipeline, 'predict_proba'):
            probability = pipeline.predict_proba(input_df)
            return prediction[0], probability[0]
        else:
            return prediction[0], None
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# =========================================
# Streamlit UI
# =========================================

def main():
    # Page configuration
    st.set_page_config(
        page_title="Loan Approval Predictor",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            height: 3.5rem;
            font-size: 1.2rem;
            font-weight: 600;
            border-radius: 10px;
            margin-top: 1rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .success-card {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            margin: 2rem 0;
        }
        .danger-card {
            background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            margin: 2rem 0;
        }
        .info-box {
            background: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            margin: 1rem 0;
        }
        h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            text-align: center;
            color: #666;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .section-header {
            color: #667eea;
            font-size: 1.3rem;
            font-weight: 600;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            border-bottom: 2px solid #667eea;
            padding-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header with gradient
    st.markdown("<h1>💰 Loan Approval Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>AI-powered loan decision system with 70% confidence threshold</p>", unsafe_allow_html=True)
    
    # Load pipeline
    pipeline = load_pipeline()
    
    # Success message with custom styling
    st.markdown("""
        <div class='info-box'>
            ✅ <strong>Pipeline Status:</strong> Model loaded and ready for predictions
        </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs(["📝 Application Form", "ℹ️ About"])
    
    with tab1:
        # Create two columns for form layout
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
        
        # Prepare input data
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
            "Employer_Category": employer_category,
        }
        
        # Prediction button with custom styling
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🔮 Analyze Application", type="primary", use_container_width=True)
        
        if predict_button:
            with st.spinner("🤖 AI is analyzing the application..."):
                prediction, probability = predict_loan_approval(pipeline, input_data)
                
                if prediction is not None:
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    if probability is not None:
                        prob_approved = probability[1] if len(probability) > 1 else probability[0]
                        prob_percentage = prob_approved * 100
                        
                        # Display result with custom cards
                        if prob_percentage >= 70:
                            st.markdown(f"""
                                <div class='success-card'>
                                    <h2 style='color: white; margin: 0;'>✅ LOAN APPROVED</h2>
                                    <p style='font-size: 1.5rem; margin: 1rem 0;'>{prob_percentage:.1f}% Confidence</p>
                                    <p style='opacity: 0.9;'>Exceeds 70% approval threshold</p>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div class='danger-card'>
                                    <h2 style='color: white; margin: 0;'>❌ LOAN NOT APPROVED</h2>
                                    <p style='font-size: 1.5rem; margin: 1rem 0;'>{prob_percentage:.1f}% Confidence</p>
                                    <p style='opacity: 0.9;'>Below 70% approval threshold</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            if prob_percentage >= 55:
                                st.warning("⚠️ **Borderline Case**: Consider improving credit score, reducing debt-to-income ratio, or increasing collateral.")
                        
                        # Detailed metrics in cards
                        col_m1, col_m2, col_m3 = st.columns(3)
                        
                        with col_m1:
                            delta_color = "normal" if prob_percentage >= 70 else "inverse"
                            st.metric(
                                label="📊 Approval Probability", 
                                value=f"{prob_percentage:.1f}%",
                                delta=f"{prob_percentage - 70:.1f}% from threshold",
                                delta_color=delta_color
                            )
                        
                        with col_m2:
                            if prob_percentage >= 75:
                                risk_level = "Low Risk"
                                risk_emoji = "🟢"
                            elif prob_percentage >= 55:
                                risk_level = "Moderate"
                                risk_emoji = "🟡"
                            else:
                                risk_level = "High Risk"
                                risk_emoji = "🔴"
                            
                            st.metric(
                                label="🎯 Risk Category",
                                value=f"{risk_emoji} {risk_level}"
                            )
                        
                        with col_m3:
                            st.metric(
                                label="⚖️ Decision Threshold",
                                value="70.0%",
                                delta="Required for approval"
                            )
                        
                        # Visual progress bar
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.progress(float(prob_approved))
                        st.caption(f"📈 Confidence Level: {prob_percentage:.1f}% | Threshold: 70%")
                        
                        # Application summary in expandable section
                        with st.expander("📋 View Complete Application Details"):
                            summary_col1, summary_col2 = st.columns(2)
                            
                            with summary_col1:
                                st.markdown("**Personal & Employment**")
                                personal_data = {k: v for k, v in input_data.items() if k in [
                                    "Applicant_ID", "Age", "Gender", "Marital_Status", "Dependents",
                                    "Education_Level", "Employment_Status", "Employer_Category"
                                ]}
                                st.dataframe(pd.DataFrame([personal_data]).T, use_container_width=True)
                            
                            with summary_col2:
                                st.markdown("**Financial & Loan Details**")
                                financial_data = {k: v for k, v in input_data.items() if k not in [
                                    "Applicant_ID", "Age", "Gender", "Marital_Status", "Dependents",
                                    "Education_Level", "Employment_Status", "Employer_Category"
                                ]}
                                st.dataframe(pd.DataFrame([financial_data]).T, use_container_width=True)
                    
                    else:
                        # Fallback if probability not available
                        if prediction == 1 or prediction == "Yes":
                            st.success("✅ **Loan Approved!**")
                        else:
                            st.error("❌ **Loan Not Approved**")
    
    with tab2:
        st.markdown("### 📖 About This System")
        
        st.markdown("""
        This **AI-powered Loan Approval System** uses machine learning to predict loan approval decisions 
        based on applicant information and financial profile.
        
        #### 🎯 Decision Criteria
        - **Approved**: Confidence ≥ 70%
        - **Borderline**: 55% - 69% confidence
        - **Not Approved**: < 55% confidence
        
        #### 📊 Risk Categories
        - 🟢 **Low Risk**: 75%+ confidence
        - 🟡 **Moderate Risk**: 55-74% confidence
        - 🔴 **High Risk**: Below 55% confidence
        
        #### ⚙️ Model Features
        The model analyzes **19 features** including:
        - Personal information (age, marital status, dependents)
        - Employment details (status, employer type, income)
        - Financial profile (credit score, existing loans, savings)
        - Loan specifics (amount, term, purpose, collateral)
        
        #### ⚠️ Important Note
        This system provides AI-assisted recommendations. Final loan decisions should involve:
        - Human review and verification
        - Additional documentation checks
        - Compliance with lending regulations
        - Fair lending practices
        """)
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: gray; font-size: 0.9em;'>
        💡 Powered by Machine Learning | Built with Streamlit
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
