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
        layout="wide"
    )
    
    # Title and description
    st.title("💰 Loan Approval Prediction System")
    st.markdown("Enter applicant information below to predict loan approval status")
    
    # Load pipeline
    pipeline = load_pipeline()
    st.success("✅ Pipeline loaded successfully!")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        applicant_id = st.number_input("Applicant ID", min_value=1, value=1, step=1)
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
        education_level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
        
        st.subheader("Employment Information")
        employment_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])
        employer_category = st.selectbox("Employer Category", ["Private", "Government", "MNC", "Unemployed"])
        applicant_income = st.number_input("Applicant Income ($)", min_value=0.0, value=5000.0, step=100.0)
        coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0.0, value=0.0, step=100.0)
    
    with col2:
        st.subheader("Financial Information")
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
        existing_loans = st.number_input("Number of Existing Loans", min_value=0, max_value=20, value=0)
        savings = st.number_input("Savings ($)", min_value=0.0, value=10000.0, step=500.0)
        dti_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        
        st.subheader("Loan Details")
        loan_amount = st.number_input("Loan Amount ($)", min_value=0.0, value=10000.0, step=500.0)
        loan_term = st.number_input("Loan Term (months)", min_value=1, max_value=480, value=36)
        loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Business", "Car", "Home", "Education"])
        collateral_value = st.number_input("Collateral Value ($)", min_value=0.0, value=20000.0, step=500.0)
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    
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
    
    # Prediction button
    st.markdown("---")
    if st.button("🔮 Predict Loan Approval", type="primary", use_container_width=True):
        with st.spinner("Analyzing application..."):
            prediction, probability = predict_loan_approval(pipeline, input_data)
            
            if prediction is not None:
                st.markdown("---")
                st.subheader("Prediction Result")
                
                # Display result with color coding
                if prediction == 1 or prediction == "Yes":
                    st.success("✅ **Loan Approved!**")
                    result_color = "green"
                else:
                    st.error("❌ **Loan Not Approved**")
                    result_color = "red"
                
                # Display probability if available
                if probability is not None:
                    prob_approved = probability[1] if len(probability) > 1 else probability[0]
                    st.metric(
                        label="Approval Probability", 
                        value=f"{prob_approved:.1%}"
                    )
                    
                    # Progress bar for visual representation
                    st.progress(float(prob_approved))
                
                # Show input summary
                with st.expander("📋 View Application Summary"):
                    summary_df = pd.DataFrame([input_data]).T
                    summary_df.columns = ["Value"]
                    st.dataframe(summary_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 0.8em;'>
        💡 This is a machine learning-based prediction system. 
        Final loan decisions should involve human review and additional verification.
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
