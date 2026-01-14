import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

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
# Visualization Functions
# =========================================

def create_gauge_chart(probability):
    """Create a gauge chart for approval probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'font': {'size': 48, 'color': '#1f2937'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#e5e7eb"},
            'bar': {'color': "#3b82f6" if probability >= 0.70 else "#ef4444"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 55], 'color': '#fee2e2'},
                {'range': [55, 70], 'color': '#fef3c7'},
                {'range': [70, 100], 'color': '#d1fae5'}
            ],
            'threshold': {
                'line': {'color': "#1f2937", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#1f2937", 'family': "SF Pro Display, -apple-system, sans-serif"}
    )
    
    return fig

def create_risk_breakdown_chart(prob_percentage):
    """Create a horizontal bar chart showing risk breakdown"""
    
    categories = ['High Risk<br><55%', 'Borderline<br>55-69%', 'Approved<br>≥70%']
    values = [55, 15, 30]  # Width of each zone
    colors = ['#fecaca', '#fde68a', '#a7f3d0']
    
    fig = go.Figure()
    
    # Add bars for each zone
    cumulative = 0
    for i, (cat, val, color) in enumerate(zip(categories, values, colors)):
        fig.add_trace(go.Bar(
            y=['Risk Level'],
            x=[val],
            orientation='h',
            name=cat,
            marker=dict(color=color, line=dict(color='#e5e7eb', width=1)),
            text=cat,
            textposition='inside',
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # Add a marker for current probability
    fig.add_trace(go.Scatter(
        x=[prob_percentage],
        y=['Risk Level'],
        mode='markers+text',
        marker=dict(size=20, color='#1f2937', symbol='diamond', line=dict(color='white', width=3)),
        text=[f'{prob_percentage:.1f}%'],
        textposition='top center',
        textfont=dict(size=14, color='#1f2937', family='SF Pro Display, -apple-system, sans-serif'),
        showlegend=False,
        hovertemplate=f'Your Score: {prob_percentage:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        barmode='stack',
        height=150,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            showticklabels=True,
            range=[0, 100],
            ticksuffix='%',
            tickfont=dict(size=11, color='#6b7280')
        ),
        yaxis=dict(showticklabels=False),
        font={'family': "SF Pro Display, -apple-system, sans-serif"}
    )
    
    return fig

def create_feature_importance_chart(input_data):
    """Create a bar chart showing key factors"""
    
    # Calculate normalized scores for key factors
    factors = {
        'Credit Score': min(input_data['Credit_Score'] / 850 * 100, 100),
        'Income': min((input_data['Applicant_Income'] + input_data['Coapplicant_Income']) / 200 * 100, 100) if input_data['Applicant_Income'] > 0 else 0,
        'DTI Ratio': (1 - input_data['DTI_Ratio']) * 100,
        'Savings': min(input_data['Savings'] / 500 * 100, 100) if input_data['Savings'] > 0 else 0,
        'Collateral': min(input_data['Collateral_Value'] / 1000 * 100, 100) if input_data['Collateral_Value'] > 0 else 0,
    }
    
    df = pd.DataFrame(list(factors.items()), columns=['Factor', 'Score'])
    df = df.sort_values('Score', ascending=True)
    
    colors = ['#3b82f6' if score >= 70 else '#f59e0b' if score >= 50 else '#ef4444' for score in df['Score']]
    
    fig = go.Figure(go.Bar(
        x=df['Score'],
        y=df['Factor'],
        orientation='h',
        marker=dict(color=colors, cornerradius=8),
        text=[f"{val:.0f}%" for val in df['Score']],
        textposition='outside',
        hovertemplate='%{y}: %{x:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=0, r=40, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True,
            gridcolor='#f3f4f6',
            range=[0, 105],
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=13, color='#1f2937')
        ),
        font={'family': "SF Pro Display, -apple-system, sans-serif", 'color': '#1f2937'}
    )
    
    return fig

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
    
    # Apple-inspired minimal CSS
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        
        .main {
            background: #f9fafb;
            padding: 1rem 2rem;
        }
        
        .stButton>button {
            width: 100%;
            height: 3.5rem;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 12px;
            border: none;
            background: #1f2937;
            color: white;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .stButton>button:hover {
            background: #111827;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transform: translateY(-1px);
        }
        
        .stButton>button:active {
            transform: translateY(0px);
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .card {
            background: white;
            padding: 1.75rem;
            border-radius: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            border: 1px solid #e5e7eb;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            height: 100%;
        }
        
        .card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            transform: translateY(-2px);
        }
        
        .result-card-success {
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            padding: 2.5rem;
            border-radius: 20px;
            border: 2px solid #86efac;
            text-align: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .result-card-danger {
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            padding: 2.5rem;
            border-radius: 20px;
            border: 2px solid #fca5a5;
            text-align: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .section-title {
            color: #1f2937;
            font-size: 0.875rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e5e7eb;
        }
        
        h1 {
            color: #1f2937;
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            color: #6b7280;
            font-size: 1.125rem;
            font-weight: 400;
            margin-bottom: 2rem;
        }
        
        .metric-square {
            background: white;
            padding: 1.5rem;
            border-radius: 16px;
            border: 1px solid #e5e7eb;
            text-align: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            aspect-ratio: 1;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        .metric-square:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            transform: translateY(-2px);
        }
        
        .status-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.875rem;
            font-weight: 600;
            background: #f3f4f6;
            color: #1f2937;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: white;
            padding: 0.5rem;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 0.625rem 1.25rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background: #1f2937;
            color: white;
        }
        
        div[data-testid="stMetricValue"] {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1f2937;
        }
        
        div[data-testid="stMetricLabel"] {
            font-size: 0.875rem;
            font-weight: 500;
            color: #6b7280;
        }
        
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > div {
            border-radius: 10px;
            border: 1.5px solid #e5e7eb;
            transition: all 0.2s ease;
        }
        
        .stNumberInput > div > div > input:focus,
        .stSelectbox > div > div > div:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }
        
        [data-testid="stExpander"] {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Minimal header
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.markdown("<h1>Loan Approval System</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>AI-powered credit decision platform</p>", unsafe_allow_html=True)
    with col_header2:
        st.markdown("<div style='text-align: right; padding-top: 1.5rem;'><span class='status-badge'>✓ Model Ready</span></div>", unsafe_allow_html=True)
    
    # Load pipeline
    pipeline = load_pipeline()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Application", "Information"])
    
    with tab1:
        # Form in cards
        col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Personal</div>", unsafe_allow_html=True)
            applicant_id = st.number_input("Applicant ID", min_value=1, value=1, step=1, label_visibility="visible")
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
            dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)
            education_level = st.selectbox("Education", ["Graduate", "Not Graduate"])
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Employment</div>", unsafe_allow_html=True)
            employment_status = st.selectbox("Status", ["Salaried", "Self-employed", "Unemployed"])
            employer_category = st.selectbox("Employer Type", ["Private", "Government", "MNC", "Unemployed"])
            applicant_income = st.number_input("Monthly Income ($)", min_value=0.0, value=5000.0, step=100.0)
            coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0.0, value=0.0, step=100.0)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Financial</div>", unsafe_allow_html=True)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
            existing_loans = st.number_input("Existing Loans", min_value=0, max_value=20, value=0)
            dti_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
            savings = st.number_input("Savings ($)", min_value=0.0, value=10000.0, step=500.0)
            collateral_value = st.number_input("Collateral Value ($)", min_value=0.0, value=20000.0, step=500.0)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Loan Details</div>", unsafe_allow_html=True)
            loan_amount = st.number_input("Loan Amount ($)", min_value=0.0, value=10000.0, step=500.0)
            loan_term = st.number_input("Term (months)", min_value=1, max_value=480, value=36)
            loan_purpose = st.selectbox("Purpose", ["Personal", "Business", "Car", "Home", "Education"])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
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
        
        # Analyze button
        predict_button = st.button("Analyze Application", type="primary")
        
        if predict_button:
            with st.spinner("Analyzing..."):
                prediction, probability = predict_loan_approval(pipeline, input_data)
                
                if prediction is not None and probability is not None:
                    prob_approved = probability[1] if len(probability) > 1 else probability[0]
                    prob_percentage = prob_approved * 100
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Result card
                    if prob_percentage >= 70:
                        st.markdown(f"""
                            <div class='result-card-success'>
                                <h2 style='color: #166534; margin: 0; font-size: 2rem; font-weight: 700;'>Approved</h2>
                                <p style='color: #15803d; font-size: 1.125rem; margin-top: 0.5rem;'>Application meets requirements</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class='result-card-danger'>
                                <h2 style='color: #991b1b; margin: 0; font-size: 2rem; font-weight: 700;'>Not Approved</h2>
                                <p style='color: #dc2626; font-size: 1.125rem; margin-top: 0.5rem;'>Below approval threshold</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Metrics in square cards
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4, gap="medium")
                    
                    with metric_col1:
                        st.markdown("<div class='metric-square'>", unsafe_allow_html=True)
                        st.metric("Confidence", f"{prob_percentage:.1f}%")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with metric_col2:
                        st.markdown("<div class='metric-square'>", unsafe_allow_html=True)
                        st.metric("Threshold", "70.0%")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with metric_col3:
                        st.markdown("<div class='metric-square'>", unsafe_allow_html=True)
                        delta = f"{prob_percentage - 70:+.1f}%"
                        st.metric("Delta", delta)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with metric_col4:
                        st.markdown("<div class='metric-square'>", unsafe_allow_html=True)
                        if prob_percentage >= 75:
                            risk = "Low"
                        elif prob_percentage >= 55:
                            risk = "Medium"
                        else:
                            risk = "High"
                        st.metric("Risk", risk)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Charts
                    chart_col1, chart_col2 = st.columns(2, gap="medium")
                    
                    with chart_col1:
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.markdown("**Approval Confidence**")
                        gauge_fig = create_gauge_chart(prob_approved)
                        st.plotly_chart(gauge_fig, use_container_width=True, config={'displayModeBar': False})
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with chart_col2:
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.markdown("**Key Factors Analysis**")
                        factors_fig = create_feature_importance_chart(input_data)
                        st.plotly_chart(factors_fig, use_container_width=True, config={'displayModeBar': False})
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Risk position chart
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("**Risk Assessment Scale**")
                    risk_fig = create_risk_breakdown_chart(prob_percentage)
                    st.plotly_chart(risk_fig, use_container_width=True, config={'displayModeBar': False})
                    st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        info_col1, info_col2 = st.columns(2, gap="medium")
        
        with info_col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Decision Criteria")
            st.markdown("""
            The system evaluates applications based on a comprehensive analysis of financial and personal factors.
            
            **Approval Thresholds:**
            - ✓ Approved: ≥70% confidence
            - ⚠ Borderline: 55-69% confidence  
            - ✗ Rejected: <55% confidence
            
            **Risk Categories:**
            - Low Risk: 75%+ confidence
            - Medium Risk: 55-74% confidence
            - High Risk: Below 55% confidence
            """)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with info_col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Key Factors")
            st.markdown("""
            The model analyzes multiple dimensions:
            
            **Financial Profile:**
            - Credit score and history
            - Income stability and sources
            - Existing debt obligations
            - Savings and collateral
            
            **Personal Information:**
            - Employment status and type
            - Age and dependents
            - Education level
            
            **Loan Specifics:**
            - Loan amount and term
            - Purpose and property area
            """)
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Important Notice")
        st.markdown("""
        This system provides AI-assisted credit recommendations. Final decisions should include:
        - Manual review by qualified personnel
        - Verification of submitted information
        - Compliance with applicable lending regulations
        - Adherence to fair lending practices and anti-discrimination laws
        
        The model serves as a decision support tool and should not be the sole determinant for loan approval.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
