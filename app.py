
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
    """Load the pre-trained loan approval pipeline from pickle file"""
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
    """Prepare input data with one-hot encoding to match training format"""
    
    input_df = pd.DataFrame([input_data])
    
    numeric_features = [
        "Applicant_ID", "Applicant_Income", "Coapplicant_Income",
        "Age", "Dependents", "Credit_Score", "Existing_Loans",
        "DTI_Ratio", "Savings", "Collateral_Value", "Loan_Amount", "Loan_Term"
    ]
    
    categorical_features = [
        "Employment_Status", "Marital_Status", "Loan_Purpose",
        "Property_Area", "Education_Level", "Gender", "Employer_Category"
    ]
    
    numeric_df = input_df[numeric_features].copy()
    categorical_df = input_df[categorical_features].copy()
    categorical_encoded = pd.get_dummies(categorical_df, drop_first=False)
    
    processed_df = pd.concat([numeric_df, categorical_encoded], axis=1)
    
    for col in expected_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0
    
    processed_df = processed_df[expected_columns]
    return processed_df

# =========================================
# Prediction function
# =========================================

def predict_loan_approval(pipeline, input_data):
    """Make prediction using the loaded pipeline"""
    try:
        if hasattr(pipeline, 'feature_names_in_'):
            expected_columns = pipeline.feature_names_in_.tolist()
        elif hasattr(pipeline, 'get_feature_names_out'):
            expected_columns = pipeline.get_feature_names_out().tolist()
        else:
            expected_columns = None
        
        if expected_columns:
            input_df = prepare_input(input_data, expected_columns)
        else:
            input_df = pd.DataFrame([input_data])
        
        prediction = pipeline.predict(input_df)
        
        if hasattr(pipeline, 'predict_proba'):
            probability = pipeline.predict_proba(input_df)
            return prediction[0], probability[0]
        else:
            return prediction[0], None
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# =========================================
# Visualization Functions (Glass & Neon Theme)
# =========================================

def create_gauge_chart(probability):
    """Create a neon gauge chart"""
    if not PLOTLY_AVAILABLE:
        return None
        
    # Determine color based on probability
    if probability >= 0.70:
        bar_color = "#00f260" # Neon Green
        step_colors = ["#2c0a0a", "#1a2e0a", "#0a2e0a"]
    elif probability >= 0.55:
        bar_color = "#f7971e" # Neon Orange
        step_colors = ["#2c0a0a", "#2e1a0a", "#0a2e0a"]
    else:
        bar_color = "#ff416c" # Neon Red
        step_colors = ["#2e0a0a", "#2e1a0a", "#0a2e0a"]
        
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'font': {'size': 42, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.2)"},
            'bar': {'color': bar_color, 'thickness': 0.4},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 55], 'color': 'rgba(255, 65, 108, 0.1)'},
                {'range': [55, 70], 'color': 'rgba(247, 151, 30, 0.1)'},
                {'range': [70, 100], 'color': 'rgba(0, 242, 96, 0.1)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 2},
                'thickness': 0.5,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Inter, sans-serif"}
    )
    
    return fig

def create_simple_gauge(probability):
    """Fallback gauge"""
    prob_percentage = probability * 100
    filled = int(prob_percentage / 5)
    empty = 20 - filled
    bar = "█" * filled + "░" * empty
    
    if prob_percentage >= 70:
        color = "#00f260"
        status = "✓ APPROVED"
    elif prob_percentage >= 55:
        color = "#f7971e"
        status = "⚠ BORDERLINE"
    else:
        color = "#ff416c"
        status = "✗ HIGH RISK"
    
    return f"""
    <div style='text-align: center; padding: 2rem; background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); border-radius: 16px; border: 1px solid rgba(255,255,255,0.1);'>
        <div style='font-size: 3rem; font-weight: 700; color: white; margin-bottom: 0.5rem; text-shadow: 0 0 10px rgba(255,255,255,0.3);'>
            {prob_percentage:.1f}%
        </div>
        <div style='font-size: 1.2rem; color: {color}; font-weight: 600; margin-bottom: 1rem; text-shadow: 0 0 10px {color};'>
            {status}
        </div>
        <div style='font-family: monospace; font-size: 1.5rem; letter-spacing: 2px; color: {color};'>
            {bar}
        </div>
    </div>
    """

def create_risk_breakdown_chart(prob_percentage):
    """Create a horizontal bar chart showing risk breakdown - Dark Mode"""
    if not PLOTLY_AVAILABLE:
        return None
    
    categories = ['High Risk<br><55%', 'Borderline<br>55-69%', 'Approved<br>≥70%']
    values = [55, 15, 30]
    # Darker, transparent colors for the background
    colors = ['rgba(255, 65, 108, 0.2)', 'rgba(247, 151, 30, 0.2)', 'rgba(0, 242, 96, 0.2)']
    
    fig = go.Figure()
    
    cumulative = 0
    for i, (cat, val, color) in enumerate(zip(categories, values, colors)):
        fig.add_trace(go.Bar(
            y=['Risk Level'],
            x=[val],
            orientation='h',
            name=cat,
            marker=dict(color=color, line=dict(color='rgba(255,255,255,0.2)', width=1)),
            text=cat,
            textposition='inside',
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # Marker color
    marker_color = "#ff416c" if prob_percentage < 55 else "#f7971e" if prob_percentage < 70 else "#00f260"
    
    fig.add_trace(go.Scatter(
        x=[prob_percentage],
        y=['Risk Level'],
        mode='markers+text',
        marker=dict(size=20, color=marker_color, symbol='diamond', line=dict(color='white', width=2)),
        text=[f'{prob_percentage:.1f}%'],
        textposition='top center',
        textfont=dict(size=14, color='white', family='Inter'),
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
            tickfont=dict(size=11, color='rgba(255,255,255,0.7)')
        ),
        yaxis=dict(showticklabels=False),
        font={'family': "Inter, sans-serif"}
    )
    
    return fig

def create_simple_risk_bar(prob_percentage):
    """Simple risk bar for no plotly"""
    if prob_percentage < 55:
        zone = "HIGH RISK"
        zone_color = "#ff416c"
    elif prob_percentage < 70:
        zone = "BORDERLINE"
        zone_color = "#f7971e"
    else:
        zone = "APPROVED"
        zone_color = "#00f260"
    
    return f"""
    <div style='background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); padding: 1.5rem; border-radius: 16px; border: 1px solid rgba(255,255,255,0.1);'>
        <div style='font-weight: 600; margin-bottom: 1rem; color: white; text-shadow: 0 0 10px rgba(255,255,255,0.3);'>Risk Assessment Scale</div>
        <div style='display: flex; height: 40px; border-radius: 8px; overflow: hidden; margin-bottom: 1rem;'>
            <div style='flex: 55; background: rgba(255, 65, 108, 0.3); display: flex; align-items: center; justify-content: center; font-size: 0.75rem; color: #ffcccb; border-right: 1px solid rgba(0,0,0,0.2);'>
                &lt;55% Risk
            </div>
            <div style='flex: 15; background: rgba(247, 151, 30, 0.3); display: flex; align-items: center; justify-content: center; font-size: 0.75rem; color: #ffe4b5; border-right: 1px solid rgba(0,0,0,0.2);'>
                55-69%
            </div>
            <div style='flex: 30; background: rgba(0, 242, 96, 0.3); display: flex; align-items: center; justify-content: center; font-size: 0.75rem; color: #ccffcc;'>
                ≥70% Good
            </div>
        </div>
        <div style='position: relative; height: 30px; background: rgba(0,0,0,0.3); border-radius: 8px;'>
            <div style='position: absolute; left: {prob_percentage}%; transform: translateX(-50%); top: -5px;'>
                <div style='background: {zone_color}; color: #000; padding: 0.25rem 0.75rem; border-radius: 6px; font-weight: 600; font-size: 0.875rem; white-space: nowrap; box-shadow: 0 0 15px {zone_color};'>
                    {prob_percentage:.1f}%
                </div>
                <div style='width: 0; height: 0; border-left: 6px solid transparent; border-right: 6px solid transparent; border-top: 6px solid {zone_color}; margin: 0 auto;'></div>
            </div>
        </div>
    </div>
    """

def create_feature_importance_chart(input_data):
    """Neon horizontal bar chart"""
    if not PLOTLY_AVAILABLE:
        return None
    
    factors = {
        'Credit Score': min(input_data['Credit_Score'] / 850 * 100, 100),
        'Income': min((input_data['Applicant_Income'] + input_data['Coapplicant_Income']) / 200 * 100, 100) if input_data['Applicant_Income'] > 0 else 0,
        'DTI Ratio': (1 - input_data['DTI_Ratio']) * 100,
        'Savings': min(input_data['Savings'] / 500 * 100, 100) if input_data['Savings'] > 0 else 0,
        'Collateral': min(input_data['Collateral_Value'] / 1000 * 100, 100) if input_data['Collateral_Value'] > 0 else 0,
    }
    
    df = pd.DataFrame(list(factors.items()), columns=['Factor', 'Score'])
    df = df.sort_values('Score', ascending=True)
    
    # Neon Palette
    colors = ['#00f260' if score >= 70 else '#f7971e' if score >= 50 else '#ff416c' for score in df['Score']]
    
    fig = go.Figure(go.Bar(
        x=df['Score'],
        y=df['Factor'],
        orientation='h',
        marker=dict(color=colors, cornerradius=8, opacity=0.8, line=dict(color='white', width=1)),
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
            gridcolor='rgba(255,255,255,0.1)',
            range=[0, 105],
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=13, color='white')
        ),
        font={'family': "Inter, sans-serif", 'color': 'white'}
    )
    
    return fig

def create_simple_factors(input_data):
    """Simple factor display"""
    factors = {
        'Credit Score': min(input_data['Credit_Score'] / 850 * 100, 100),
        'Income': min((input_data['Applicant_Income'] + input_data['Coapplicant_Income']) / 200 * 100, 100) if input_data['Applicant_Income'] > 0 else 0,
        'DTI Ratio': (1 - input_data['DTI_Ratio']) * 100,
        'Savings': min(input_data['Savings'] / 500 * 100, 100) if input_data['Savings'] > 0 else 0,
        'Collateral': min(input_data['Collateral_Value'] / 1000 * 100, 100) if input_data['Collateral_Value'] > 0 else 0,
    }
    
    html = "<div style='background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); padding: 1.5rem; border-radius: 16px; border: 1px solid rgba(255,255,255,0.1);'>"
    html += "<div style='font-weight: 600; margin-bottom: 1rem; color: white; text-shadow: 0 0 10px rgba(255,255,255,0.3);'>Key Factors Analysis</div>"
    
    for factor, score in sorted(factors.items(), key=lambda x: x[1], reverse=True):
        if score >= 70:
            color = "#00f260"
        elif score >= 50:
            color = "#f7971e"
        else:
            color = "#ff416c"
        
        html += f"""
        <div style='margin-bottom: 0.75rem;'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 0.25rem;'>
                <span style='font-size: 0.875rem; color: rgba(255,255,255,0.9);'>{factor}</span>
                <span style='font-size: 0.875rem; font-weight: 600; color: {color}; text-shadow: 0 0 5px {color};'>{score:.0f}%</span>
            </div>
            <div style='height: 8px; background: rgba(0,0,0,0.3); border-radius: 4px; overflow: hidden;'>
                <div style='height: 100%; background: {color}; width: {score}%; transition: width 0.3s ease; box-shadow: 0 0 10px {color};'></div>
            </div>
        </div>
        """
    
    html += "</div>"
    return html

# =========================================
# Streamlit UI
# =========================================

def main():
    st.set_page_config(
        page_title="Loan Approval System",
        page_icon="💎",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for glass-like aesthetic and footer
    st.markdown("""
        <style>
        /* Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Background - Deep Space Gradient */
        .stApp {
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            background-attachment: fixed;
        }
        
        /* Text Colors */
        h1, h2, h3, h4, h5, h6, p, span, div, label {
            font-family: 'Inter', sans-serif;
            color: rgba(255, 255, 255, 0.95) !important;
        }
        
        /* Glass Card Class */
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 1.75rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            height: 100%;
        }
        
        .glass-card:hover {
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
        }

        /* Section Title */
        .section-title {
            color: rgba(255, 255, 255, 0.6);
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            margin-bottom: 1.25rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Main Headers */
        h1 {
            font-size: 2.5rem;
            font-weight: 800;
            background: -webkit-linear-gradient(45deg, #fff, #a5b4fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
            text-shadow: 0 0 30px rgba(165, 180, 252, 0.3);
        }
        
        .subtitle {
            color: rgba(255, 255, 255, 0.6);
            font-size: 1rem;
            margin-bottom: 2rem;
            font-weight: 300;
        }

        /* Inputs - Dark Glass */
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > div,
        textarea {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: white;
        }
        
        .stNumberInput > div > div > input:focus,
        .stSelectbox > div > div > div:focus {
            border-color: #a5b4fc;
            box-shadow: 0 0 0 3px rgba(165, 180, 252, 0.1);
        }

        /* Inputs Labels */
        label {
            font-weight: 500;
            color: rgba(255, 255, 255, 0.8) !important;
        }

        /* Primary Button */
        .stButton > button[kind="primary"] {
            width: 100%;
            height: 3.5rem;
            font-size: 1rem;
            font-weight: 700;
            border-radius: 16px;
            border: none;
            background: linear-gradient(90deg, #00f260 0%, #0575E6 100%);
            color: #000;
            box-shadow: 0 0 20px rgba(0, 242, 96, 0.4);
            transition: all 0.3s ease;
        }
        
        .stButton > button[kind="primary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 0 30px rgba(0, 242, 96, 0.6);
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            background: rgba(255, 255, 255, 0.05);
            padding: 0.5rem;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            color: rgba(255, 255, 255, 0.6);
        }
        
        .stTabs [aria-selected="true"] {
            background: rgba(255, 255, 255, 0.15);
            color: white;
            box-shadow: 0 0 15px rgba(255, 255, 255, 0.1);
        }

        /* Metrics */
        div[data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
            color: white;
        }
        
        div[data-testid="stMetricLabel"] {
            font-size: 0.85rem;
            font-weight: 500;
            color: rgba(255, 255, 255, 0.6);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Result Cards */
        .result-success {
            background: rgba(0, 242, 96, 0.1);
            border: 2px solid #00f260;
            box-shadow: 0 0 40px rgba(0, 242, 96, 0.2);
        }
        
        .result-danger {
            background: rgba(255, 65, 108, 0.1);
            border: 2px solid #ff416c;
            box-shadow: 0 0 40px rgba(255, 65, 108, 0.2);
        }
        
        /* Signature Label (Footer) */
        .chef-label {
            text-align: center;
            margin-top: 3rem;
            padding: 1rem;
            color: rgba(255, 255, 255, 0.4);
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            letter-spacing: 2px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        .chef-label span {
            background: linear-gradient(90deg, #FFD700, #fdb931);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            font-size: 1.1rem;
            text-transform: uppercase;
        }

        </style>
    """, unsafe_allow_html=True)
    
    # Header
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.markdown("<h1>Loan Approval System</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>Next-Gen Credit Intelligence</p>", unsafe_allow_html=True)
    with col_header2:
        st.markdown("<div style='text-align: right; padding-top: 1rem;'><span style='background: rgba(0, 242, 96, 0.2); color: #00f260; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.8rem; font-weight: 700; border: 1px solid rgba(0, 242, 96, 0.3);'>✓ AI MODEL ACTIVE</span></div>", unsafe_allow_html=True)
    
    pipeline = load_pipeline()
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Application", "Information"])
    
    with tab1:
        col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
        
        with col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Personal Identity</div>", unsafe_allow_html=True)
            applicant_id = st.number_input("Applicant ID", min_value=1, value=1, step=1)
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
            dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)
            education_level = st.selectbox("Education", ["Graduate", "Not Graduate"])
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Employment Details</div>", unsafe_allow_html=True)
            employment_status = st.selectbox("Status", ["Salaried", "Self-employed", "Unemployed"])
            employer_category = st.selectbox("Employer Type", ["Private", "Government", "MNC", "Unemployed"])
            applicant_income = st.number_input("Monthly Income ($)", min_value=0.0, value=5000.0, step=100.0)
            coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0.0, value=0.0, step=100.0)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Financial Health</div>", unsafe_allow_html=True)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
            existing_loans = st.number_input("Existing Loans", min_value=0, max_value=20, value=0)
            dti_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
            savings = st.number_input("Savings ($)", min_value=0.0, value=10000.0, step=500.0)
            collateral_value = st.number_input("Collateral Value ($)", min_value=0.0, value=20000.0, step=500.0)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Loan Configuration</div>", unsafe_allow_html=True)
            loan_amount = st.number_input("Loan Amount ($)", min_value=0.0, value=10000.0, step=500.0)
            loan_term = st.number_input("Term (months)", min_value=1, max_value=480, value=36)
            loan_purpose = st.selectbox("Purpose", ["Personal", "Business", "Car", "Home", "Education"])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Data Prep
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
        
        predict_button = st.button("Analyze Application", type="primary")
        
        if predict_button:
            with st.spinner("Analyzing credit profile..."):
                prediction, probability = predict_loan_approval(pipeline, input_data)
                
                if prediction is not None and probability is not None:
                    prob_approved = probability[1] if len(probability) > 1 else probability[0]
                    prob_percentage = prob_approved * 100
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Result Card with Neon Glow
                    if prob_percentage >= 70:
                        st.markdown(f"""
                            <div class='glass-card result-success' style='text-align: center;'>
                                <h2 style='color: #00f260; margin: 0; font-size: 2.5rem; font-weight: 800; text-shadow: 0 0 20px rgba(0, 242, 96, 0.5);'>APPROVED</h2>
                                <p style='color: rgba(255,255,255,0.8); font-size: 1.1rem; margin-top: 0.5rem;'>Application meets risk criteria</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class='glass-card result-danger' style='text-align: center;'>
                                <h2 style='color: #ff416c; margin: 0; font-size: 2.5rem; font-weight: 800; text-shadow: 0 0 20px rgba(255, 65, 108, 0.5);'>REJECTED</h2>
                                <p style='color: rgba(255,255,255,0.8); font-size: 1.1rem; margin-top: 0.5rem;'>Below threshold requirements</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Metrics
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4, gap="medium")
                    
                    with metric_col1:
                        st.markdown("<div class='glass-card' style='text-align:center;'>", unsafe_allow_html=True)
                        st.metric("Confidence", f"{prob_percentage:.1f}%")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with metric_col2:
                        st.markdown("<div class='glass-card' style='text-align:center;'>", unsafe_allow_html=True)
                        st.metric("Threshold", "70.0%")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with metric_col3:
                        st.markdown("<div class='glass-card' style='text-align:center;'>", unsafe_allow_html=True)
                        delta = f"{prob_percentage - 70:+.1f}%"
                        st.metric("Delta", delta)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with metric_col4:
                        st.markdown("<div class='glass-card' style='text-align:center;'>", unsafe_allow_html=True)
                        risk = "Low" if prob_percentage >= 75 else "Medium" if prob_percentage >= 55 else "High"
                        st.metric("Risk", risk)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Charts
                    chart_col1, chart_col2 = st.columns(2, gap="medium")
                    
                    with chart_col1:
                        if PLOTLY_AVAILABLE:
                            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                            st.markdown("**Approval Confidence**")
                            gauge_fig = create_gauge_chart(prob_approved)
                            st.plotly_chart(gauge_fig, use_container_width=True, config={'displayModeBar': False})
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            simple_gauge = create_simple_gauge(prob_approved)
                            st.markdown(simple_gauge, unsafe_allow_html=True)
                    
                    with chart_col2:
                        if PLOTLY_AVAILABLE:
                            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                            st.markdown("**Key Factors Analysis**")
                            factors_fig = create_feature_importance_chart(input_data)
                            st.plotly_chart(factors_fig, use_container_width=True, config={'displayModeBar': False})
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            simple_factors = create_simple_factors(input_data)
                            st.markdown(simple_factors, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Risk position chart
                    if PLOTLY_AVAILABLE:
                        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                        st.markdown("**Risk Assessment Scale**")
                        risk_fig = create_risk_breakdown_chart(prob_percentage)
                        st.plotly_chart(risk_fig, use_container_width=True, config={'displayModeBar': False})
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        simple_risk = create_simple_risk_bar(prob_percentage)
                        st.markdown(simple_risk, unsafe_allow_html=True)
    
    with tab2:
        info_col1, info_col2 = st.columns(2, gap="medium")
        
        with info_col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("### Decision Criteria")
            st.markdown("""
            The system evaluates applications based on a comprehensive analysis of financial and personal factors.
            
            **Approval Thresholds:**
            - ✓ Approved: ≥70% confidence
            - ⚠ Borderline: 55-69% confidence  
            - ✗ Rejected: <55% confidence
            """)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with info_col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("### Key Factors")
            st.markdown("""
            The model analyzes multiple dimensions:
            
            **Financial Profile:**
            - Credit score and history
            - Income stability and sources
            - Existing debt obligations
            - Savings and collateral
            """)
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### Important Notice")
        st.markdown("""
        This system provides AI-assisted credit recommendations. Final decisions should include manual review.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    # Footer label 'cooked by chef sachin'
    st.markdown("""
        <div class='chef-label'>
            COOKED BY <span>CHEF SACHIN</span>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
