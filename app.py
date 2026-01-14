@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Page background */
.main {
    background: linear-gradient(135deg, #0f172a, #020617);
    padding: 1.5rem 2rem;
}

/* Glass cards */
.card,
.metric-square,
.result-card-success,
.result-card-danger,
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 18px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    color: #e5e7eb;
}

/* Success / danger glow */
.result-card-success {
    border: 1px solid rgba(34, 197, 94, 0.6);
    box-shadow: 0 0 40px rgba(34, 197, 94, 0.25);
}

.result-card-danger {
    border: 1px solid rgba(239, 68, 68, 0.6);
    box-shadow: 0 0 40px rgba(239, 68, 68, 0.25);
}

/* Buttons */
.stButton > button {
    width: 100%;
    height: 3.5rem;
    font-size: 1rem;
    font-weight: 600;
    border-radius: 14px;
    border: none;
    background: linear-gradient(135deg, #38bdf8, #6366f1);
    color: white;
    box-shadow: 0 10px 30px rgba(99, 102, 241, 0.4);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 45px rgba(99, 102, 241, 0.6);
}

/* Inputs */
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background: rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    color: #e5e7eb !important;
}

/* Headings */
h1 {
    color: #f8fafc;
    font-size: 2.6rem;
    font-weight: 800;
}

.subtitle {
    color: #94a3b8;
}

/* Section title */
.section-title {
    color: #c7d2fe;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.2);
    padding-bottom: 0.4rem;
}

/* Tabs glass */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.08);
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.15);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #38bdf8, #6366f1);
    color: white;
    border-radius: 10px;
}

/* Metrics */
div[data-testid="stMetricValue"] {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f8fafc;
}

div[data-testid="stMetricLabel"] {
    color: #94a3b8;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 3rem;
    padding: 1.5rem;
    color: #94a3b8;
    font-size: 0.9rem;
    letter-spacing: 0.05em;
}
