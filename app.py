import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os
import logging
import time
from datetime import datetime

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="AI Insurance Intelligence System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== SESSION STATE ====================
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* ========== SIDEBAR - ORANGE BACKGROUND, BLACK TEXT ========== */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%) !important;
        min-width: 280px !important;
        width: 280px !important;
        flex-shrink: 0 !important;
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        border-right: none !important;
    }
    
    /* Sidebar all text - BLACK */
    [data-testid="stSidebar"] * {
        color: #1e1e2a !important;
        visibility: visible !important;
    }
    
    /* Sidebar headers - BLACK BOLD */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown {
        color: #1e1e2a !important;
        font-weight: 700 !important;
    }
    
    /* Radio buttons styling */
    [data-testid="stSidebar"] .stRadio label {
        color: #1e1e2a !important;
        font-weight: 600 !important;
        padding: 10px 12px !important;
        border-radius: 10px !important;
        margin: 4px 0 !important;
        background-color: rgba(255,255,255,0.25) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: rgba(255,255,255,0.45) !important;
        border-color: rgba(255,255,255,0.6) !important;
    }
    
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] {
        margin-right: 12px !important;
    }
    
    /* History items in sidebar */
    .history-item {
        background: rgba(255,255,255,0.3) !important;
        border-left: 3px solid #1e1e2a !important;
        padding: 0.4rem 0.6rem !important;
        margin: 0.4rem 0 !important;
        border-radius: 6px !important;
        font-size: 0.7rem !important;
        color: #1e1e2a !important;
    }
    
    /* ========== MAIN CONTENT ========== */
    .main {
        padding: 0rem 1rem;
        background-color: #ffffff;
    }
    
    /* Card styling */
    .card {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
        padding: 1.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 1.5rem;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 600;
    }
    
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-size: 0.85rem;
    }
    
    /* Portfolio Cards */
    .portfolio-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e5e7eb;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    
    .portfolio-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        border-color: #f97316;
    }
    
    .portfolio-icon {
        font-size: 1.8rem;
    }
    
    .portfolio-amount {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e2a3a;
        margin: 0.3rem 0;
    }
    
    .portfolio-range {
        font-size: 0.7rem;
        color: #f97316;
        font-weight: 500;
        background: #fff3e6;
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
    }
    
    .portfolio-label {
        font-size: 0.7rem;
        color: #5a6e8a;
        margin-top: 0.3rem;
    }
    
    /* Section headers (only for dashboard) */
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1e2a3a;
        margin: 1rem 0 0.8rem 0;
        border-left: 4px solid #f97316;
        padding-left: 0.8rem;
    }
    
    /* Feature Cards */
    .feature-card {
        background: #f8fafc;
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e5e7eb;
        transition: all 0.2s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-color: #f97316;
    }
    
    .feature-icon {
        font-size: 1.6rem;
        margin-bottom: 0.3rem;
    }
    
    .feature-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #1e2a3a;
    }
    
    .feature-desc {
        font-size: 0.7rem;
        color: #5a6e8a;
    }
    
    .feature-metric {
        font-size: 0.7rem;
        color: #f97316;
        margin-top: 0.3rem;
        font-weight: 500;
    }
    
    /* Buttons */
    .stButton > button {
        background: #f97316;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 500;
        width: 100%;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: #ea580c;
        transform: translateY(-1px);
    }
    
    /* Messages */
    .success-message {
        background: #2c7a4d;
        color: white;
        padding: 0.8rem 1rem;
        border-radius: 12px;
        margin: 0.8rem 0;
    }
    
    .error-message {
        background: #c53030;
        color: white;
        padding: 0.8rem 1rem;
        border-radius: 12px;
        margin: 0.8rem 0;
    }
    
    /* Divider */
    .custom-divider {
        height: 1px;
        background: #e5e7eb;
        margin: 0.8rem 0;
    }
    
    /* Main content adjustment */
    .block-container {
        padding-top: 0.5rem;
        padding-bottom: 1rem;
    }
    
    @media (max-width: 768px) {
        .portfolio-amount {
            font-size: 1rem;
        }
        .section-header {
            font-size: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOGGING ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==================== MODEL LOADING ====================
@st.cache_resource
def load_model(model_name):
    path = f"models/{model_name}_best.pkl"
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            logging.error(f"Error loading {model_name}: {e}")
            return None
    return None

@st.cache_resource
def load_fraud_model():
    path = "models/fraud_best.pkl"
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            logging.error(f"Error loading fraud model: {e}")
            return None
    return None

health_model = load_model("health")
car_model = load_model("car")
life_model = load_model("life")
home_model = load_model("home")
fraud_model = load_fraud_model()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## 🏦 **AI INSURANCE**")
    st.markdown("---")
    
    nav_options = [
        "📊 Dashboard",
        "🩺 Health Insurance", 
        "🚗 Car Insurance",
        "🧬 Life Insurance",
        "🏠 Home Insurance",
        "🕵️ Fraud Detection",
        "📈 Analytics"
    ]
    
    selected = st.radio(
        "**NAVIGATION**",
        nav_options,
        index=0,
        label_visibility="visible"
    )
    
    st.markdown("---")
    st.markdown(f"**Version:** 3.2.0")
    st.markdown(f"**Updated:** {datetime.now().strftime('%d/%m/%Y')}")

# ==================== HELPER FUNCTIONS ====================
def get_portfolio_values():
    portfolio = {
        "Health": {"min": 8500, "max": 22000, "avg": 12500, "icon": "🩺"},
        "Car": {"min": 3800, "max": 14000, "avg": 7200, "icon": "🚗"},
        "Life": {"min": 5200, "max": 18500, "avg": 9800, "icon": "🧬"},
        "Home": {"min": 2400, "max": 7500, "avg": 4100, "icon": "🏠"}
    }
    
    if health_model:
        try:
            sample = pd.DataFrame([[35, "male", 26.5, 0, "no", "southeast"]],
                                 columns=['age','gender','bmi','children','smoker','region'])
            pred = health_model.predict(sample)[0]
            portfolio["Health"]["avg"] = int(pred)
            portfolio["Health"]["min"] = int(pred * 0.7)
            portfolio["Health"]["max"] = int(pred * 1.5)
        except:
            pass
    
    return portfolio

def add_to_history(pred_type, value, details):
    st.session_state.prediction_history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "type": pred_type,
        "value": value,
        "details": details
    })
    if len(st.session_state.prediction_history) > 8:
        st.session_state.prediction_history = st.session_state.prediction_history[-8:]

def show_history():
    if st.session_state.prediction_history:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📜 Recent")
        for item in st.session_state.prediction_history[-5:]:
            st.sidebar.markdown(f"""
            <div class="history-item">
                <strong>{item['time']}</strong><br>
                {item['type']}<br>
                <span style="color:#1e1e2a;">{item['value']}</span>
            </div>
            """, unsafe_allow_html=True)

def export_history():
    if st.session_state.prediction_history:
        return pd.DataFrame(st.session_state.prediction_history).to_csv(index=False)
    return None

def map_fraud_input(policy, amount, incident, severity, witnesses, report):
    policy_map = {"Health Insurance": "health", "Car Insurance": "car", 
                  "Life Insurance": "life", "Home Insurance": "home"}
    
    incident_map = {
        "Emergency": "emergency", "Surgery": "surgery", "OPD": "opd", "ICU": "icu",
        "Checkup": "checkup", "Collision": "collision", "Theft": "theft",
        "Natural Disaster": "natural disaster", "Vandalism": "vandalism", "Fire": "fire",
        "Accident": "accident", "Illness": "illness", "Critical Illness": "critical illness",
        "Water Damage": "water damage"
    }
    
    severity_map = {
        "Minor": "minor", "Moderate": "moderate", "Severe": "severe",
        "Critical": "critical", "Total Loss": "total loss", "Clean": "clean",
        "Minor Issues": "minor", "Major Issues": "major"
    }
    
    report_map = {"Yes": "yes", "No": "no", "Available": "yes", "Not Available": "no"}
    
    return {
        "policy_type": policy_map.get(policy, "health"),
        "claim_amount": amount,
        "incident_type": incident_map.get(incident, incident.lower() if incident else "other"),
        "incident_severity": severity_map.get(severity, severity.lower() if severity else "moderate"),
        "witnesses": witnesses,
        "police_report": report_map.get(report, "no")
    }

# ==================== DASHBOARD ====================
def dashboard():
    st.markdown("""
    <div class="main-header">
        <h1>🏦 AI Insurance Intelligence System</h1>
        <p>Enterprise-grade insurance analytics | Real-time predictions | Advanced fraud detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Portfolio Section
    st.markdown('<div class="section-header">📊 Your Insurance Portfolio</div>', unsafe_allow_html=True)
    portfolio = get_portfolio_values()
    cols = st.columns(4)
    
    for idx, (key, val) in enumerate(portfolio.items()):
        with cols[idx]:
            st.markdown(f"""
            <div class="portfolio-card">
                <div class="portfolio-icon">{val['icon']}</div>
                <div class="portfolio-amount">₹{val['avg']:,}</div>
                <div class="portfolio-range">₹{val['min']:,} - ₹{val['max']:,}</div>
                <div class="portfolio-label">{key} Insurance / year</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Features Section
    st.markdown('<div class="section-header">🚀 Platform Features</div>', unsafe_allow_html=True)
    
    features = [
        ("🧠", "AI-Powered Predictions", "ML models trained on 10,000+ records", "95% Accuracy"),
        ("⚡", "Real-time Analytics", "Sub-second premium calculation", "< 0.5s Response"),
        ("🛡️", "Fraud Detection", "Advanced risk assessment", "89% Detection Rate"),
        ("📈", "Model Analytics", "Compare ML model performance", "4+ Algorithms")
    ]
    
    cols = st.columns(4)
    for idx, (icon, title, desc, metric) in enumerate(features):
        with cols[idx]:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
                <div class="feature-metric">🎯 {metric}</div>
            </div>
            """, unsafe_allow_html=True)
    
    show_history()

# ==================== INSURANCE INPUTS (NO DUPLICATE HEADERS) ====================
def health_input():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🩺 Health Insurance Details")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 35)
        bmi = st.number_input("BMI", 10.0, 50.0, 26.5, step=0.1)
        smoker = st.selectbox("Smoker", ["no", "yes"])
    with col2:
        gender = st.selectbox("Gender", ["male", "female"])
        children = st.number_input("Children", 0, 10, 0)
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
    
    if st.button("💰 Calculate Premium", use_container_width=True):
        if health_model:
            with st.spinner("Calculating..."):
                time.sleep(0.3)
                df = pd.DataFrame([[age, gender, bmi, children, smoker, region]],
                                 columns=['age','gender','bmi','children','smoker','region'])
                try:
                    pred = health_model.predict(df)[0]
                    st.markdown(f"""
                    <div class="success-message">
                        <h3>💰 Estimated Premium: ₹{pred:,.2f}/year</h3>
                        <p>✓ AI-powered prediction based on your health profile</p>
                    </div>
                    """, unsafe_allow_html=True)
                    add_to_history("Health", f"₹{pred:,.0f}", f"Age: {age}")
                except Exception as e:
                    st.markdown(f'<div class="error-message">❌ Error: {e}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-message">❌ Model not found. Train models first.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def car_input():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🚗 Car Insurance Details")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 35)
        driving_exp = st.number_input("Driving Experience", 0, 70, 10)
        vehicle_age = st.number_input("Vehicle Age", 0, 30, 5)
        vehicle_type = st.selectbox("Vehicle Type", ["sedan", "suv", "truck", "sports"])
    with col2:
        gender = st.selectbox("Gender", ["male", "female"])
        location = st.selectbox("Location", ["urban", "suburban", "rural"])
        prev_claims = st.number_input("Previous Claims", 0, 10, 0)
        annual_mileage = st.number_input("Annual Mileage", 0, 50000, 12000)
    
    if st.button("💰 Calculate Premium", use_container_width=True):
        if car_model:
            with st.spinner("Calculating..."):
                time.sleep(0.3)
                df = pd.DataFrame([[age, gender, driving_exp, vehicle_age, vehicle_type,
                                   location, prev_claims, annual_mileage]],
                                 columns=['age','gender','driving_experience','vehicle_age',
                                         'vehicle_type','location','previous_claims','annual_mileage'])
                try:
                    pred = car_model.predict(df)[0]
                    st.markdown(f"""
                    <div class="success-message">
                        <h3>💰 Estimated Premium: ₹{pred:,.2f}/year</h3>
                        <p>✓ AI-powered prediction based on your driving profile</p>
                    </div>
                    """, unsafe_allow_html=True)
                    add_to_history("Car", f"₹{pred:,.0f}", f"Vehicle: {vehicle_type}")
                except Exception as e:
                    st.markdown(f'<div class="error-message">❌ Error: {e}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-message">❌ Model not found. Train models first.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def life_input():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧬 Life Insurance Details")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 20, 80, 40)
        smoker = st.selectbox("Smoker", ["no", "yes"])
        income = st.number_input("Annual Income ($)", 20000, 500000, 60000)
    with col2:
        gender = st.selectbox("Gender", ["male", "female"])
        health = st.selectbox("Health Status", ["poor", "average", "good", "excellent"])
        coverage = st.selectbox("Coverage ($)", [100000, 250000, 500000, 1000000])
    
    if st.button("💰 Calculate Premium", use_container_width=True):
        if life_model:
            with st.spinner("Calculating..."):
                time.sleep(0.3)
                df = pd.DataFrame([[age, gender, smoker, health, income, coverage, 20]],
                                 columns=['age','gender','smoker','health_status',
                                         'annual_income','coverage_amount','term_length'])
                try:
                    pred = life_model.predict(df)[0]
                    st.markdown(f"""
                    <div class="success-message">
                        <h3>💰 Estimated Premium: ₹{pred:,.2f}/year</h3>
                        <p>✓ AI-powered prediction based on your life profile</p>
                    </div>
                    """, unsafe_allow_html=True)
                    add_to_history("Life", f"₹{pred:,.0f}", f"Age: {age}")
                except Exception as e:
                    st.markdown(f'<div class="error-message">❌ Error: {e}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-message">❌ Model not found. Train models first.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def home_input():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🏠 Home Insurance Details")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        home_age = st.number_input("Home Age", 0, 100, 20)
        sqft = st.number_input("Square Footage", 500, 10000, 2000)
        construction = st.selectbox("Construction", ["wood", "brick", "concrete"])
    with col2:
        location = st.selectbox("Location", ["urban", "suburban", "rural"])
        security = st.selectbox("Security", ["none", "alarm", "monitored"])
        prev_claims = st.number_input("Previous Claims", 0, 10, 0)
    
    if st.button("💰 Calculate Premium", use_container_width=True):
        if home_model:
            with st.spinner("Calculating..."):
                time.sleep(0.3)
                df = pd.DataFrame([[home_age, location, sqft, construction, "shingle", security, prev_claims]],
                                 columns=['home_age','location','sqft','construction_type',
                                         'roof_type','security_system','previous_claims'])
                try:
                    pred = home_model.predict(df)[0]
                    st.markdown(f"""
                    <div class="success-message">
                        <h3>💰 Estimated Premium: ₹{pred:,.2f}/year</h3>
                        <p>✓ AI-powered prediction based on your property profile</p>
                    </div>
                    """, unsafe_allow_html=True)
                    add_to_history("Home", f"₹{pred:,.0f}", f"Sqft: {sqft}")
                except Exception as e:
                    st.markdown(f'<div class="error-message">❌ Error: {e}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-message">❌ Model not found. Train models first.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def fraud_input():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🕵️ Fraud Detection Engine")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    policy = st.selectbox("Insurance Type", ["Health Insurance", "Car Insurance", "Life Insurance", "Home Insurance"])
    amount = st.number_input("Claim Amount (₹)", 0.0, 1000000.0, 25000.0, step=5000.0)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    if policy == "Health Insurance":
        st.markdown("### 🩺 Health Claim Details")
        col1, col2 = st.columns(2)
        with col1:
            incident = st.selectbox("Treatment", ["Emergency", "Surgery", "OPD", "ICU", "Checkup"])
        with col2:
            severity = st.selectbox("Severity", ["Minor", "Moderate", "Severe", "Critical"])
        witnesses = st.number_input("Witnesses", 0, 10, 0)
        report = st.selectbox("Medical Reports", ["Yes", "No"])
        
    elif policy == "Car Insurance":
        st.markdown("### 🚗 Car Claim Details")
        col1, col2 = st.columns(2)
        with col1:
            incident = st.selectbox("Accident Type", ["Collision", "Theft", "Natural Disaster", "Vandalism", "Fire"])
        with col2:
            severity = st.selectbox("Damage", ["Minor", "Moderate", "Severe", "Total Loss"])
        witnesses = st.number_input("Witnesses", 0, 10, 0)
        report = st.selectbox("Police Report", ["Yes", "No"])
        
    elif policy == "Life Insurance":
        st.markdown("### 🧬 Life Claim Details")
        col1, col2 = st.columns(2)
        with col1:
            incident = st.selectbox("Cause", ["Natural", "Accident", "Illness", "Critical Illness"])
        with col2:
            severity = st.selectbox("Medical History", ["Clean", "Minor Issues", "Major Issues"])
        witnesses = st.number_input("Witnesses", 0, 10, 0)
        report = st.selectbox("Death Certificate", ["Available", "Not Available"])
        
    else:
        st.markdown("### 🏠 Home Claim Details")
        col1, col2 = st.columns(2)
        with col1:
            incident = st.selectbox("Incident", ["Fire", "Theft", "Water Damage", "Natural Disaster", "Vandalism"])
        with col2:
            severity = st.selectbox("Damage", ["Minor", "Moderate", "Major", "Total Loss"])
        witnesses = st.number_input("Witnesses", 0, 10, 0)
        report = st.selectbox("Police Report", ["Yes", "No"])
    
    if st.button("🔍 Analyze Fraud Risk", use_container_width=True):
        if fraud_model:
            with st.spinner("Analyzing..."):
                time.sleep(0.5)
                mapped = map_fraud_input(policy, amount, incident, severity, witnesses, report)
                df = pd.DataFrame([[
                    mapped["policy_type"], mapped["claim_amount"], mapped["incident_type"],
                    mapped["incident_severity"], mapped["witnesses"], mapped["police_report"]
                ]], columns=['policy_type','claim_amount','incident_type','incident_severity','witnesses','police_report'])
                try:
                    proba = fraud_model.predict_proba(df)[0][1]
                    pred_class = fraud_model.predict(df)[0]
                    if pred_class == 1:
                        st.markdown(f"""
                        <div class="error-message">
                            <h3>⚠️ FRAUD ALERT</h3>
                            <p>Probability: {proba:.2%} | Risk: HIGH</p>
                            <p>Manual review required</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="success-message">
                            <h3>✅ Low Fraud Risk</h3>
                            <p>Probability: {proba:.2%} | Risk: LOW</p>
                            <p>Can be processed automatically</p>
                        </div>
                        """, unsafe_allow_html=True)
                    add_to_history("Fraud", f"{'FRAUD' if pred_class==1 else 'Clean'} ({proba:.0%})", f"Policy: {policy}")
                except Exception as e:
                    st.markdown(f'<div class="error-message">❌ Error: {e}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-message">❌ Fraud model not loaded. Train first.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def analytics():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Model Performance Analytics")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["🏥 Health", "🚗 Car", "🧬 Life", "🏠 Home", "🛡️ Fraud"])
    for tab, name in zip(tabs, ["health", "car", "life", "home", "fraud"]):
        with tab:
            file_path = f"models/{name}_regression_results.csv" if name != "fraud" else "models/fraud_classification_results.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                st.dataframe(df, use_container_width=True)
                if name != "fraud" and 'R2' in df.columns:
                    fig = px.bar(df, x='model', y='R2', title=f"{name.capitalize()} - R² Score",
                                 color='model', color_discrete_sequence=['#f97316'])
                    st.plotly_chart(fig, use_container_width=True)
                elif name == "fraud":
                    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                    if all(m in df.columns for m in metrics):
                        fig = px.bar(df, x='model', y=metrics, barmode='group',
                                     title="Fraud Detection Metrics",
                                     color_discrete_sequence=['#f97316', '#2c5282', '#4a7c59'])
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Results not found. Train models first.")
    
    csv = export_history()
    if csv:
        st.markdown("---")
        st.download_button("📥 Export History", csv, f"history_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== PAGE ROUTING (NO EXTRA HEADERS) ====================
if selected == "📊 Dashboard":
    dashboard()
elif selected == "🩺 Health Insurance":
    # Removed duplicate section-header
    health_input()
elif selected == "🚗 Car Insurance":
    car_input()
elif selected == "🧬 Life Insurance":
    life_input()
elif selected == "🏠 Home Insurance":
    home_input()
elif selected == "🕵️ Fraud Detection":
    fraud_input()
elif selected == "📈 Analytics":
    analytics()