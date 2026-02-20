import streamlit as st
import pandas as pd
import numpy as np
from engine import LoanClassifierEngine
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_lottie import st_lottie
import requests
import time

st.set_page_config(page_title="LOAN RISK CLASSIFIER", page_icon="🧪", layout="wide")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Assets
lottie_success = load_lottieurl("https://lottie.host/855490a6-8e54-47f2-8c1b-741989f6356c/J9u29f8f8f.json")
lottie_fail = load_lottieurl("https://lottie.host/79060b26-3d77-448f-9721-50e588f28fa7/7U7Y4W4m8m.json")
lottie_coin = load_lottieurl("https://lottie.host/4694da95-5858-4680-99e7-4f404e38c5b0/e0vH4Z0W4m.json") # Coin Toss Animation

# Massive Cyberpunk Redesign Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --primary: #00F0FF;
    --secondary: #FF00E5;
    --bg-dark: #000B16;
    --card-bg: rgba(255, 255, 255, 0.03);
    --border: rgba(0, 240, 255, 0.2);
}

* { font-family: 'Inter', sans-serif !important; }

.stApp { 
    background-color: var(--bg-dark);
    background-image: radial-gradient(circle at 50% -20%, #002B4D 0%, transparent 50%),
                      radial-gradient(circle at 0% 100%, #1A002E 0%, transparent 40%);
    color: #E0F2FE; 
}

/* Glassmorphism Containers */
.glass-card { 
    background: var(--card-bg);
    backdrop-filter: blur(15px);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 30px;
    margin-bottom: 25px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5), inset 0 0 15px rgba(0, 240, 255, 0.05);
    transition: 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
.glass-card:hover { 
    border-color: rgba(0, 240, 255, 0.5);
    box-shadow: 0 15px 50px rgba(0, 240, 255, 0.1);
    transform: translateY(-5px);
}

/* Header Styling */
.hero-title { 
    font-size: 3.5rem !important; 
    font-weight: 900 !important; 
    text-align: center;
    background: linear-gradient(90deg, #00F0FF, #7000FF, #FF00E5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -2px;
    margin-bottom: 5px;
    filter: drop-shadow(0 0 15px rgba(0, 240, 255, 0.3));
}
.hero-subtitle {
    text-align: center;
    color: var(--primary);
    font-size: 0.9rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 5px;
    margin-bottom: 40px;
    opacity: 0.8;
}

/* Section Labels */
.sec-label { 
    font-size: 0.75rem; 
    font-weight: 800; 
    color: var(--secondary); 
    text-transform: uppercase; 
    letter-spacing: 3px; 
    margin-bottom: 20px;
    display: flex;
    align-items: center;
}
.sec-label::before {
    content: '';
    display: inline-block;
    width: 30px;
    height: 1px;
    background: var(--secondary);
    margin-right: 10px;
}

/* Metrics */
.metric-container { text-align: center; }
.metric-title { font-size: 0.7rem; font-weight: 700; color: rgba(224, 242, 254, 0.6); letter-spacing: 2px; }
.metric-value { font-size: 2.8rem; font-weight: 900; color: var(--primary); text-shadow: 0 0 20px rgba(0, 240, 255, 0.4); }

/* Custom Button */
.stButton > button { 
    width: 100%; 
    background: linear-gradient(90deg, #00C2FF, #00F0FF) !important; 
    color: #000B16 !important; 
    font-weight: 800 !important; 
    border-radius: 16px !important; 
    padding: 1.2em !important; 
    border: none !important; 
    box-shadow: 0 0 25px rgba(0, 240, 255, 0.2) !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    transition: 0.3s !important;
}
.stButton > button:hover { 
    transform: scale(1.02) !important; 
    box-shadow: 0 0 35px rgba(0, 240, 255, 0.5) !important;
}

/* Results */
.result-box {
    border-radius: 20px;
    padding: 35px;
    position: relative;
    overflow: hidden;
    margin-top: 20px;
    text-align: center;
}
.result-approved { 
    background: rgba(0, 240, 255, 0.05); 
    border: 1px solid rgba(0, 240, 255, 0.3);
    box-shadow: 0 0 40px rgba(0, 240, 255, 0.1);
}
.result-rejected { 
    background: rgba(255, 0, 229, 0.05); 
    border: 1px solid rgba(255, 0, 229, 0.3);
    box-shadow: 0 0 40px rgba(255, 0, 229, 0.1);
}
.conf-bar-outer { background: rgba(255,255,255,0.05); border-radius: 100px; height: 16px; margin-top: 25px; }
.conf-bar-inner-g { background: linear-gradient(90deg, #00F0FF, #7000FF); height: 100%; border-radius: 100px; box-shadow: 0 0 15px rgba(0, 240, 255, 0.5); }
.conf-bar-inner-r { background: linear-gradient(90deg, #FF00E5, #7000FF); height: 100%; border-radius: 100px; box-shadow: 0 0 15px rgba(255, 0, 229, 0.5); }

/* Sidebar Adjustments */
[data-testid="stSidebar"] { background-color: rgba(0, 11, 22, 0.95); border-right: 1px solid var(--border); }

/* Currency Rain Animation */
@keyframes fall {
    0% { transform: translateY(-10vh) rotate(0deg); opacity: 0; }
    10% { opacity: 1; }
    90% { opacity: 1; }
    100% { transform: translateY(110vh) rotate(720deg); opacity: 0; }
}
.coin-rain {
    position: fixed;
    top: -10vh;
    z-index: 9999;
    pointer-events: none;
    animation: fall linear forwards;
}
</style>
""", unsafe_allow_html=True)

if 'engine' not in st.session_state:
    st.session_state.engine = LoanClassifierEngine()

def currency_rain():
    import random
    emojis = ["💰", "💵", "🪙", "💎", "💸", "💹"]
    rain_html = ""
    for i in range(50):
        emoji = random.choice(emojis)
        left = random.randint(0, 95)
        delay = random.uniform(0, 3)
        duration = random.uniform(3, 6)
        size = random.randint(20, 50)
        rain_html += f"<div class='coin-rain' style='left:{left}%; animation-delay:{delay}s; animation-duration:{duration}s; font-size:{size}px;'>{emoji}</div>"
    st.markdown(rain_html, unsafe_allow_html=True)

# Sidebar: Config & Info
with st.sidebar:
    st.markdown("<h2 style='color:#00F0FF; margin-top:0;'>🧪 CORE CONFIG</h2>", unsafe_allow_html=True)
    algo_choice = st.radio("CORE ALGORITHM", ["LOGISTIC REGRESSION", "SVM"])
    C_param = st.slider("REGULARIZATION (C)", 0.01, 10.0, 1.0)
    
    st.divider()
    
    df_raw = st.session_state.engine.load_data()
    if df_raw is not None:
        X_scaled, y, _ = st.session_state.engine.preprocess(df_raw)
        model, acc, auc = st.session_state.engine.optimize_and_train(X_scaled, y, 
                                                                   algorithm='LR' if algo_choice == "LOGISTIC REGRESSION" else 'SVM', 
                                                                   C=C_param)
    else:
        st.error("CORE DATA MISSING")
        st.stop()

# Header Section
st.markdown("<h1 class='hero-title'>LOAN RISK CLASSIFIER</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>Advanced Financial Security Dashboard v2.0</p>", unsafe_allow_html=True)

# Top Metrics Grid
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f"""
    <div class='glass-card metric-container'>
        <div class='metric-title'>ANALYSIS ACCURACY</div>
        <div class='metric-value'>{acc*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class='glass-card metric-container'>
        <div class='metric-title'>ROC-AUC PRECISION</div>
        <div class='metric-value'>{auc:.3f}</div>
    </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class='glass-card metric-container'>
        <div class='metric-title'>OPTIMIZATION LAYER</div>
        <div class='metric-value'>{'OPT' if algo_choice == 'LOGISTIC REGRESSION' else 'HYB'}</div>
    </div>
    """, unsafe_allow_html=True)

# Assessment Section
st.markdown("<div class='glass-card'><div class='sec-label'>APPLICANT LOAN DATA</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("GENDER", ["Male", "Female"])
    married = st.selectbox("MARITAL STATUS", ["Yes", "No"])
    dependents = st.selectbox("DEPENDENTS", ["0", "1", "2", "3+"])
    education = st.selectbox("EDUCATION LEVEL", ["Graduate", "Not Graduate"])
    self_emp = st.selectbox("SELF EMPLOYED", ["Yes", "No"])

with col2:
    income = st.number_input("APPLICANT INCOME ($)", min_value=0, value=5000)
    co_income = st.number_input("CO-APPLICANT INCOME ($)", min_value=0, value=0)
    loan_amt = st.number_input("LOAN AMOUNT (K)", min_value=0, value=150)
    term = st.selectbox("REPAYMENT CYCLE (M)", [360, 180, 120, 60])
    credit_hist = st.selectbox("CREDIT HISTORY", ["1.0", "0.0"])

if st.button("🚀 INITIATE RISK ANALYSIS"):
    input_data = {
        'Gender': gender, 'Married': married, 'Dependents': dependents,
        'Education': education, 'Self_Employed': self_emp, 'ApplicantIncome': income,
        'CoapplicantIncome': co_income, 'LoanAmount': loan_amt, 'Loan_Amount_Term': term,
        'Credit_History': float(credit_hist), 'Property_Area': 'Semiurban'
    }
    
    # NEW: Coin Throwing Effect / Processing state
    processing_placeholder = st.empty()
    with processing_placeholder.container():
        st.markdown("<div class='glass-card' style='text-align:center;'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-label'>LOAN RISK ANALYSIS</div>", unsafe_allow_html=True)
        if lottie_coin:
            st_lottie(lottie_coin, height=300, key="processing_coin")
        else:
            st.write("🪙 Analyzing Risk Factors...")
        st.markdown("</div>", unsafe_allow_html=True)
        time.sleep(2.5) # The "Throwing" effect duration
    
    processing_placeholder.empty() # Clear the animation
    
    input_df = pd.DataFrame([input_data])
    X_input = st.session_state.engine.preprocess(input_df, is_train=False)
    
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]
    conf = proba[pred] * 100
    
    res_cls = "result-approved" if pred == 1 else "result-rejected"
    res_lbl = "RISK CLEAR // APPROVED" if pred == 1 else "CRITICAL RISK // REJECTED"
    bar_cls = "conf-bar-inner-g" if pred == 1 else "conf-bar-inner-r"
    accent_color = "#00F0FF" if pred == 1 else "#FF00E5"
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        if pred == 1:
            currency_rain() # Success celebration with money
            if lottie_success:
                st_lottie(lottie_success, height=200, key="success")
            else:
                st.success("✅ APPROVED")
        else:
            if lottie_fail:
                st_lottie(lottie_fail, height=200, key="fail")
            else:
                st.error("❌ REJECTED")
            
    with c2:
        st.markdown(f"""
        <div class='result-box {res_cls}'>
            <div style='color:{accent_color}; font-size:2rem; font-weight:900;'>{res_lbl}</div>
            <div style='font-size:1.1rem; opacity:0.8; margin-top:5px;'>LOAN CONFIDENCE MODULE: {conf:.1f}%</div>
            <div class='conf-bar-outer'><div class='{bar_cls}' style='width:{conf}%'></div></div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
