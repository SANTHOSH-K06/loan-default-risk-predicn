import streamlit as st
import pandas as pd
import numpy as np
from engine import LoanClassifierEngine
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Loan Risk Dashboard", page_icon="🏦", layout="wide")

# Premium Emerald/Dark Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Poppins', sans-serif !important; }
.stApp { background: linear-gradient(135deg, #00120b 0%, #002618 50%, #000e0a 100%); color: #e8fff4; }
.hero { background: linear-gradient(135deg, rgba(0,255,150,0.15), rgba(0,180,120,0.1)); border: 1px solid rgba(0,255,160,0.2); border-radius: 24px; padding: 40px; text-align: center; margin-bottom: 30px; }
.hero h1 { font-size: 3rem !important; font-weight: 800 !important; background: linear-gradient(90deg, #5dfdcb, #4ade80, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.glass { background: rgba(255,255,255,0.04); border: 1px solid rgba(0,255,160,0.15); border-radius: 20px; padding: 24px; margin-bottom: 20px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); transition: 0.3s; }
.glass:hover { border: 1px solid rgba(0,255,160,0.4); transform: translateY(-3px); }
.sec-title { font-size: 0.9rem; font-weight: 700; color: #4ade80; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 15px; }
.stButton > button { width: 100%; background: linear-gradient(135deg, #059669, #10b981) !important; color: #002015 !important; font-weight: 800 !important; border-radius: 14px !important; padding: 0.8em !important; border: none !important; transition: 0.3s !important; }
.stButton > button:hover { transform: scale(1.02); box-shadow: 0 0 20px rgba(16, 185, 129, 0.4); }
.result-approved { background: rgba(52,211,153,0.12); border: 1px solid rgba(52,211,153,0.35); border-left: 8px solid #4ade80; border-radius: 20px; padding: 30px; }
.result-rejected { background: rgba(248,113,113,0.12); border: 1px solid rgba(248,113,113,0.35); border-left: 8px solid #f87171; border-radius: 20px; padding: 30px; }
.metric-val { font-size: 2.2rem; font-weight: 900; color: #4ade80; text-shadow: 0 0 10px rgba(74, 222, 128, 0.3); }
.conf-bar { background: rgba(255,255,255,0.08); border-radius: 50px; height: 14px; margin-top: 20px; overflow: hidden; }
.conf-fill-g { height: 100%; background: linear-gradient(90deg, #10b981, #5dfdcb); }
.conf-fill-r { height: 100%; background: linear-gradient(90deg, #e11d48, #fb7185); }
</style>
""", unsafe_allow_html=True)

if 'engine' not in st.session_state:
    st.session_state.engine = LoanClassifierEngine()

# Sidebar Optimization
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135706.png", width=80)
    st.title("⚙️ Model Tuning")
    algo_choice = st.radio("Primary Algorithm", ["Logistic Regression", "SVM"])
    C_param = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
    
    st.divider()
    
    # Train model on startup
    df_raw = st.session_state.engine.load_data()
    if df_raw is not None:
        X_scaled, y, _ = st.session_state.engine.preprocess(df_raw)
        model, acc, auc = st.session_state.engine.optimize_and_train(X_scaled, y, 
                                                                   algorithm='LR' if algo_choice == "Logistic Regression" else 'SVM', 
                                                                   C=C_param)
        st.success("Model Trained Successfully!")
    else:
        st.error("Dataset 'train.csv' not found.")
        st.stop()

# Header
st.markdown('<div class="hero"><h1>🏦 Loan Risk Dashboard</h1><p>Intelligent Financial Risk Analytics System</p></div>', unsafe_allow_html=True)

# Performance Metrics
m1, m2, m3 = st.columns(3)
with m1: st.markdown(f'<div class="glass"><div class="sec-title">Accuracy</div><div class="metric-val">{acc*100:.2f}%</div></div>', unsafe_allow_html=True)
with m2: st.markdown(f'<div class="glass"><div class="sec-title">ROC-AUC</div><div class="metric-val">{auc:.4f}</div></div>', unsafe_allow_html=True)
with m3: st.markdown(f'<div class="glass"><div class="sec-title">Active Model</div><div class="metric-val">{"LR" if algo_choice == "Logistic Regression" else "SVM"}</div></div>', unsafe_allow_html=True)

# Main Application Form
st.markdown('<div class="glass"><div class="sec-title">🧪 Applicant Risk Assessment</div>', unsafe_allow_html=True)
ca, cb = st.columns(2)

with ca:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_emp = st.selectbox("Self Employed", ["Yes", "No"])

with cb:
    income = st.number_input("Applicant Monthly Income ($)", min_value=0, value=5000)
    co_income = st.number_input("Co-applicant Monthly Income ($)", min_value=0, value=0)
    loan_amt = st.number_input("Loan Amount (thousands)", min_value=0, value=150)
    term = st.selectbox("Loan Term (Months)", [360, 180, 120, 60, 36])
    credit_hist = st.selectbox("Credit History", ["1.0", "0.0"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.button("🔍 Analyze Default Risk"):
    input_data = {
        'Gender': gender, 'Married': married, 'Dependents': dependents,
        'Education': education, 'Self_Employed': self_emp, 'ApplicantIncome': income,
        'CoapplicantIncome': co_income, 'LoanAmount': loan_amt, 'Loan_Amount_Term': term,
        'Credit_History': float(credit_hist), 'Property_Area': property_area
    }
    
    input_df = pd.DataFrame([input_data])
    X_input = st.session_state.engine.preprocess(input_df, is_train=False)
    
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]
    conf = proba[pred] * 100
    
    res_cls = "result-approved" if pred == 1 else "result-rejected"
    res_lbl = "Approved (Low Risk)" if pred == 1 else "Rejected (High Risk)"
    bar_cls = "conf-fill-g" if pred == 1 else "conf-fill-r"
    
    st.markdown(f"""
    <div class="{res_cls}">
        <div style="font-size:2.2rem;font-weight:900;">{res_lbl}</div>
        <div style="font-size:1.1rem;margin-top:5px;opacity:0.8;">Prediction Confidence: {conf:.1f}%</div>
        <div class="conf-bar"><div class="{bar_cls}" style="width:{conf}%"></div></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
