import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time

# ========================
# Page configuration
# ========================
st.set_page_config(
    page_title="Heart Risk Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========================
# Custom CSS for styling
# ========================
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global styles */
.main { font-family: 'Inter', sans-serif; }

/* Header animation */
.header-container { text-align:center; padding:2rem 0; background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); border-radius:15px; margin-bottom:2rem; color:white;}
.header-title { font-size:2.5rem; font-weight:700; margin-bottom:0.5rem; animation:pulse 2s infinite;}
.header-subtitle { font-size:1.1rem; opacity:0.9; font-weight:300; }

/* Card styling */
.info-card { background:linear-gradient(135deg,#f5f7fa 0%,#c3cfe2 100%); padding:1.5rem; border-radius:12px; border-left:4px solid #667eea; margin:1rem 0; box-shadow:0 4px 15px rgba(0,0,0,0.1); }
.input-section { background:white; padding:2rem; border-radius:15px; box-shadow:0 8px 25px rgba(0,0,0,0.1); margin:2rem 0; border:1px solid #e1e5e9; }
.section-header { color:#2c3e50; font-size:1.5rem; font-weight:600; margin-bottom:1.5rem; padding-bottom:0.5rem; border-bottom:2px solid #667eea; }
.results-card { background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:white; padding:2rem; border-radius:15px; margin:2rem 0; box-shadow:0 10px 30px rgba(102,126,234,0.3); }
.results-title { font-size:1.8rem; font-weight:600; margin-bottom:1rem; text-align:center; }

/* Progress bars */
.progress-container { background: rgba(255,255,255,0.2); border-radius:25px; padding:0.3rem; margin:0.8rem 0; overflow:hidden; }
.progress-bar { height:25px; border-radius:20px; transition: width 2s ease-in-out; display:flex; align-items:center; justify-content:center; font-weight:600; font-size:0.9rem; color:#2c3e50; }
.risk-low { background:linear-gradient(90deg,#56ab2f,#a8e6cf);}
.risk-medium { background:linear-gradient(90deg,#f7971e,#ffd200);}
.risk-high { background:linear-gradient(90deg,#c94b4b,#ff6b6b); }

/* Animations */
@keyframes pulse { 0%,100% {transform:scale(1);} 50% {transform:scale(1.05);} }

</style>
""", unsafe_allow_html=True)

# ========================
# Load trained model
# ========================
@st.cache_resource
def load_model():
    return joblib.load("../models/final_model.pkl")

try:
    model = load_model()
    model_loaded = True
except:
    st.error("⚠️ Model file not found. Please ensure 'final_model.pkl' is in the models folder.")
    model_loaded = False

# ========================
# Header
# ========================
st.markdown("""
<div class="header-container">
    <div class="header-title">❤️ Heart Risk Prediction</div>
    <div class="header-subtitle">Professional Cardiovascular Risk Assessment Tool</div>
</div>
""", unsafe_allow_html=True)

# ========================
# Info card
# ========================
st.markdown("""
<div class="info-card">
    <h4>🏥 Clinical Decision Support</h4>
    <p>This tool uses machine learning to estimate the likelihood of heart disease. Enter accurate patient information for best results.</p>
</div>
""", unsafe_allow_html=True)

# ========================
# Input form
# ========================
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown('<h2 class="section-header">📋 Patient Information</h2>', unsafe_allow_html=True)

# Columns
col1, col2 = st.columns(2)
input_data = {}

with col1:
    st.markdown("### 👤 Demographics & Vitals")
    input_data["age"] = st.number_input("Age (years)", 1, 120, 40)
    input_data["trestbps"] = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    input_data["chol"] = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    input_data["thalach"] = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    input_data["oldpeak"] = st.number_input("ST Depression (exercise vs rest)", 0.0, 6.0, 0.0, step=0.1)

with col2:
    st.markdown("### 🔬 Clinical Parameters")
    cp_map = {"Typical Angina":1, "Atypical Angina":2, "Non-anginal Pain":3, "Asymptomatic":4}
    thal_map = {"Normal":3, "Fixed Defect":6, "Reversible Defect":7}

    input_data["cp"] = st.selectbox("Chest Pain Type", list(cp_map.values()))
    input_data["thal"] = st.selectbox("Thalassemia", list(thal_map.values()))
    input_data["ca"] = st.selectbox("Number of Major Vessels (0–3)", [0,1,2,3])
    input_data["exang"] = st.selectbox("Exercise Induced Angina", [0,1])

st.markdown('</div>', unsafe_allow_html=True)

# ========================
# Prediction
# ========================
col1, col2, col3 = st.columns([1,2,1])
with col2:
    if st.button("🔍 Analyze Patient Risk") and model_loaded:
        with st.spinner("Analyzing patient data..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i+1)
            
            # Prepare input for prediction
            df_input = pd.DataFrame([input_data])
            print(df_input)
            df_input = df_input[["thalach", "oldpeak", "age", "chol", "trestbps", "ca", "thal", "cp", "exang"]]
            print(df_input)
            
            probs = model.predict_proba(df_input)[0]
            pred_class = np.argmax(probs)
            confidence = probs[pred_class]*100
            
            progress_bar.empty()
        
        # Results card
        st.markdown('<div class="results-card">', unsafe_allow_html=True)
        st.markdown('<div class="results-title">🩺 Clinical Assessment Results</div>', unsafe_allow_html=True)

        # Main prediction
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='metric-card'><h3>Primary Classification</h3><h2>Class {pred_class}</h2></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><h3>Confidence Level</h3><h2>{confidence:.1f}%</h2></div>", unsafe_allow_html=True)
        
        # Probability breakdown
        st.markdown("### 📊 Detailed Risk Breakdown")
        for i, p in enumerate(probs):
            perc = p*100
            risk_class = "risk-low" if perc<30 else "risk-medium" if perc<70 else "risk-high"
            st.markdown(f"""
            <div style="margin:1rem 0;">
                <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem;">
                    <span>Class {i}</span><span>{perc:.2f}%</span>
                </div>
                <div class="progress-container">
                    <div class="progress-bar {risk_class}" style="width:{perc}%;">{perc:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Notes
        st.markdown("""
        <div class="info-card">
            <h4>📝 Clinical Notes</h4>
            <p><strong>Disclaimer:</strong> Results are for clinical support only and should not replace professional judgment.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center; padding:2rem; color:#666; border-top:1px solid #e1e5e9; margin-top:3rem;">
    <p>💡 <strong>Healthcare AI Assistant</strong> | For medical use only.</p>
</div>
""", unsafe_allow_html=True)
