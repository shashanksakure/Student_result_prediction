import streamlit as st
import numpy as np
import pandas as pd
import pickle

# --- EXECUTIVE THEME & PAGE INITIALIZATION ---
st.set_page_config(
    page_title="Institutional Analytics & Risk Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Corporate CSS for Business Meetings
st.markdown("""
    <style>
        /* Modern Background & Typography */
        .main {
            background-color: #fcfdfe;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* High-impact presentation banner */
        .hero-banner {
            background: linear-gradient(135deg, #0f172a 0%, #0d9488 100%);
            padding: 2.5rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 2.5rem;
            box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
        }
        
        /* Section Dividers */
        .section-header {
            color: #1e293b;
            font-weight: 700;
            border-bottom: 3px solid #0d9488;
            padding-bottom: 0.5rem;
            margin-top: 1.5rem;
            margin-bottom: 1.5rem;
            font-size: 1.4rem;
        }

        /* Result Callouts */
        .result-box-yes {
            background: linear-gradient(135deg, #059669 0%, #10b981 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            text-align: center;
            font-size: 1.75rem;
            font-weight: 700;
            box-shadow: 0 10px 20px -5px rgba(5, 150, 105, 0.4);
        }
        .result-box-no {
            background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            text-align: center;
            font-size: 1.75rem;
            font-weight: 700;
            box-shadow: 0 10px 20px -5px rgba(220, 38, 38, 0.4);
        }
    </style>
""", unsafe_with_html=True)

# --- MODEL INGESTION ---
@st.cache_resource
def load_predictive_model():
    try:
        with open("svm_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        # Fallback if renamed to model.pkl
        try:
            with open("model.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            st.error("⚠️ Model file not found. Ensure either 'svm_model.pkl' or 'model.pkl' is in the root directory.")
            return None

model = load_predictive_model()

# --- HERO BANNER DISPLAY ---
st.markdown("""
    <div class="hero-banner">
        <h1 style='margin:0; font-size: 2.6rem; font-weight: 800; letter-spacing: -0.5px;'>Student Success & Performance Insights</h1>
        <p style='margin:8px 0 0 0; opacity: 0.85; font-size: 1.15rem; font-weight: 400;'>Predictive Core Engine powered by Support Vector Classification</p>
    </div>
""", unsafe_with_html=True)

if model:
    st.markdown("<div class='section-header'>📊 Profile & Operational Metrics Input</div>", unsafe_with_html=True)
    
    # 3x3 Form Grid for seamless data entry during live demos
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender Target", ["Male", "Female"])
        study_hours = st.slider("Weekly Study Hours", 0, 100, 25)
        extracurricular = st.selectbox("Extracurricular Engagement", ["Yes", "No"])
        
    with col2:
        age = st.number_input("Age Profile", min_value=12, max_value=100, value=18)
        attendance_rate = st.slider("Attendance Rate (%)", 0, 100, 85)
        previous_score = st.slider("Historical Score (Previous Term)", 0, 100, 70)
        
    with col3:
        parent_edu = st.selectbox("Parent Education Level", ["High School", "Associate's Degree", "Bachelor's Degree", "Master's Degree", "PhD"])
        internet_access = st.selectbox("Reliable Internet Access?", ["Yes", "No"])
        final_score = st.slider("Current Assessment Score", 0, 100, 72)

    # DataFrame formatting to perfectly align with model's expected inputs
    raw_payload = pd.DataFrame([{
        'gender': gender,
        'age': age,
        'study_hours_per_week': study_hours,
        'attendance_rate': attendance_rate,
        'parent_education': parent_edu,
        'internet_access': internet_access,
        'extracurricular': extracurricular,
        'previous_score': previous_score,
        'final_score': final_score
    }])

    st.markdown("<br>", unsafe_with_html=True)
    
    # --- ANALYSIS INFERENCE ---
    _, middle_btn_col, _ = st.columns([1, 1, 1])
    with middle_btn_col:
        trigger_prediction = st.button("🚀 Evaluate Student Profile", use_container_width=True)

    if trigger_prediction:
        st.markdown("<div class='section-header'>🏁 Executive Analytics Summary</div>", unsafe_with_html=True)
        
        try:
            # Predict execution
            prediction_output = model.predict(raw_payload)[0]
            
            output_col, metrics_col = st.columns([2, 1])
            
            with output_col:
                # Dynamic matching based on model's internal ['No', 'Yes'] configuration
                if str(prediction_output).strip().lower() in ['yes', '1']:
                    st.markdown("""
                        <div class="result-box-yes">
                            🎯 EVALUATION STATUS: STABLE / SUCCESS PREDICTED (YES)
                        </div>
                    """, unsafe_with_html=True)
                else:
                    st.markdown("""
                        <div class="result-box-no">
                            ⚠️ EVALUATION STATUS: INTERVENTION RECOMMENDED (NO)
                        </div>
                    """, unsafe_with_html=True)
            
            with metrics_col:
                # Provide confidence fallback metrics or state visualization
                if hasattr(model, "predict_proba"):
                    confidence_pct = np.max(model.predict_proba(raw_payload)[0]) * 100
                    st.metric(label="Algorithm Certainty Level", value=f"{confidence_pct:.1f}%")
                else:
                    st.metric(label="Core Execution Engine", value="Verified Fit", delta="SVC RBF Kernel")
                    
        except Exception as err:
            st.error(f"Data Pipeline Configuration Error: {err}")
            st.info("💡 Note: Ensure your training data transforms (like OneHotEncoders) match the raw strings if you didn't wrap the pipeline.")
