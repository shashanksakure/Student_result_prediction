import streamlit as st
import numpy as np
import pandas as pd
import pickle

# --- PRESENTATION CONFIGURATION ---
st.set_page_config(
    page_title="Predictive Insights Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- PREMIUM EXECUTIVE STYLING (CSS) ---
st.markdown("""
    <style>
        /* Base page styling */
        .main {
            background-color: #f8fafc;
            font-family: 'Inter', system-ui, sans-serif;
        }
        
        /* High-contrast dashboard header */
        .presentation-title {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
            padding: 2.5rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 2.5rem;
            box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
            border-left: 8px solid #0ea5e9;
        }
        
        /* Section Dividers */
        .metric-heading {
            color: #1e293b;
            font-weight: 700;
            font-size: 1.4rem;
            border-bottom: 2px solid #cbd5e1;
            padding-bottom: 0.5rem;
            margin-top: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        /* Status Card Layouts */
        .status-card-yes {
            background: linear-gradient(135deg, #047857 0%, #10b981 100%);
            color: white;
            padding: 2.2rem;
            border-radius: 12px;
            text-align: center;
            font-size: 1.8rem;
            font-weight: 800;
            letter-spacing: 0.5px;
            box-shadow: 0 10px 20px -5px rgba(16, 185, 129, 0.4);
        }
        
        .status-card-no {
            background: linear-gradient(135deg, #be123c 0%, #f43f5e 100%);
            color: white;
            padding: 2.2rem;
            border-radius: 12px;
            text-align: center;
            font-size: 1.8rem;
            font-weight: 800;
            letter-spacing: 0.5px;
            box-shadow: 0 10px 20px -5px rgba(244, 63, 94, 0.4);
        }
    </style>
""", unsafe_with_html=True)

# --- BINARY DATA INGESTION ---
@st.cache_resource
def load_serialized_model():
    try:
        with open("svm_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        try:
            with open("model.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            st.error("❌ Critical: Serialization file ('svm_model.pkl' / 'model.pkl') not detected in the root path.")
            return None

model = load_serialized_model()

# --- HEADER PRESENTATION BANNER ---
st.markdown("""
    <div class="presentation-title">
        <h1 style='margin:0; font-size: 2.6rem; font-weight: 800;'>Executive Performance Risk Suite</h1>
        <p style='margin:6px 0 0 0; opacity: 0.85; font-size: 1.15rem; font-weight: 400;'>Core Engine: Support Vector Machine (RBF Kernel Variant)</p>
    </div>
""", unsafe_with_html=True)

if model:
    st.markdown("<div class='metric-heading'>📋 Target Feature Parameter Array</div>", unsafe_with_html=True)
    
    # Clean 3x3 Grid Layout mapping precisely to model signatures
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender Demographic", ["Male", "Female"])
        study_hours = st.slider("Weekly Study Hours Allocation", 0, 168, 20)
        extracurricular = st.selectbox("Extracurricular Engagement", ["Yes", "No"])
        
    with col2:
        age = st.number_input("Age Cohort", min_value=5, max_value=100, value=18)
        attendance_rate = st.slider("Institutional Attendance Rate (%)", 0, 100, 92)
        previous_score = st.slider("Historical Baseline Score", 0, 100, 75)
        
    with col3:
        parent_edu = st.selectbox("Parent Educational Tier", ["High School", "Associate's Degree", "Bachelor's Degree", "Master's Degree", "PhD"])
        internet_access = st.selectbox("Broadband Network Connectivity", ["Yes", "No"])
        final_score = st.slider("Current Milestone Evaluation Score", 0, 100, 78)

    # Building structural dataframe mirroring model features
    features_payload = pd.DataFrame([{
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
    
    # Centered interactive activation trigger
    _, btn_center, _ = st.columns([1, 1, 1])
    with btn_center:
        execute_inference = st.button("🔮 Compute Predictive Inference", use_container_width=True)

    if execute_inference:
        st.markdown("<div class='metric-heading'>📊 Analytical Classification Summary</div>", unsafe_with_html=True)
        
        try:
            # Execute model classification array 
            raw_prediction = model.predict(features_payload)[0]
            prediction_str = str(raw_prediction).strip().lower()
            
            ui_col, stats_col = st.columns([2, 1])
            
            with ui_col:
                if prediction_str in ['yes', '1']:
                    st.markdown("""
                        <div class="status-card-yes">
                            🎯 ANALYTICAL OUTCOME: STABLE CRITERIA ACHIEVED (YES)
                        </div>
                    """, unsafe_with_html=True)
                else:
                    st.markdown("""
                        <div class="status-card-no">
                            ⚠️ ANALYTICAL OUTCOME: ESCALATION / RISK TRIGGERED (NO)
                        </div>
                    """, unsafe_with_html=True)
            
            with stats_col:
                # Fallback parameters for models missing native multi-class probabilities
                if hasattr(model, "predict_proba"):
                    confidence_matrix = model.predict_proba(features_payload)[0]
                    top_confidence = np.max(confidence_matrix) * 100
                    st.metric(label="Statistical Core Certainty", value=f"{top_confidence:.2f}%")
                else:
                    st.metric(label="Algorithmic Decision Matrix", value="Verified Fit", delta="SVC Engine OVR")
                    
        except Exception as error_context:
            st.error(f"Execution Error Encountered: {str(error_context)}")
            st.info("💡 Presentation Note: Ensure input strings exactly track the training text labels if label encoding wasn't natively baked into your pickling process.")
