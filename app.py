import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI-Assisted CVD Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ AI-Assisted Cardiovascular Disease Risk Prediction")
st.write(
    "Predict cardiovascular disease risk and receive "
    "personalized, rule-based health suggestions."
)

# ==========================================
# 2. LOAD TRAINED MODEL (joblib, not pickle)
# ==========================================

@st.cache_resource
def load_model():
    model = joblib.load("cvd_xgb_model.pkl")
    feature_order = joblib.load("cvd_feature_order.pkl")
    return model, feature_order


try:
    model, feature_order = load_model()
except Exception as e:
    st.error(f"Unable to load the trained model: {e}")
    st.stop()
