import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="ECG Classifier", page_icon="❤️", layout="wide")

st.title("❤️ Machine Learning ECG Cardiac Abnormality Detector")

# Get absolute path to the model relative to app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "src", "ecg_random_forest.pkl")

@st.cache_resource
def load_ecg_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at path: {MODEL_PATH}")
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_ecg_model()

if model is None:
    st.warning("App stopped because the model failed to load. Check the path above.")
    st.stop()

# --- Rest of your UI / prediction logic below ---