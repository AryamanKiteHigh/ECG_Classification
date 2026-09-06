import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="ECG Classifier", page_icon="❤️", layout="wide")

st.title("❤️ Machine Learning ECG Cardiac Abnormality Detector")
st.write("Upload an ECG record from the MIT-BIH dataset (`mitbih_test.csv`) to classify heartbeats in real-time.")

# 2. Dynamic Model Path Resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "src", "ecg_random_forest.pkl")

# Label mappings according to MIT-BIH Arrhythmia Database standards
CLASS_NAMES = {
    0: "Normal (N)",
    1: "Supraventricular Premature (S)",
    2: "Premature Ventricular Contraction (V)",
    3: "Fusion of Ventricular & Normal (F)",
    4: "Unclassifiable Beat (Q)"
}

@st.cache_resource
def load_ecg_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file missing at: {MODEL_PATH}")
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Exception loading model: {e}")
        return None

model = load_ecg_model()

if model is None:
    st.stop()

# 3. File Upload Interface
uploaded_file = st.file_uploader("Choose a CSV file (e.g., mitbih_test.csv)", type=["csv"])

if uploaded_file is not None:
    try:
        # Load CSV without header as MIT-BIH dataset contains raw signal values
        df = pd.read_csv(uploaded_file, header=None)
        
        # Features are first 187 signal values
        X = df.iloc[:, :187]
        
        st.success(f"Successfully loaded {len(df):,} ECG signal records!")
        
        # Record selector slider
        sample_index = st.slider("Select Heartbeat Record Index to Analyze:", 0, len(df) - 1, 0)
        
        # Get selected sample
        sample_signal = X.iloc[sample_index].values
        
        # Perform prediction
        prediction = model.predict([sample_signal])[0]
        prediction_prob = model.predict_proba([sample_signal])[0]
        predicted_class_name = CLASS_NAMES.get(int(prediction), "Unknown")

        col1, col2 = st.columns([2, 1])

        # Plot ECG Waveform
        with col1:
            st.subheader(f"ECG Signal Waveform (Record #{sample_index})")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(sample_signal, color='#e63946', linewidth=2, label="ECG Amplitude")
            ax.set_title(f"Heartbeat #{sample_index} Profile", fontsize=12)
            ax.set_xlabel("Time Step (Sample Points)", fontsize=10)
            ax.set_ylabel("Normalized Amplitude", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper right")
            st.pyplot(fig)

        # Display Diagnostic Predictions
        with col2:
            st.subheader("Model Diagnostic Result")
            if prediction == 0:
                st.success(f"**Classification:** {predicted_class_name}")
            else:
                st.error(f"**Classification:** {predicted_class_name}")

            st.markdown("---")
            st.write("**Prediction Probabilities across classes:**")
            
            prob_df = pd.DataFrame({
                "Heartbeat Category": [CLASS_NAMES[i] for i in range(5)],
                "Probability": [f"{p*100:.2f}%" for p in prediction_prob]
            })
            st.table(prob_df)

    except Exception as err:
        st.error(f"Error processing CSV file: {err}")
else:
    st.info("👆 Please upload `mitbih_test.csv` above to begin prediction testing.")