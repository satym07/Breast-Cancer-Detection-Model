import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Breast Cancer Detection Web App", layout="centered")
st.title("🩺 Breast Cancer Detector")

st.markdown("""
This simple tool helps assess whether a tumor is likely to be **Benign (0)** or **Malignant (1)**  
based on a few easy-to-answer questions.

Early detection can help save lives — this app aims to make that step more accessible.
""")

st.header("📋 Answer a Few Questions")

def map_inputs_to_features(size, texture, smooth, concave, hard):
    return pd.DataFrame([{
        'radius_mean': 12 if size == "Small" else 18,
        'texture_mean': 10 if texture == "Uniform" else 20,
        'perimeter_mean': 75 if size == "Small" else 120,
        'area_mean': 500 if size == "Small" else 1000,
        'smoothness_mean': 0.08 if smooth == "Smooth" else 0.15,
        'compactness_mean': 0.05 if smooth == "Smooth" else 0.2,
        'concavity_mean': 0.02 if concave == "No" else 0.2,
        'concave points_mean': 0.01 if concave == "No" else 0.1,
        'symmetry_mean': 0.15 if texture == "Uniform" else 0.25,
        'fractal_dimension_mean': 0.05 if hard == "Soft" else 0.1,
    }])

with st.form("input_form"):
    size = st.radio("What is the approximate size of the tumor?", ["Small", "Large"])
    texture = st.radio("Is the texture of the tissue uniform?", ["Uniform", "Irregular"])
    smooth = st.radio("Does the tumor surface feel smooth?", ["Smooth", "Rough"])
    concave = st.radio("Does the tumor have deep indentations?", ["No", "Yes"])
    hard = st.radio("Is the tumor soft or hard to the touch?", ["Soft", "Hard"])

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    st.info("Processing your input...")

    features = map_inputs_to_features(size, texture, smooth, concave, hard)
    model = joblib.load("model_lr_simple.pkl")
    scaler = joblib.load("scaler_lr_simple.pkl")

    scaled_input = scaler.transform(features)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][prediction]

    if prediction == 0:
        st.success("🟢 The tumor is likely **Benign (0)**.")
    else:
        st.error("🔴 The tumor is likely **Malignant (1)**.")

    st.markdown(f"**Prediction Confidence:** `{round(probability * 100, 2)}%`")

    st.markdown("---")
    st.caption("⚠️ This is not a medical diagnosis. For professional advice, please consult a certified healthcare provider.")
