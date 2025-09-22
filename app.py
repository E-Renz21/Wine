import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load("wine_model (1).pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("Wine Quality Predictor")

st.write("Enter the wine characteristics to predict its quality.")

fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, step=0.1)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, step=0.01)
citric_acid = st.number_input("Citric Acid", min_value=0.0, step=0.01)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, step=0.1)
chlorides = st.number_input("Chlorides", min_value=0.0, step=0.001)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, step=1.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, step=1.0)
density = st.number_input("Density", min_value=0.0, step=0.0001)
pH = st.number_input("pH", min_value=0.0, step=0.01)
sulphates = st.number_input("Sulphates", min_value=0.0, step=0.01)
alcohol = st.number_input("Alcohol", min_value=0.0, step=0.1)

input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                        residual_sugar, chlorides, free_sulfur_dioxide,
                        total_sulfur_dioxide, density, pH,
                        sulphates, alcohol]])

input_scaled = scaler.transform(input_data)

if st.button("Predict Quality"):
    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled)[0][prediction]

    if prediction == 1:
        st.success(f"✅ This wine is predicted to be **Good Quality** 🍷 (Confidence: {confidence:.2f})")
    else:
        st.error(f"❌ This wine is predicted to be **Not Good Quality** (Confidence: {confidence:.2f})")
