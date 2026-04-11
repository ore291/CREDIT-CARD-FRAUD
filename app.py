import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd

# -----------------------------
# Load model + label encoder + config
# -----------------------------
model = joblib.load("fraud_model.pkl")
le = joblib.load("label_encoder.pkl")

with open("model_config.json") as f:
    config = json.load(f)

FEATURES = config["features"]
THRESHOLD = config["threshold"]

st.title("Credit Card Fraud Detection App")
st.write(f"Model: {config['model_name']}")
st.write(f"Threshold: {THRESHOLD}")

# -----------------------------
# Input fields
# -----------------------------
st.sidebar.header("Input Transaction Data")

categories = list(le.classes_)

selected_category = st.sidebar.selectbox(
    "Transaction Category",
    categories
)

if selected_category in le.classes_:
    category_encoded = le.transform([selected_category])[0]
else:
    category_encoded = -1  

def user_input():
    data = {
        "amt": st.sidebar.number_input("Transaction Amount", 0.0, 10000.0, 100.0),
        "age": st.sidebar.number_input("Customer Age", 18, 100, 30),
        "hour": st.sidebar.slider("Hour", 0, 23, 12),
        "day_of_week": st.sidebar.slider("Day of Week", 0, 6, 2),
        "month": st.sidebar.slider("Month", 1, 12, 6),
        "distance_km": st.sidebar.number_input("Distance (km)", 0.0, 1000.0, 5.0),
        "category_enc": category_encoded,
        "city_pop": st.sidebar.number_input("City Population", 0, 10000000, 500000),
        "lat": st.sidebar.number_input("Latitude", -90.0, 90.0, 40.0),
        "long": st.sidebar.number_input("Longitude", -180.0, 180.0, -73.0),
        "merch_lat": st.sidebar.number_input("Merchant Latitude", -90.0, 90.0, 40.5),
        "merch_long": st.sidebar.number_input("Merchant Longitude", -180.0, 180.0, -73.5),
    }
    return pd.DataFrame([data])

input_df = user_input()

st.subheader("Input Data")
st.write(input_df)

# -----------------------------
# Prediction
# -----------------------------
proba = model.predict_proba(input_df[FEATURES])[0][1]
prediction = int(proba >= THRESHOLD)

st.subheader("Prediction")

if prediction == 1:
    st.error(f"⚠️ Fraud Detected! (Risk: {proba:.2f})")
else:
    st.success(f"✅ Legit Transaction (Risk: {proba:.2f})")

