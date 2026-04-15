import streamlit as st
import joblib
import json
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

# Custom CSS for better mobile responsiveness
st.markdown("""
<style> 
    /* Make form elements more touch-friendly on mobile */
    @media (max-width: 768px) {
        .stNumberInput, .stSelectbox, .stSlider {
            margin-bottom: 0.5rem;
        }
        button {
            font-size: 1rem !important;
            padding: 0.5rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

st.title("Credit Card Fraud Detection App")
st.write(f"Model: {config['model_name']}")
st.write(f"Threshold: {THRESHOLD}")

# -----------------------------
# SESSION STATE 
# -----------------------------
if "inputs" not in st.session_state:
    st.session_state.inputs = {
        "amt": 100.0,
        "age": 30,
        "hour": 12,
        "day_of_week": 2,
        "month": 6,
        "distance_km": 5.0,
        "category_enc": 0,
        "city_pop": 500000,
        "lat": 40.0,
        "long": -73.0,
        "merch_lat": 40.5,
        "merch_long": -73.5,
    }

if "submitted" not in st.session_state:
    st.session_state.submitted = False

# -----------------------------
# TWO-COLUMN LAYOUT FOR DESKTOP (auto-stacks on mobile)
# -----------------------------
col_form, col_result = st.columns([2, 1], gap="medium")

# ==================== LEFT COLUMN: INPUT FORM ====================
with col_form:
    st.header("Enter Transaction Details")
    
    categories = list(le.classes_)
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        # -------- LEFT SUB-COLUMN --------
        with col1:
            selected_category = st.selectbox("Transaction Category", categories)
            amt = st.number_input("Transaction Amount", 0.0, 1000.0, st.session_state.inputs["amt"])
            age = st.number_input("Customer Age", 18, 100, st.session_state.inputs["age"])
            
            # IMPROVED: Better hour selection with AM/PM display
            hour_options = [f"{h:02d}:00 {'AM' if h < 12 else 'PM'}" for h in range(24)]
            hour_display = hour_options[st.session_state.inputs["hour"]]
            hour_selection = st.selectbox(
                "Hour",
                options=hour_options,
                index=st.session_state.inputs["hour"],
                help="Select the hour of the transaction"
            )
            hour = hour_options.index(hour_selection)
            
            # IMPROVED: Better day of week selection with names
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_selection = st.selectbox(
                "Day of Week",
                options=day_names,
                index=st.session_state.inputs["day_of_week"],
                help="Select the day of the transaction"
            )
            day_of_week = day_names.index(day_selection)
            
            # IMPROVED: Better month selection with names
            month_names = ["January", "February", "March", "April", "May", "June", 
                          "July", "August", "September", "October", "November", "December"]
            month_selection = st.selectbox(
                "Month",
                options=month_names,
                index=st.session_state.inputs["month"] - 1,
                help="Select the month of the transaction"
            )
            month = month_names.index(month_selection) + 1
        
        # -------- RIGHT SUB-COLUMN --------
        with col2:
            distance_km = st.number_input("Distance (km)", 0.0, 1000.0, st.session_state.inputs["distance_km"])
            city_pop = st.number_input("City Population", 0, 10000000, st.session_state.inputs["city_pop"])
            lat = st.number_input("Latitude", -90.0, 90.0, st.session_state.inputs["lat"])
            long = st.number_input("Longitude", -180.0, 180.0, st.session_state.inputs["long"])
            merch_lat = st.number_input("Merchant Latitude", -90.0, 90.0, st.session_state.inputs["merch_lat"])
            merch_long = st.number_input("Merchant Longitude", -180.0, 180.0, st.session_state.inputs["merch_long"])
        
        # Update session state before submission
        if selected_category in le.classes_:
            category_encoded = le.transform([selected_category])[0]
        else:
            category_encoded = -1
        
        st.session_state.inputs.update({
            "amt": amt,
            "age": age,
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            "distance_km": distance_km,
            "category_enc": category_encoded,
            "city_pop": city_pop,
            "lat": lat,
            "long": long,
            "merch_lat": merch_lat,
            "merch_long": merch_long,
        })
        
        submitted = st.form_submit_button("Predict Fraud Risk", use_container_width=True)
        
        if submitted:
            st.session_state.submitted = True

# ==================== RIGHT COLUMN: RESULTS ====================
with col_result:
    st.header("Risk Analysis")
    if st.session_state.submitted or submitted:
        # Make prediction
        input_df = pd.DataFrame([st.session_state.inputs])
        proba = model.predict_proba(input_df[FEATURES])[0][1]
        prediction = int(proba >= THRESHOLD)
        
        # Display metrics
        st.metric(
            label="Fraud Risk Score",
            value=f"{proba*100:.1f}%",
            delta="High Risk" if proba >= 0.5 else ("Medium Risk" if proba >= 0.3 else "Low Risk"),
            delta_color="inverse" if proba >= 0.3 else "normal"
        )

        # Progress bar
        st.progress(int(proba * 100))

        # Color-coded result
        if proba >= 0.5:
            st.error("High Risk of Fraud")
        elif proba >= 0.3:
            st.warning("Medium Risk - Review Transaction")
        else:
            st.success("Low Risk - Transaction appears legitimate")
    else:
        st.metric(label="Fraud Risk Score", value="0.0%", delta="Waiting for input")
        st.progress(0)
        st.info("Fill in the transaction details and click predict")
