from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Airfare Predictor", page_icon="✈️", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "airfare_random_forest_pipeline.pkl"
OPTIONS_PATH = BASE_DIR / "airfare_input_options.pkl"

FEATURES = [
    "airline",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class",
    "duration",
    "days_left",
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_options():
    return joblib.load(OPTIONS_PATH)


st.title("✈️ Airfare Predictor")
st.write("Predict flight ticket prices using Random Forest Regressor.")

if not MODEL_PATH.exists() or not OPTIONS_PATH.exists():
    st.error("Model files are missing. Copy both .pkl files from your Jupyter project into this folder.")
    st.stop()

try:
    model = load_model()
    options = load_options()
except Exception as exc:
    st.error(f"Could not load model files: {exc}")
    st.stop()

with st.form("flight_form"):
    airline = st.selectbox("Airline", options["airline"])

    col1, col2 = st.columns(2)
    with col1:
        source_city = st.selectbox("Source City", options["source_city"])
    with col2:
        destination_city = st.selectbox("Destination City", options["destination_city"])

    col3, col4 = st.columns(2)
    with col3:
        departure_time = st.selectbox("Departure Time", options["departure_time"])
        stops = st.selectbox("Stops", options["stops"])
    with col4:
        arrival_time = st.selectbox("Arrival Time", options["arrival_time"])
        travel_class = st.selectbox("Class", options["class"])

    duration = st.number_input("Flight Duration (hours)", min_value=0.1, max_value=60.0, value=2.0, step=0.1)
    days_left = st.number_input("Days Until Departure", min_value=0, max_value=365, value=15, step=1)
    submitted = st.form_submit_button("Predict Ticket Price", use_container_width=True)

if submitted:
    if source_city == destination_city:
        st.warning("Source and destination cities must be different.")
    else:
        flight = pd.DataFrame([{
            "airline": airline,
            "source_city": source_city,
            "departure_time": departure_time,
            "stops": stops,
            "arrival_time": arrival_time,
            "destination_city": destination_city,
            "class": travel_class,
            "duration": float(duration),
            "days_left": int(days_left),
        }], columns=FEATURES)

        try:
            price = float(model.predict(flight)[0])
            st.success(f"Estimated Ticket Price: ₹{price:,.2f}")
            st.caption("Estimated price in Indian Rupees (INR). Actual ticket prices can differ.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
