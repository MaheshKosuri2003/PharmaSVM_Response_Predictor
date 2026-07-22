import streamlit as st
import pandas as pd
import joblib


# Load model and scaler
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")


# App title
st.title("PharmaSVM Response Predictor")

st.write("Enter the following details to predict Drug Response.")


# Input fields
drug_dosage = st.number_input(
    "Drug Dosage (mg)",
    value=0.0
)

systolic_blood_pressure = st.number_input(
    "Systolic Blood Pressure (mmHg)",
    value=0.0
)

heart_rate = st.number_input(
    "Heart Rate (BPM)",
    value=0.0
)

liver_toxicity_index = st.number_input(
    "Liver Toxicity Index (U/L)",
    value=0.0
)

blood_glucose_level = st.number_input(
    "Blood Glucose Level (mg/dL)",
    value=0.0
)


# Prediction button
if st.button("Predict Drug Response"):

    # Create input DataFrame
    input_data = pd.DataFrame(
        [[
            drug_dosage,
            systolic_blood_pressure,
            heart_rate,
            liver_toxicity_index,
            blood_glucose_level
        ]],
        columns=[
            "Drug Dosage (mg)",
            "Systolic Blood Pressure (mmHg)",
            "Heart Rate (BPM)",
            "Liver Toxicity Index (U/L)",
            "Blood Glucose Level (mg/dL)"
        ]
    )


    # Scale the input data
    input_data_scaled = scaler.transform(input_data)


    # Make prediction
    prediction = model.predict(input_data_scaled)


    # Display result
    if prediction[0] == 1:
        st.success("Positive Drug Response")
    else:
        st.error("Negative Drug Response")