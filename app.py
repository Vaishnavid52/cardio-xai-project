import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Cardio Status Prototype", layout="centered")
st.title("Explainable Cardiovascular Disease-Status Prototype")
st.warning("Educational prototype. Not a medical device and not clinically validated. "
           "It classifies recorded disease status from a dataset; it does not predict the future.")

saved = joblib.load("best_model.joblib")
model, features = saved["model"], saved["features"]

col1, col2 = st.columns(2)
age_years = col1.number_input("Age (years)", 18, 100, 50)
gender = col1.selectbox("Gender (1 = women, 2 = men)", [1, 2])
height = col1.number_input("Height (cm)", 50, 260, 165)
weight = col1.number_input("Weight (kg)", 10.0, 250.0, 70.0)
ap_hi = col2.number_input("Systolic BP, ap_hi (mmHg)", -50, 300, 120)
ap_lo = col2.number_input("Diastolic BP, ap_lo (mmHg)", -50, 250, 80)
chol = col2.selectbox("Cholesterol (1 normal, 2 above normal, 3 well above)", [1, 2, 3])
gluc = col2.selectbox("Glucose (1 normal, 2 above normal, 3 well above)", [1, 2, 3])
smoke = col1.selectbox("Smoker (0 no, 1 yes)", [0, 1])
alco = col1.selectbox("Alcohol (0 no, 1 yes)", [0, 1])
active = col2.selectbox("Physically active (0 no, 1 yes)", [0, 1], index=1)

if st.button("Check"):
    # Input quality flags (screening rules, not clinical thresholds)
    problems = []
    if ap_hi < ap_lo:
        problems.append("Systolic is lower than diastolic.")
    if ap_hi < 70 or ap_hi > 250:
        problems.append("Systolic BP is outside the plausible range 70-250.")
    if ap_lo < 40 or ap_lo > 200:
        problems.append("Diastolic BP is outside the plausible range 40-200.")
    if height < 100 or height > 220:
        problems.append("Height is outside the plausible range 100-220 cm.")
    if weight < 30:
        problems.append("Weight is below 30 kg.")
    for p in problems:
        st.error("Input flag: " + p)

    row = pd.DataFrame([{
        "gender": gender, "height": height, "weight": weight, "ap_hi": ap_hi,
        "ap_lo": ap_lo, "cholesterol": chol, "gluc": gluc, "smoke": smoke,
        "alco": alco, "active": active, "age_years": age_years}])[features]

    prob = model.predict_proba(row)[0, 1]
    st.metric("Model probability of recorded disease status", f"{prob:.1%}")
    st.write("Label:", "Disease recorded" if prob >= 0.5 else "No disease recorded")
    if problems:
        st.info("Result may be unreliable because of the flagged inputs.")