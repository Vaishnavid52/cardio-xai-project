import re
import joblib
import pandas as pd
import streamlit as st

# pypdf is optional: if it is not installed, only the paste box works
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

st.set_page_config(page_title="Cardio Status Prototype", layout="centered")
st.title("Explainable Cardiovascular Disease-Status Prototype")
st.warning("Educational prototype. Not a medical device and not clinically validated. "
           "It classifies recorded disease status from a dataset; it does not predict the future.")

saved = joblib.load("best_model.joblib")
model, features = saved["model"], saved["features"]

# ------------------------------------------------------------------
# Part 1: Cardiovascular disease-status check
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Part 2: Report value checker (rule-based, no AI)
# ------------------------------------------------------------------
st.divider()
st.header("Report Value Checker (rule-based, no AI)")
st.warning("Checks values against reference ranges. Not a diagnosis and not medical "
           "advice. Use sample or dummy reports only.")

# Rough general adult ranges, used ONLY when the report has no range of its own.
# Real ranges vary by lab, age and gender, so the report's own range is preferred.
DEFAULTS = {
    "hemoglobin": (12.0, 17.5),
    "glucose": (70, 99),
    "cholesterol": (0, 200),
    "triglycerides": (0, 150),
    "creatinine": (0.6, 1.3),
}

LINE = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z \(\)/%\.]{2,40}?)\s*[:\-]?\s*"
    r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z/%]*)\s*"
    r"(?:\(?\s*(?P<lo>\d+(?:\.\d+)?)\s*(?:-|to)\s*(?P<hi>\d+(?:\.\d+)?)\s*\)?)?")

if HAS_PYPDF:
    up = st.file_uploader("Upload report (PDF or TXT)", type=["pdf", "txt"])
else:
    up = st.file_uploader("Upload report (TXT only, PDF support not installed)", type=["txt"])
pasted = st.text_area("...or paste report text here", height=150)

if st.button("Check report"):
    text = pasted
    if up is not None:
        if up.name.endswith(".pdf") and HAS_PYPDF:
            text = "\n".join((p.extract_text() or "") for p in PdfReader(up).pages)
        else:
            text = up.read().decode("utf-8", errors="ignore")

    rows = []
    for line in text.splitlines():
        m = LINE.search(line.strip())
        if not m:
            continue
        name, val = m["name"].strip(), float(m["val"])
        if m["lo"] and m["hi"]:
            lo, hi, src = float(m["lo"]), float(m["hi"]), "report"
        else:
            key = next((k for k in DEFAULTS if k in name.lower()), None)
            if key is None:
                continue
            (lo, hi), src = DEFAULTS[key], "general (rough)"
        status = "Low" if val < lo else "High" if val > hi else "Within range"
        rows.append({"Test": name, "Value": val, "Unit": m["unit"],
                     "Range": f"{lo} - {hi}", "Range source": src, "Status": status})

    if rows:
        st.dataframe(pd.DataFrame(rows))
        st.caption("Values outside range are worth discussing with a doctor. "
                   "This tool does not diagnose.")
    else:
        st.error("No values found. Scanned or image reports are not supported.")