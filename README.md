# Explainable Cardiovascular Disease-Status Prototype

Educational prototype. Not a medical device. Not clinically validated.

## Files
- convert.py: converts the OpenML ARFF file to CSV
- step1_eda.py: data profiling and implausible-value flags
- step2_models.py: stratified split, 4 models, metrics
- step3_shap.py: SHAP plots
- app.py: Streamlit app

## Run
python -m streamlit run app.py

## Status
Done: EDA, 4 models, SHAP, Streamlit app.
Pending: SQLite history, cross-validation, tuning, calibration.

## Data
Dataset not included. Source: OpenML dataset 45547 (original source and licence unverified).
