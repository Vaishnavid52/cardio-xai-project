from scipy.io import arff
import pandas as pd

SRC = "dataset_"

data, meta = arff.loadarff(SRC)
df = pd.DataFrame(data)

# ARFF text values load as bytes; decode them and turn them into numbers
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.decode("utf-8")
    try:
        df[col] = pd.to_numeric(df[col])
    except Exception:
        pass

df.to_csv("cardio_clean.csv", index=False)
print("Saved cardio_clean.csv")
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print(df.head())