import pandas as pd

FILE = "cardio_clean.csv"

# Some versions of this dataset use ';' as the separator, some use ','
df = pd.read_csv(FILE, sep=";")
if df.shape[1] == 1:
    df = pd.read_csv(FILE, sep=",")

print("Shape (rows, columns):", df.shape)
print("\nColumn types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nClass balance (cardio):\n", df["cardio"].value_counts())
print("\nClass balance (%):\n", (df["cardio"].value_counts(normalize=True) * 100).round(2))
print("\nSummary statistics:\n", df.describe().T)

# Implausible-value flags (these are screening rules, not clinical thresholds)
flag = (
    (df["height"] < 100) | (df["height"] > 220) |
    (df["weight"] < 30) |
    (df["ap_hi"] < df["ap_lo"]) |
    (df["ap_hi"] < 70) | (df["ap_hi"] > 250) |
    (df["ap_lo"] < 40) | (df["ap_lo"] > 200)
)
print("\nFlagged records:", int(flag.sum()))
print("Acceptable records:", int((~flag).sum()))