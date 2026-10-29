import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

SEED = 42
saved = joblib.load("best_model.joblib")
model, features, mname = saved["model"], saved["features"], saved["name"]
print("Explaining model:", mname)

df = pd.read_csv("cardio_clean.csv")
flag = (
    (df["height"] < 100) | (df["height"] > 220) | (df["weight"] < 30) |
    (df["ap_hi"] < df["ap_lo"]) | (df["ap_hi"] < 70) | (df["ap_hi"] > 250) |
    (df["ap_lo"] < 40) | (df["ap_lo"] > 200)
)
df["age_years"] = df["age"] / 365.25
df = df.drop(columns=["id", "age"])
data = df[~flag]

X = data.drop(columns=["cardio"])[features]
y = data["cardio"]
# Same split and seed as step 2, so the test set is the same one
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

sample = X_test.sample(1000, random_state=SEED)

# Tree models use TreeExplainer; Logistic Regression is a Pipeline, so use a linear explainer
if hasattr(model, "named_steps"):
    scaler = model.named_steps["scale"]
    lr = model.named_steps["model"]
    explainer = shap.LinearExplainer(lr, scaler.transform(X_train))
    sv = explainer(scaler.transform(sample))
    sv.feature_names = features
else:
    explainer = shap.TreeExplainer(model)
    sv = explainer(sample)
    if sv.values.ndim == 3:        # some versions return one set per class
        sv = sv[:, :, 1]

plt.figure()
shap.plots.bar(sv, show=False)
plt.tight_layout()
plt.savefig("shap_global_bar.png", dpi=150)
plt.close()

plt.figure()
shap.plots.beeswarm(sv, show=False)
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=150)
plt.close()

plt.figure()
shap.plots.waterfall(sv[0], show=False)
plt.tight_layout()
plt.savefig("shap_local_example.png", dpi=150)
plt.close()

print("Saved shap_global_bar.png, shap_beeswarm.png, shap_local_example.png")