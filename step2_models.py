import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

SEED = 42   # same seed everywhere = reproducible results

df = pd.read_csv("cardio_clean.csv")

# Same screening rules as step 1
flag = (
    (df["height"] < 100) | (df["height"] > 220) |
    (df["weight"] < 30) |
    (df["ap_hi"] < df["ap_lo"]) |
    (df["ap_hi"] < 70) | (df["ap_hi"] > 250) |
    (df["ap_lo"] < 40) | (df["ap_lo"] > 200)
)

# Age in days -> years, and drop id (it has no medical meaning)
df["age_years"] = df["age"] / 365.25
df = df.drop(columns=["id", "age"])

# Two scenarios so you can report the effect of the flagged rows
scenarios = {
    "A_acceptable_only": df[~flag],
    "B_all_records": df,
}

def make_models():
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=SEED))]),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, n_jobs=-1, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=SEED),
    }

for name, data in scenarios.items():
    print("\n" + "=" * 60)
    print("SCENARIO:", name, "| rows:", len(data))

    X = data.drop(columns=["cardio"])
    y = data["cardio"]

    # Split BEFORE any fitting; stratify keeps the class ratio equal in both parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    print("Train rows:", len(X_train), "| Test rows:", len(X_test))

    rows, fitted = [], {}
    for mname, model in make_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": mname,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "ROC_AUC": roc_auc_score(y_test, prob),
        })
        fitted[mname] = model
        print("\nConfusion matrix -", mname, "(rows = actual 0/1, cols = predicted 0/1)")
        print(confusion_matrix(y_test, pred))

    results = pd.DataFrame(rows).round(4)
    print("\nMETRICS TABLE -", name)
    print(results.to_string(index=False))
    results.to_csv(f"metrics_{name}.csv", index=False)

    # Save the best model (by ROC-AUC) from scenario A for the app
    if name == "A_acceptable_only":
        best = results.sort_values("ROC_AUC", ascending=False).iloc[0]["Model"]
        joblib.dump({"model": fitted[best], "features": list(X.columns), "name": best},
                    "best_model.joblib")
        print("\nSaved best model:", best, "-> best_model.joblib")