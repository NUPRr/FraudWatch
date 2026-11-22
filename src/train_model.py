"""Trains fraud-detection models, compares them, and saves results for the dashboard."""
import json
import os

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

THRESHOLD = 0.5  # a transaction is flagged when its fraud score is at least this

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/transactions_clean.csv")
df["log_amount"] = np.log1p(df["amount"])
df["is_night"] = (df["hour"] < 6).astype(int)

NUM = ["log_amount", "hour", "is_night", "is_foreign", "new_device"]
CAT = ["channel", "merchant_category"]
X, y = df[NUM + CAT], df["is_fraud"]

# stratify keeps the same ~2% fraud share in both the training and the test part
X_train, X_test, y_train, y_test, _, df_test = train_test_split(
    X, y, df, test_size=0.25, stratify=y, random_state=42)

prep = ColumnTransformer([
    ("num", StandardScaler(), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
])

models = {
    "Logistic Regression": Pipeline([
        ("prep", prep),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
    "Random Forest": Pipeline([
        ("prep", prep),
        ("model", RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                         class_weight="balanced_subsample",
                                         n_jobs=-1, random_state=42))]),
}

print(f"Training on {len(X_train)} transactions, testing on {len(X_test)} "
      f"({int(y_test.sum())} of them are fraud)\n")

results, probas = {}, {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    results[name] = {
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
    }
    probas[name] = proba

print(f"{'Model':<22}{'Precision':>10}{'Recall':>9}{'F1':>8}{'ROC-AUC':>10}{'PR-AUC':>9}")
for name, m in results.items():
    print(f"{name:<22}{m['precision']:>10.3f}{m['recall']:>9.3f}{m['f1']:>8.3f}"
          f"{m['roc_auc']:>10.3f}{m['pr_auc']:>9.3f}")

best = max(results, key=lambda n: results[n]["pr_auc"])
proba = probas[best]
pred = (proba >= THRESHOLD).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

fraud_total = float(df_test.loc[df_test["is_fraud"] == 1, "amount"].sum())
fraud_caught = float(df_test.loc[(pred == 1) & (df_test["is_fraud"] == 1), "amount"].sum())

print(f"\nBest model: {best}  (threshold {THRESHOLD})")
print(f"Fraud caught:      {tp} of {tp + fn}")
print(f"Fraud missed:      {fn}")
print(f"False alarms:      {fp}  (good transactions flagged by mistake)")
print(f"Fraud value caught: Rs {fraud_caught:,.0f} of Rs {fraud_total:,.0f} "
      f"({100 * fraud_caught / fraud_total:.1f}%)")

# ---- Save everything the dashboard will need ----
out = df_test.copy()
out["fraud_score"] = proba.round(4)
out["flagged"] = pred
out.sort_values("fraud_score", ascending=False).to_csv("outputs/scored_transactions.csv", index=False)

summary = {
    "best_model": best, "threshold": THRESHOLD,
    "test_size": int(len(y_test)), "fraud_in_test": int(y_test.sum()),
    "caught": int(tp), "missed": int(fn), "false_alarms": int(fp),
    "fraud_value_caught": fraud_caught, "fraud_value_total": fraud_total,
    "models": results,
}
with open("outputs/metrics.json", "w") as f:
    json.dump(summary, f, indent=2)
joblib.dump(models[best], "outputs/fraud_model.joblib")

# ---- Chart 1: which signals matter most (Random Forest) ----
rf = models["Random Forest"]
names = [n.split("__", 1)[1] for n in rf.named_steps["prep"].get_feature_names_out()]
imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names).nlargest(10)
plt.figure(figsize=(7, 4.5))
imp.sort_values().plot.barh(color="#1b4f9c")
plt.title("What the model looks at most")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150)
plt.close()

# ---- Chart 2: precision vs recall for both models ----
plt.figure(figsize=(6, 4.5))
for name, pr in probas.items():
    p_, r_, _ = precision_recall_curve(y_test, pr)
    plt.plot(r_, p_, label=f"{name} (PR-AUC {results[name]['pr_auc']:.2f})")
plt.xlabel("Recall (share of fraud caught)")
plt.ylabel("Precision (share of flags that are real fraud)")
plt.title("Model comparison")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/pr_curve.png", dpi=150)
plt.close()

print("\nSaved to outputs/: scored_transactions.csv, metrics.json, fraud_model.joblib, "
      "feature_importance.png, pr_curve.png")