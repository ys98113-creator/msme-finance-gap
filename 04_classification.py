# ============================================================
# MSME FINANCE GAP — CLASSIFICATION MODEL
# Goal: Predict which firms are credit-constrained
# Model: Random Forest Classifier
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─── Load Data ───────────────────────────────────────────────
df = pd.read_csv("data/enterprise_survey_india.csv")

print("Data loaded!")
print(f"Total firms: {len(df)}")
print(f"Credit-constrained: {df['credit_constrained'].sum()} ({df['credit_constrained'].mean()*100:.1f}%)")
print(f"Not constrained:    {(1-df['credit_constrained']).sum()} ({(1-df['credit_constrained']).mean()*100:.1f}%)")


# ─── Feature Engineering ─────────────────────────────────────
# Encode categorical variables for the model
le_state  = LabelEncoder()
le_sector = LabelEncoder()
le_size   = LabelEncoder()

df["state_encoded"]  = le_state.fit_transform(df["state"])
df["sector_encoded"] = le_sector.fit_transform(df["sector"])
df["size_encoded"]   = le_size.fit_transform(df["firm_size"].astype(str))

feature_cols = [
    "size_encoded",
    "sector_encoded",
    "state_encoded",
    "has_bank_account",
    "annual_revenue_usd",
    "data_sparse_region"
]

feature_labels = [
    "Firm Size",
    "Sector",
    "State",
    "Has Bank Account",
    "Annual Revenue",
    "Data-Sparse Region"
]

X = df[feature_cols]
y = df["credit_constrained"]


# ─── Random Forest Model ─────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=4,
    random_state=42,
    class_weight="balanced"
)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")

print(f"\nRandom Forest (5-fold CV):")
print(f"  Mean ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Fit on full data for visualisation
rf.fit(X, y)
y_pred      = rf.predict(X)
y_pred_prob = rf.predict_proba(X)[:, 1]

print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=["Not Constrained", "Credit Constrained"]))


# ─── Feature Importance ──────────────────────────────────────
importance_df = pd.DataFrame({
    "feature": feature_labels,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=True)

print("Feature Importances:")
print(importance_df.to_string(index=False))


# ─── Firm Risk Scores ────────────────────────────────────────
df["constraint_probability"] = y_pred_prob

print("\nHighest-risk firms:")
print(df[["firm_id", "state", "firm_size", "sector", "constraint_probability"]]
      .sort_values("constraint_probability", ascending=False)
      .head(10).to_string(index=False))


# ─── Visualisation ───────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("MSME Credit Constraint — Random Forest Classifier",
             fontsize=14, fontweight="bold")

# --- Chart 1: Feature importance ---
ax1 = axes[0, 0]
colors = ["#1565C0" if imp == importance_df["importance"].max() else "steelblue"
          for imp in importance_df["importance"]]
bars = ax1.barh(importance_df["feature"], importance_df["importance"],
                color=colors, edgecolor="white")
ax1.set_title("Feature Importance\n(What predicts credit constraints?)")
ax1.set_xlabel("Importance Score")
ax1.grid(True, alpha=0.3)
for bar, val in zip(bars, importance_df["importance"]):
    ax1.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", fontsize=9)

# --- Chart 2: Confusion matrix ---
ax2 = axes[0, 1]
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2,
            xticklabels=["Not Constrained", "Constrained"],
            yticklabels=["Not Constrained", "Constrained"],
            linewidths=0.5)
ax2.set_title("Confusion Matrix")
ax2.set_ylabel("Actual")
ax2.set_xlabel("Predicted")

# --- Chart 3: ROC Curve ---
ax3 = axes[1, 0]
fpr, tpr, _ = roc_curve(y, y_pred_prob)
roc_auc = auc(fpr, tpr)
ax3.plot(fpr, tpr, color="steelblue", linewidth=2,
         label=f"ROC Curve (AUC = {roc_auc:.2f})")
ax3.plot([0,1], [0,1], "r--", linewidth=1.5, label="Random classifier")
ax3.fill_between(fpr, tpr, alpha=0.1, color="steelblue")
ax3.set_title("ROC Curve")
ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.legend()
ax3.grid(True, alpha=0.3)

# --- Chart 4: Risk score distribution by firm size ---
ax4 = axes[1, 1]
size_order = ["Micro", "Small", "Medium"]
palette    = {"Micro": "crimson", "Small": "coral", "Medium": "steelblue"}
for size in size_order:
    subset = df[df["firm_size"] == size]["constraint_probability"]
    ax4.hist(subset, bins=8, alpha=0.6, label=size,
             color=palette[size], edgecolor="white")
ax4.set_title("Credit Constraint Probability\nby Firm Size")
ax4.set_xlabel("Predicted Probability of Being Constrained")
ax4.set_ylabel("Number of Firms")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/msme_classification.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nClassification complete. Chart saved to outputs/msme_classification.png")
print("Next step: run 05_gap_prediction.py")
