# ============================================================
# MSME FINANCE GAP — GAP SIZE PREDICTION
# Goal: Predict country-level finance gap using macro indicators
# Model: XGBoost Regressor + SHAP explainability
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ─── Load IFC Country-Level Data ─────────────────────────────
df = pd.read_csv("data/ifc_msme_gap.csv")

print("Data loaded!")
print(df[["country", "year", "finance_gap_usd_bn", "credit_constrained_pct",
          "formal_msmes_pct", "gdp_per_capita_usd"]].to_string(index=False))


# ─── Feature Engineering ─────────────────────────────────────
# Features: macro indicators that predict gap size
feature_cols = [
    "credit_constrained_pct",
    "formal_msmes_pct",
    "gdp_per_capita_usd",
    "msme_count_millions"
]

target_col = "finance_gap_usd_bn"

X = df[feature_cols]
y = df[target_col]

print(f"\nFeatures: {feature_cols}")
print(f"Target: {target_col}")
print(f"Observations: {len(df)}")


# ─── Baseline: Ridge Regression ──────────────────────────────
# Start simple before XGBoost — good practice and interpretable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ridge = Ridge(alpha=1.0)
loo = LeaveOneOut()
ridge_scores = cross_val_score(ridge, X_scaled, y, cv=loo, scoring="r2")

print(f"\nRidge Regression (Leave-One-Out CV):")
print(f"  Mean R²: {ridge_scores.mean():.3f}")


# ─── XGBoost Model ───────────────────────────────────────────
xgb = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    verbosity=0
)

xgb_scores = cross_val_score(xgb, X, y, cv=loo, scoring="r2")
print(f"\nXGBoost (Leave-One-Out CV):")
print(f"  Mean R²: {xgb_scores.mean():.3f}")

# Fit on full data for visualisation
xgb.fit(X, y)
y_pred = xgb.predict(X)

mae = mean_absolute_error(y, y_pred)
r2  = r2_score(y, y_pred)
print(f"  In-sample MAE: ${mae:.1f}B")
print(f"  In-sample R²:  {r2:.3f}")


# ─── Feature Importance ──────────────────────────────────────
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": xgb.feature_importances_
}).sort_values("importance", ascending=True)

print("\nFeature Importances:")
print(importance_df.to_string(index=False))


# ─── Country Predictions ─────────────────────────────────────
df["predicted_gap"] = y_pred
df["residual"] = df["finance_gap_usd_bn"] - df["predicted_gap"]

print("\nActual vs Predicted Gap (USD Billion):")
print(df[["country", "year", "finance_gap_usd_bn", "predicted_gap", "residual"]
         ].round(1).to_string(index=False))


# ─── Visualisation ───────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("MSME Finance Gap — Prediction Model (XGBoost)",
             fontsize=14, fontweight="bold")

# --- Chart 1: Actual vs Predicted ---
ax1 = axes[0, 0]
ax1.scatter(y, y_pred, color="steelblue", s=80, edgecolors="white", linewidth=1.5, zorder=3)
min_val = min(y.min(), y_pred.min()) - 20
max_val = max(y.max(), y_pred.max()) + 20
ax1.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect prediction")
for i, row in df.iterrows():
    ax1.annotate(f"{row['country']} ({row['year']})",
                 (row["finance_gap_usd_bn"], row["predicted_gap"]),
                 fontsize=7, xytext=(5, 5), textcoords="offset points")
ax1.set_title("Actual vs Predicted Finance Gap")
ax1.set_xlabel("Actual Gap (USD Billion)")
ax1.set_ylabel("Predicted Gap (USD Billion)")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.text(0.05, 0.92, f"R² = {r2:.2f}\nMAE = ${mae:.0f}B",
         transform=ax1.transAxes, fontsize=9,
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

# --- Chart 2: Feature Importance ---
ax2 = axes[0, 1]
colors = ["#2196F3" if imp == importance_df["importance"].max() else "steelblue"
          for imp in importance_df["importance"]]
bars = ax2.barh(importance_df["feature"], importance_df["importance"],
                color=colors, edgecolor="white")
ax2.set_title("Feature Importance\n(XGBoost — what drives the gap?)")
ax2.set_xlabel("Importance Score")
ax2.grid(True, alpha=0.3)
for bar, val in zip(bars, importance_df["importance"]):
    ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", fontsize=9)

# --- Chart 3: India gap trend + prediction ---
ax3 = axes[1, 0]
india = df[df["country"] == "India"].sort_values("year")
ax3.plot(india["year"], india["finance_gap_usd_bn"], "o-",
         color="steelblue", linewidth=2, label="Actual", markersize=7)
ax3.plot(india["year"], india["predicted_gap"], "s--",
         color="crimson", linewidth=2, label="Predicted", markersize=7)
ax3.fill_between(india["year"], india["finance_gap_usd_bn"],
                 india["predicted_gap"], alpha=0.15, color="coral")
ax3.set_title("India: Actual vs Predicted Gap Over Time")
ax3.set_xlabel("Year")
ax3.set_ylabel("Finance Gap (USD Billion)")
ax3.legend()
ax3.grid(True, alpha=0.3)

# --- Chart 4: Gap vs GDP per capita (macro relationship) ---
ax4 = axes[1, 1]
scatter = ax4.scatter(df["gdp_per_capita_usd"], df["finance_gap_usd_bn"],
                      c=df["credit_constrained_pct"], cmap="RdYlGn_r",
                      s=100, edgecolors="white", linewidth=1.5)
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label("Credit Constrained %", fontsize=9)
for _, row in df[df["country"] == "India"].iterrows():
    ax4.annotate("India", (row["gdp_per_capita_usd"], row["finance_gap_usd_bn"]),
                 fontsize=8, xytext=(8, 4), textcoords="offset points", color="navy")
ax4.set_title("GDP per Capita vs Finance Gap\n(color = credit constraint rate)")
ax4.set_xlabel("GDP per Capita (USD)")
ax4.set_ylabel("Finance Gap (USD Billion)")
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/msme_gap_prediction.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nGap prediction complete. Chart saved to outputs/msme_gap_prediction.png")
print("\nProject complete. All outputs in /outputs folder.")
print("Next step: add README.md to complete the GitHub portfolio.")
