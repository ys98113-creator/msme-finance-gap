# ============================================================
# MSME FINANCE GAP — CLUSTERING ANALYSIS
# Goal: Group Indian states by finance gap profile using K-Means
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# ─── Load Data ───────────────────────────────────────────────
df = pd.read_csv("data/enterprise_survey_india.csv")

print("Data loaded successfully!")
print(f"Firms: {len(df)} across {df['state'].nunique()} states")


# ─── Build State-Level Features ──────────────────────────────
# Aggregate firm-level data up to state level for clustering
state_features = df.groupby("state").agg(
    total_firms        = ("firm_id", "count"),
    credit_constrained_pct = ("credit_constrained", "mean"),
    bank_account_pct   = ("has_bank_account", "mean"),
    credit_applied_pct = ("applied_for_credit", "mean"),
    avg_credit_gap_usd = ("credit_gap_usd", "mean"),
    avg_revenue_usd    = ("annual_revenue_usd", "mean"),
    micro_pct          = ("firm_size", lambda x: (x == "Micro").mean()),
    data_sparse        = ("data_sparse_region", "max")
).reset_index()

print("\nState-level features:")
print(state_features[["state", "credit_constrained_pct", "bank_account_pct", "avg_credit_gap_usd"]])


# ─── Standardise Features ────────────────────────────────────
# K-Means is distance-based — features need to be on same scale
feature_cols = [
    "credit_constrained_pct",
    "bank_account_pct",
    "credit_applied_pct",
    "avg_credit_gap_usd",
    "micro_pct"
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(state_features[feature_cols])


# ─── Find Optimal Number of Clusters (Elbow Method) ─────────
inertia = []
silhouette = []
K_range = range(2, 6)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
    silhouette.append(silhouette_score(X_scaled, km.labels_))

print("\nSilhouette scores by K:")
for k, s in zip(K_range, silhouette):
    print(f"  K={k}: {s:.3f}")


# ─── Fit Final Model (K=3) ───────────────────────────────────
km_final = KMeans(n_clusters=3, random_state=42, n_init=10)
state_features["cluster"] = km_final.fit_predict(X_scaled)

# Label clusters meaningfully based on their profile
cluster_means = state_features.groupby("cluster")[feature_cols].mean()
print("\nCluster profiles:")
print(cluster_means.round(3))

# Assign human-readable labels
cluster_labels = {
    state_features.groupby("cluster")["credit_constrained_pct"].mean().idxmax(): "High Risk",
    state_features.groupby("cluster")["credit_constrained_pct"].mean().idxmin(): "Low Risk",
}
middle = [c for c in [0,1,2] if c not in cluster_labels][0]
cluster_labels[middle] = "Moderate Risk"

state_features["cluster_label"] = state_features["cluster"].map(cluster_labels)

print("\nState cluster assignments:")
print(state_features[["state", "cluster_label", "credit_constrained_pct",
                        "bank_account_pct", "data_sparse"]].to_string(index=False))


# ─── Visualisation ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("MSME Finance Gap — State Clustering Analysis (K-Means)",
             fontsize=14, fontweight="bold")

# --- Chart 1: Elbow curve ---
ax1 = axes[0]
ax1.plot(list(K_range), inertia, marker="o", color="steelblue", linewidth=2)
ax1.axvline(x=3, color="crimson", linestyle="--", alpha=0.7, label="Chosen K=3")
ax1.set_title("Elbow Method — Optimal K")
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("Inertia")
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Chart 2: Cluster scatter plot ---
ax2 = axes[1]
colors = {"High Risk": "crimson", "Moderate Risk": "coral", "Low Risk": "seagreen"}
for label, group in state_features.groupby("cluster_label"):
    ax2.scatter(
        group["bank_account_pct"],
        group["credit_constrained_pct"],
        label=label,
        color=colors[label],
        s=150,
        edgecolors="white",
        linewidth=1.5
    )
    for _, row in group.iterrows():
        ax2.annotate(row["state"], (row["bank_account_pct"], row["credit_constrained_pct"]),
                     fontsize=7.5, xytext=(4, 4), textcoords="offset points")
ax2.set_title("State Clusters\n(Bank Access vs Credit Constraint)")
ax2.set_xlabel("Bank Account Access Rate")
ax2.set_ylabel("Credit Constraint Rate")
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- Chart 3: Cluster profile heatmap ---
ax3 = axes[2]
heatmap_data = state_features.groupby("cluster_label")[feature_cols].mean()
heatmap_data.columns = ["Constrained %", "Bank Account %", "Applied %", "Avg Gap $", "Micro %"]
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn_r",
    ax=ax3,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
ax3.set_title("Cluster Profiles\n(Feature Averages)")
ax3.set_ylabel("")

plt.tight_layout()
plt.savefig("outputs/msme_clusters.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nClustering complete. Chart saved to outputs/msme_clusters.png")
print("Next step: run 05_gap_prediction.py")
