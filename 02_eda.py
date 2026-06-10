# ============================================================
# MSME FINANCE GAP — EXPLORATORY DATA ANALYSIS (EDA)
# Focus: India — firm-level credit constraints + regional gaps
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ─── Load Cleaned Data ────────────────────────────────────────
df_ifc = pd.read_csv("data/ifc_msme_gap.csv")
df = pd.read_csv("data/enterprise_survey_india.csv")

print("Data loaded successfully!")
print(f"IFC dataset: {df_ifc.shape}")
print(f"Enterprise survey: {df.shape}")


# ─── Basic Exploration ────────────────────────────────────────
print("\n=== ENTERPRISE SURVEY OVERVIEW ===")
print(df.describe())

print("\n=== CREDIT CONSTRAINT RATE BY FIRM SIZE ===")
size_summary = df.groupby("firm_size", observed=True)["credit_constrained"].agg(
    total="count",
    constrained="sum"
)
size_summary["constrained_pct"] = (size_summary["constrained"] / size_summary["total"] * 100).round(1)
print(size_summary)

print("\n=== CREDIT CONSTRAINT RATE BY STATE ===")
state_summary = df.groupby("state")["credit_constrained"].agg(
    total="count",
    constrained="sum"
)
state_summary["constrained_pct"] = (state_summary["constrained"] / state_summary["total"] * 100).round(1)
state_summary = state_summary.sort_values("constrained_pct", ascending=False)
print(state_summary)

print("\n=== DATA SPARSE REGIONS ===")
sparse = df[df["data_sparse_region"] == 1]
print(f"Firms in data-sparse regions: {len(sparse)} ({len(sparse)/len(df)*100:.1f}% of sample)")
print(f"Credit constraint rate — sparse regions: {sparse['credit_constrained'].mean()*100:.1f}%")
print(f"Credit constraint rate — data-rich regions: {df[df['data_sparse_region']==0]['credit_constrained'].mean()*100:.1f}%")


# ─── Visualisation ───────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("MSME Finance Gap — India Analysis\n(IFC Data + World Bank Enterprise Survey)",
             fontsize=15, fontweight="bold", y=1.01)

# --- Chart 1: Finance gap over time (India) ---
ax1 = axes[0, 0]
india = df_ifc[df_ifc["country"] == "India"].sort_values("year")
ax1.bar(india["year"], india["finance_gap_usd_bn"], color="steelblue", edgecolor="white")
ax1.set_title("India MSME Finance Gap Over Time")
ax1.set_xlabel("Year")
ax1.set_ylabel("Finance Gap (USD Billion)")
ax1.grid(True, alpha=0.3)
for i, row in india.iterrows():
    ax1.text(row["year"], row["finance_gap_usd_bn"] + 5, f"${row['finance_gap_usd_bn']}B",
             ha="center", fontsize=9)

# --- Chart 2: Credit constraint rate by firm size ---
ax2 = axes[0, 1]
size_colors = {"Micro": "crimson", "Small": "coral", "Medium": "steelblue"}
bars = ax2.bar(size_summary.index, size_summary["constrained_pct"],
               color=[size_colors[s] for s in size_summary.index], edgecolor="white")
ax2.set_title("Credit Constraint Rate by Firm Size")
ax2.set_xlabel("Firm Size")
ax2.set_ylabel("% Credit Constrained")
ax2.grid(True, alpha=0.3)
for bar, val in zip(bars, size_summary["constrained_pct"]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{val}%", ha="center", fontsize=10, fontweight="bold")

# --- Chart 3: Constraint rate by state ---
ax3 = axes[0, 2]
colors = ["#d32f2f" if s in ["Bihar", "Odisha", "Jharkhand", "Uttar Pradesh", "Rajasthan"]
          else "steelblue" for s in state_summary.index]
bars = ax3.barh(state_summary.index, state_summary["constrained_pct"],
                color=colors, edgecolor="white")
ax3.set_title("Credit Constraint Rate by State\n(Red = Data-Sparse Region)")
ax3.set_xlabel("% Credit Constrained")
ax3.grid(True, alpha=0.3)
rich_patch = mpatches.Patch(color="steelblue", label="Data-rich state")
sparse_patch = mpatches.Patch(color="#d32f2f", label="Data-sparse state")
ax3.legend(handles=[rich_patch, sparse_patch], fontsize=8)

# --- Chart 4: Bank account vs credit access ---
ax4 = axes[1, 0]
categories = ["Has Bank Account", "Applied for Credit", "Credit Approved"]
values = [df["has_bank_account"].mean()*100,
          df["applied_for_credit"].mean()*100,
          df["credit_approved"].mean()*100]
bar_colors = ["steelblue", "coral", "seagreen"]
bars = ax4.bar(categories, values, color=bar_colors, edgecolor="white")
ax4.set_title("Financial Access Funnel")
ax4.set_ylabel("% of Firms")
ax4.set_ylim(0, 100)
ax4.grid(True, alpha=0.3)
for bar, val in zip(bars, values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")

# --- Chart 5: Credit gap by sector ---
ax5 = axes[1, 1]
sector_gap = df.groupby("sector")["credit_gap_usd"].mean().sort_values(ascending=False)
ax5.barh(sector_gap.index, sector_gap.values / 1000, color="mediumpurple", edgecolor="white")
ax5.set_title("Average Credit Gap by Sector")
ax5.set_xlabel("Average Credit Gap (USD Thousands)")
ax5.grid(True, alpha=0.3)

# --- Chart 6: Regional data coverage issue ---
ax6 = axes[1, 2]
region_counts = df.groupby(["state", "data_sparse_region"]).size().reset_index(name="count")
region_sample = df["state"].value_counts()
bar_colors_region = ["#d32f2f" if s in ["Bihar", "Odisha", "Jharkhand", "Uttar Pradesh", "Rajasthan"]
                     else "steelblue" for s in region_sample.index]
ax6.bar(region_sample.index, region_sample.values, color=bar_colors_region, edgecolor="white")
ax6.set_title("Sample Coverage by State\n(Red = Data-Sparse — Key Limitation)")
ax6.set_xlabel("State")
ax6.set_ylabel("Number of Firms Surveyed")
ax6.tick_params(axis="x", rotation=45)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/msme_eda.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nEDA complete. Chart saved to outputs/msme_eda.png")
print("Next step: run 03_clustering.py")
