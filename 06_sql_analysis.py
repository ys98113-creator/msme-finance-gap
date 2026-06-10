# ============================================================
# MSME FINANCE GAP — SQL ANALYSIS
# Goal: Query cleaned data using SQLite to answer key policy questions
# ============================================================

import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─── STEP 1: Load CSVs into SQLite Database ──────────────────
# SQLite is a lightweight database that runs inside Python
# Think of it like Excel but queryable with SQL

conn = sqlite3.connect(":memory:")  # in-memory database

df_firms   = pd.read_csv("data/enterprise_survey_india.csv")
df_ifc     = pd.read_csv("data/ifc_msme_gap.csv")

df_firms.to_sql("firms",   conn, index=False, if_exists="replace")
df_ifc.to_sql("ifc_gap",   conn, index=False, if_exists="replace")

print("Database ready. Tables loaded: firms, ifc_gap")
print(f"  firms:   {len(df_firms)} rows")
print(f"  ifc_gap: {len(df_ifc)} rows")


# ─── Helper Function ─────────────────────────────────────────
def query(sql, label=""):
    result = pd.read_sql_query(sql, conn)
    if label:
        print(f"\n{'='*55}")
        print(f"  {label}")
        print('='*55)
        print(result.to_string(index=False))
    return result


# ─── QUERY 1: Credit constraint rate by firm size ────────────
q1 = query("""
    SELECT
        firm_size,
        COUNT(*)                                      AS total_firms,
        SUM(credit_constrained)                       AS constrained_firms,
        ROUND(AVG(credit_constrained) * 100, 1)       AS constrained_pct,
        ROUND(AVG(annual_revenue_usd), 0)             AS avg_revenue_usd
    FROM firms
    GROUP BY firm_size
    ORDER BY constrained_pct DESC
""", "Q1: Credit Constraint Rate by Firm Size")


# ─── QUERY 2: Top 5 states by average credit gap ─────────────
q2 = query("""
    SELECT
        state,
        COUNT(*)                                      AS total_firms,
        ROUND(AVG(credit_gap_usd), 0)                 AS avg_credit_gap_usd,
        ROUND(AVG(credit_constrained) * 100, 1)       AS constrained_pct,
        MAX(data_sparse_region)                       AS data_sparse
    FROM firms
    GROUP BY state
    ORDER BY avg_credit_gap_usd DESC
    LIMIT 5
""", "Q2: Top 5 States by Average Credit Gap")


# ─── QUERY 3: Financial access funnel ────────────────────────
q3 = query("""
    SELECT
        'Has Bank Account'     AS stage,
        ROUND(AVG(has_bank_account) * 100, 1)     AS pct_of_firms
    FROM firms
    UNION ALL
    SELECT
        'Applied for Credit'   AS stage,
        ROUND(AVG(applied_for_credit) * 100, 1)   AS pct_of_firms
    FROM firms
    UNION ALL
    SELECT
        'Credit Approved'      AS stage,
        ROUND(AVG(credit_approved) * 100, 1)      AS pct_of_firms
    FROM firms
""", "Q3: Financial Access Funnel")


# ─── QUERY 4: Constraint rate by sector ──────────────────────
q4 = query("""
    SELECT
        sector,
        COUNT(*)                                      AS total_firms,
        ROUND(AVG(credit_constrained) * 100, 1)       AS constrained_pct,
        ROUND(AVG(credit_gap_usd), 0)                 AS avg_gap_usd
    FROM firms
    GROUP BY sector
    ORDER BY constrained_pct DESC
""", "Q4: Credit Constraint Rate by Sector")


# ─── QUERY 5: Data-sparse vs data-rich regions ───────────────
q5 = query("""
    SELECT
        CASE WHEN data_sparse_region = 1
             THEN 'Data-Sparse Region'
             ELSE 'Data-Rich Region' END             AS region_type,
        COUNT(*)                                     AS total_firms,
        ROUND(AVG(credit_constrained) * 100, 1)      AS constrained_pct,
        ROUND(AVG(credit_gap_usd), 0)                AS avg_gap_usd,
        ROUND(AVG(has_bank_account) * 100, 1)        AS bank_account_pct
    FROM firms
    GROUP BY data_sparse_region
""", "Q5: Data-Sparse vs Data-Rich Regions")


# ─── QUERY 6: Firm size × sector cross-tab ───────────────────
q6 = query("""
    SELECT
        firm_size,
        sector,
        COUNT(*)                                     AS firms,
        ROUND(AVG(credit_constrained) * 100, 1)      AS constrained_pct
    FROM firms
    WHERE sector IN ('Manufacturing', 'Services', 'Agriculture', 'Retail')
    GROUP BY firm_size, sector
    ORDER BY firm_size, constrained_pct DESC
""", "Q6: Constraint Rate by Firm Size x Sector")


# ─── QUERY 7: India finance gap trend (IFC table) ────────────
q7 = query("""
    SELECT
        year,
        finance_gap_usd_bn,
        ROUND(credit_constrained_pct * 100, 1)       AS constrained_pct,
        gdp_per_capita_usd,
        ROUND(formal_msmes_pct * 100, 1)             AS formal_msmes_pct
    FROM ifc_gap
    WHERE country = 'India'
    ORDER BY year
""", "Q7: India Finance Gap Trend (IFC Data)")


# ─── QUERY 8: Countries ranked by finance gap per MSME ───────
q8 = query("""
    SELECT
        country,
        year,
        finance_gap_usd_bn,
        msme_count_millions,
        ROUND(finance_gap_usd_bn / msme_count_millions, 2) AS gap_per_msme_usd_mn,
        gdp_per_capita_usd
    FROM ifc_gap
    ORDER BY gap_per_msme_usd_mn DESC
""", "Q8: Finance Gap per MSME by Country")


# ─── Visualisation: SQL Results as Charts ────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("MSME Finance Gap — SQL Query Results",
             fontsize=14, fontweight="bold")

# --- Chart 1: Constraint rate by firm size (Q1) ---
ax1 = axes[0, 0]
colors = {"Micro": "crimson", "Small": "coral", "Medium": "steelblue"}
bar_colors = [colors.get(s, "gray") for s in q1["firm_size"]]
bars = ax1.bar(q1["firm_size"], q1["constrained_pct"], color=bar_colors, edgecolor="white")
ax1.set_title("Credit Constraint Rate by Firm Size")
ax1.set_ylabel("% Credit Constrained")
ax1.grid(True, alpha=0.3)
for bar, val in zip(bars, q1["constrained_pct"]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{val}%", ha="center", fontsize=11, fontweight="bold")

# --- Chart 2: Financial access funnel (Q3) ---
ax2 = axes[0, 1]
funnel_colors = ["steelblue", "coral", "seagreen"]
bars = ax2.bar(q3["stage"], q3["pct_of_firms"], color=funnel_colors, edgecolor="white")
ax2.set_title("Financial Access Funnel")
ax2.set_ylabel("% of Firms")
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3)
for bar, val in zip(bars, q3["pct_of_firms"]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{val}%", ha="center", fontsize=11, fontweight="bold")

# --- Chart 3: Constraint by sector (Q4) ---
ax3 = axes[1, 0]
ax3.barh(q4["sector"], q4["constrained_pct"], color="mediumpurple", edgecolor="white")
ax3.set_title("Credit Constraint Rate by Sector")
ax3.set_xlabel("% Credit Constrained")
ax3.grid(True, alpha=0.3)

# --- Chart 4: Data-sparse vs rich (Q5) ---
ax4 = axes[1, 1]
metrics = ["constrained_pct", "bank_account_pct"]
labels  = ["Constrained %", "Bank Account %"]
x = range(len(metrics))
width = 0.35
for i, (region, color) in enumerate(zip(q5["region_type"], ["crimson", "steelblue"])):
    vals = [q5.iloc[i][m] for m in metrics]
    ax4.bar([xi + i*width for xi in x], vals, width, label=region,
            color=color, edgecolor="white", alpha=0.85)
ax4.set_title("Data-Sparse vs Data-Rich Regions")
ax4.set_ylabel("%")
ax4.set_xticks([xi + width/2 for xi in x])
ax4.set_xticklabels(labels)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/msme_sql_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nSQL analysis complete. Chart saved to outputs/msme_sql_analysis.png")
conn.close()
