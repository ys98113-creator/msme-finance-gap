# ============================================================
# MSME FINANCE GAP — EXPORT DATA FOR TABLEAU
# Goal: Prepare clean, analysis-ready CSVs for Tableau dashboard
# ============================================================

import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("tableau_data", exist_ok=True)

df_firms = pd.read_csv("data/enterprise_survey_india.csv")
df_ifc   = pd.read_csv("data/ifc_msme_gap.csv")


# ─── Export 1: Firm-level data ───────────────────────────────
df_firms.to_csv("tableau_data/01_firms.csv", index=False)
print(f"Exported 01_firms.csv — {len(df_firms)} rows")


# ─── Export 2: State summary ─────────────────────────────────
state_summary = df_firms.groupby("state").agg(
    total_firms            = ("firm_id", "count"),
    constrained_pct        = ("credit_constrained", lambda x: round(x.mean()*100, 1)),
    bank_account_pct       = ("has_bank_account",   lambda x: round(x.mean()*100, 1)),
    credit_applied_pct     = ("applied_for_credit", lambda x: round(x.mean()*100, 1)),
    avg_credit_gap_usd     = ("credit_gap_usd",     lambda x: round(x.mean(), 0)),
    avg_revenue_usd        = ("annual_revenue_usd", lambda x: round(x.mean(), 0)),
    micro_pct              = ("firm_size",          lambda x: round((x=="Micro").mean()*100,1)),
    data_sparse            = ("data_sparse_region", "max")
).reset_index()

state_summary["region_type"] = state_summary["data_sparse"].apply(
    lambda x: "Data-Sparse" if x == 1 else "Data-Rich"
)

state_summary.to_csv("tableau_data/02_state_summary.csv", index=False)
print(f"Exported 02_state_summary.csv — {len(state_summary)} rows")


# ─── Export 3: Firm size summary ─────────────────────────────
size_summary = df_firms.groupby("firm_size", observed=True).agg(
    total_firms        = ("firm_id", "count"),
    constrained_pct    = ("credit_constrained", lambda x: round(x.mean()*100,1)),
    bank_account_pct   = ("has_bank_account",   lambda x: round(x.mean()*100,1)),
    avg_credit_gap_usd = ("credit_gap_usd",     lambda x: round(x.mean(),0)),
    avg_revenue_usd    = ("annual_revenue_usd", lambda x: round(x.mean(),0))
).reset_index()

size_summary.to_csv("tableau_data/03_firm_size_summary.csv", index=False)
print(f"Exported 03_firm_size_summary.csv — {len(size_summary)} rows")


# ─── Export 4: Sector summary ────────────────────────────────
sector_summary = df_firms.groupby("sector").agg(
    total_firms        = ("firm_id", "count"),
    constrained_pct    = ("credit_constrained", lambda x: round(x.mean()*100,1)),
    avg_credit_gap_usd = ("credit_gap_usd",     lambda x: round(x.mean(),0)),
    bank_account_pct   = ("has_bank_account",   lambda x: round(x.mean()*100,1))
).reset_index().sort_values("constrained_pct", ascending=False)

sector_summary.to_csv("tableau_data/04_sector_summary.csv", index=False)
print(f"Exported 04_sector_summary.csv — {len(sector_summary)} rows")


# ─── Export 5: IFC country trends ────────────────────────────
df_ifc["constrained_pct"] = (df_ifc["credit_constrained_pct"] * 100).round(1)
df_ifc["formal_msmes_pct_display"] = (df_ifc["formal_msmes_pct"] * 100).round(1)

df_ifc.to_csv("tableau_data/05_ifc_country_trends.csv", index=False)
print(f"Exported 05_ifc_country_trends.csv — {len(df_ifc)} rows")


# ─── Export 6: Access funnel ─────────────────────────────────
funnel = pd.DataFrame({
    "stage": ["Has Bank Account", "Applied for Credit", "Credit Approved"],
    "stage_order": [1, 2, 3],
    "pct_of_firms": [
        round(df_firms["has_bank_account"].mean()*100, 1),
        round(df_firms["applied_for_credit"].mean()*100, 1),
        round(df_firms["credit_approved"].mean()*100, 1)
    ]
})

funnel.to_csv("tableau_data/06_access_funnel.csv", index=False)
print(f"Exported 06_access_funnel.csv — {len(funnel)} rows")


print("\nAll Tableau data exports complete.")
print("Files saved to: /tableau_data/")
print("\nNext step: open Tableau Public and connect to these CSV files.")
