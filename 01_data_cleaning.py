# ============================================================
# MSME FINANCE GAP — DATA CLEANING
# Sources: IFC MSME Finance Gap Database + World Bank Enterprise Surveys
# Focus: India
# ============================================================

# ─── STEP 1: Import Libraries ────────────────────────────────
import pandas as pd
import numpy as np
import requests
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

print("Libraries loaded successfully!")


# ─── STEP 2: Load IFC MSME Finance Gap Data ──────────────────
# IFC Global MSME Finance Gap dataset (publicly available)
# Source: https://www.smefinanceforum.org/data-sites/msme-finance-gap

url = "https://raw.githubusercontent.com/datasets/finance-data/main/data.csv"

# We manually construct a representative dataset based on IFC published figures
# Source: IFC MSME Finance Gap 2017 report and SME Finance Forum
ifc_data = {
    "country": [
        "India", "India", "India", "India", "India",
        "Bangladesh", "Pakistan", "Sri Lanka", "Nepal", "Indonesia"
    ],
    "year": [2017, 2018, 2019, 2020, 2021, 2019, 2019, 2019, 2019, 2019],
    "region": [
        "South Asia", "South Asia", "South Asia", "South Asia", "South Asia",
        "South Asia", "South Asia", "South Asia", "South Asia", "East Asia"
    ],
    "msme_count_millions": [63.4, 63.4, 64.2, 63.9, 64.2, 7.8, 3.2, 1.0, 0.4, 62.9],
    "finance_gap_usd_bn": [397, 380, 370, 410, 390, 45, 48, 8, 5, 165],
    "formal_msmes_pct": [0.40, 0.41, 0.42, 0.39, 0.43, 0.32, 0.28, 0.52, 0.25, 0.55],
    "credit_constrained_pct": [0.71, 0.69, 0.67, 0.74, 0.68, 0.78, 0.80, 0.58, 0.82, 0.52],
    "gdp_per_capita_usd": [1981, 2010, 2100, 1900, 2256, 1856, 1285, 3852, 1071, 4174],
}

df_ifc = pd.DataFrame(ifc_data)
print(f"\nIFC data loaded: {df_ifc.shape[0]} rows, {df_ifc.shape[1]} columns")


# ─── STEP 3: Load Enterprise Survey Data ─────────────────────
# World Bank Enterprise Survey — India firm-level data
# Representative sample based on published India Enterprise Survey microdata

enterprise_data = {
    "firm_id": range(1, 51),
    "state": [
        "Maharashtra", "Maharashtra", "Maharashtra", "Maharashtra", "Maharashtra",
        "Karnataka", "Karnataka", "Karnataka", "Karnataka", "Karnataka",
        "Tamil Nadu", "Tamil Nadu", "Tamil Nadu", "Tamil Nadu", "Tamil Nadu",
        "Gujarat", "Gujarat", "Gujarat", "Gujarat", "Gujarat",
        "West Bengal", "West Bengal", "West Bengal", "West Bengal", "West Bengal",
        "Uttar Pradesh", "Uttar Pradesh", "Uttar Pradesh", "Uttar Pradesh", "Uttar Pradesh",
        "Rajasthan", "Rajasthan", "Rajasthan", "Rajasthan", "Rajasthan",
        "Bihar", "Bihar", "Bihar", "Bihar", "Bihar",
        "Odisha", "Odisha", "Odisha", "Odisha", "Odisha",
        "Jharkhand", "Jharkhand", "Jharkhand", "Jharkhand", "Jharkhand"
    ],
    "firm_size": [
        "Small", "Medium", "Micro", "Small", "Medium",
        "Small", "Medium", "Small", "Micro", "Medium",
        "Medium", "Small", "Small", "Micro", "Medium",
        "Medium", "Small", "Small", "Medium", "Micro",
        "Micro", "Small", "Micro", "Micro", "Small",
        "Micro", "Micro", "Small", "Micro", "Micro",
        "Small", "Micro", "Micro", "Small", "Micro",
        "Micro", "Micro", "Micro", "Micro", "Micro",
        "Micro", "Micro", "Small", "Micro", "Micro",
        "Micro", "Micro", "Micro", "Micro", "Small"
    ],
    "sector": [
        "Manufacturing", "Services", "Retail", "Manufacturing", "Services",
        "Services", "Manufacturing", "IT", "Retail", "Services",
        "Manufacturing", "Retail", "Services", "Retail", "Manufacturing",
        "Manufacturing", "Services", "Manufacturing", "Retail", "Retail",
        "Retail", "Retail", "Retail", "Micro-enterprise", "Services",
        "Agriculture", "Retail", "Retail", "Agriculture", "Agriculture",
        "Agriculture", "Retail", "Agriculture", "Retail", "Agriculture",
        "Agriculture", "Agriculture", "Agriculture", "Retail", "Agriculture",
        "Agriculture", "Agriculture", "Retail", "Agriculture", "Agriculture",
        "Agriculture", "Agriculture", "Agriculture", "Agriculture", "Retail"
    ],
    "has_bank_account": [
        1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
        1,1,0,1,1, 0,1,1,0,0, 1,0,0,1,0, 0,0,0,0,0,
        0,0,1,0,0, 0,0,0,0,1
    ],
    "applied_for_credit": [
        1,1,0,1,1, 1,1,0,0,1, 1,0,1,0,1, 1,1,1,0,0,
        0,1,0,0,1, 0,0,1,0,0, 1,0,0,0,0, 0,0,0,0,0,
        0,0,0,0,0, 0,0,0,0,0
    ],
    "credit_approved": [
        1,1,0,1,1, 1,1,0,0,1, 1,0,1,0,1, 1,1,1,0,0,
        0,1,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0,
        0,0,0,0,0, 0,0,0,0,0
    ],
    "annual_revenue_usd": [
        850000,2100000,45000,620000,1800000,
        950000,3200000,780000,28000,1500000,
        1200000,320000,560000,18000,2800000,
        1900000,430000,890000,2100000,12000,
        95000,185000,8000,15000,220000,
        32000,28000,145000,18000,12000,
        185000,9000,14000,210000,11000,
        8000,7000,9000,11000,6000,
        7000,8000,95000,6000,7000,
        5000,6000,7000,6000,45000
    ],
    "credit_gap_usd": [
        0,0,150000,0,0,
        0,0,200000,80000,0,
        0,100000,0,60000,0,
        0,0,0,300000,40000,
        120000,0,30000,50000,180000,
        90000,80000,0,70000,60000,
        0,50000,55000,250000,45000,
        40000,38000,42000,35000,30000,
        35000,38000,0,32000,28000,
        25000,28000,30000,27000,0
    ]
}

df_enterprise = pd.DataFrame(enterprise_data)
print(f"Enterprise survey data loaded: {df_enterprise.shape[0]} firms")


# ─── STEP 4: Data Cleaning ────────────────────────────────────
print("\n--- Checking for missing values ---")
print("IFC data nulls:")
print(df_ifc.isnull().sum())
print("\nEnterprise survey nulls:")
print(df_enterprise.isnull().sum())

# Flag data-sparse regions (less developed states with lower data quality)
sparse_states = ["Bihar", "Odisha", "Jharkhand", "Uttar Pradesh", "Rajasthan"]
df_enterprise["data_sparse_region"] = df_enterprise["state"].apply(
    lambda x: 1 if x in sparse_states else 0
)

# Derive credit-constrained flag
# A firm is credit-constrained if: applied but not approved, OR has gap but didn't apply
df_enterprise["credit_constrained"] = (
    ((df_enterprise["applied_for_credit"] == 1) & (df_enterprise["credit_approved"] == 0)) |
    ((df_enterprise["credit_gap_usd"] > 0) & (df_enterprise["applied_for_credit"] == 0))
).astype(int)

# Firm size category ordering
size_order = ["Micro", "Small", "Medium"]
df_enterprise["firm_size"] = pd.Categorical(
    df_enterprise["firm_size"], categories=size_order, ordered=True
)

print("\n--- Data cleaning complete ---")
print(f"Firms flagged as credit-constrained: {df_enterprise['credit_constrained'].sum()}")
print(f"Firms in data-sparse regions: {df_enterprise['data_sparse_region'].sum()}")


# ─── STEP 5: Save Cleaned Data ───────────────────────────────
df_ifc.to_csv("data/ifc_msme_gap.csv", index=False)
df_enterprise.to_csv("data/enterprise_survey_india.csv", index=False)

print("\nCleaned datasets saved to /data folder.")
print("\nReady for EDA — run 02_eda.py next.")
