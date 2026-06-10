# MSME Finance Gap — India Analysis

A Python-based analytical study of the financing gap facing micro, small, and medium enterprises (MSMEs) in India, built on IFC MSME Finance Gap data and World Bank Enterprise Survey microdata.

This project replicates and extends research conducted at the **International Finance Corporation (IFC), World Bank Group**, applying machine learning techniques to identify which firms and regions are most financially excluded — and why.

---

## Research Question

> *"What drives financial exclusion for MSMEs in India, and can we predict which firms and regions are most at risk?"*

---

## Data Sources

| Dataset | Description |
|---------|-------------|
| IFC MSME Finance Gap Database | Country-level unmet credit demand estimates |
| World Bank Enterprise Survey (India) | Firm-level access to finance microdata |

**Key challenge:** Uneven geographic coverage — rural and lower-income Indian states had significantly thinner data, requiring a timeliness-vs-completeness tradeoff in the analysis.

---

## Project Structure

```
msme-finance-gap/
├── 01_data_cleaning.py      # Load, harmonize, and clean IFC + Enterprise Survey data
├── 02_eda.py                # Exploratory analysis by firm size, sector, and state
├── 03_clustering.py         # K-Means: segment states by finance gap profile
├── 04_classification.py     # Random Forest: predict firm-level credit constraints
├── 05_gap_prediction.py     # XGBoost: predict country-level finance gap size
├── 06_sql_analysis.py       # SQLite: 8 policy queries across firms and IFC tables
├── data/                    # Cleaned datasets
└── outputs/                 # All charts and visualizations
```

---

## Methodology

### 1. Data Cleaning & EDA
- Harmonized IFC aggregate estimates with field-level Enterprise Survey data
- Engineered credit-constraint flag combining application rejection and unmet demand
- Identified data-sparse regions and documented coverage limitations

### 2. K-Means Clustering
- Aggregated firm-level data to state level (credit constraint rate, bank access, firm size mix)
- Standardized features and used elbow method + silhouette scores to select K=3
- Output: **High Risk / Moderate Risk / Low Risk** state segments

### 3. Random Forest Classification
- Predicted firm-level credit-constraint probability from firm size, sector, state, revenue, and bank account status
- Used class balancing to handle imbalanced labels
- Output: ROC-AUC score, feature importance, risk scores per firm

### 4. XGBoost Gap Prediction
- Predicted country-level finance gap (USD billions) from macro indicators
- Features: credit constraint rate, formal MSME share, GDP per capita, MSME count
- Output: Predicted vs actual gap, feature importance, India trend analysis

---

## Key Findings

- **Micro enterprises** face the highest credit constraint rates (80%+), far exceeding small and medium firms
- **Data-sparse states** (Bihar, Odisha, Jharkhand, Uttar Pradesh, Rajasthan) show both higher constraint rates and lower analytical confidence — a compounding disadvantage
- **Bank account access** is the strongest predictor of whether a firm applies for credit — the access funnel breaks at the first step
- **Credit constraint rate** is the dominant predictor of country-level finance gap size, followed by the share of formal MSMEs

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

# Run in order
python 01_data_cleaning.py
python 02_eda.py
python 03_clustering.py
python 04_classification.py
python 05_gap_prediction.py
python 06_sql_analysis.py
```

Charts are saved to the `outputs/` folder.

---

## Background

This analysis is based on research conducted at the IFC, World Bank Group, where the author worked on quantifying India's MSME finance gap using IFC data and Enterprise Survey microdata collected directly from field officers across Indian states.

---

### 5. SQL Analysis (SQLite)
- Loaded cleaned datasets into an in-memory SQLite database
- Wrote 8 policy-focused queries across two tables (firms, ifc_gap)
- Questions answered: constraint rate by size/sector/state, access funnel drop-off, data-sparse vs rich regions, gap per MSME by country

---

*Built with Python | pandas · matplotlib · seaborn · scikit-learn · xgboost · sqlite3*
