# 📊 Telco Customer Churn 

Exploratory analysis of a telecom dataset to uncover why customers leave. Identifies high-risk segments through univariate, bivariate, and multivariate analysis, culminating in a data-backed high-risk customer profile that sets the foundation for churn prediction modeling.

---

## 📁 Project Structure

```
├── customer_churn_eda.ipynb          # Main EDA notebook
├── RawData_Telco-Customer-Churn.csv  # Raw dataset (original)
├── EDA_telco_churn.csv               # Cleaned & encoded dataset (post EDA)
└── README.md

```

---

## 📦 Dataset

- **Source:** IBM Sample Telco Customer Churn Dataset
- **Rows:** 7,043 customers
- **Columns:** 21 features
- **Target Variable:** `Churn` (Yes / No)

| Feature Type | Columns |
|---|---|
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Account Info | tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges |
| Services | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |

---

## 🔧 Libraries Used

```python
numpy
pandas
matplotlib
seaborn
```

---

## 🧹 Data Cleaning

- `TotalCharges` was stored as string — converted to float using `pd.to_numeric`
- 11 rows with missing `TotalCharges` dropped (< 0.2% of data)
- Created a clean working copy using `df.copy()` to preserve the original

---

## 🛠️ Feature Engineering

- **Tenure Grouping:** Customers grouped into 12-month bins (`1-12`, `13-24`, ... `61-72`) for easier pattern analysis
- **Charge Segmentation:** `MonthlyCharges` split into Low / Medium / High segments
- **Add-on Count:** New feature counting how many add-on services each customer has (0–6)

---

## 📊 Analysis Performed

### Univariate Analysis
- Churn distribution (73% No, 27% Yes — imbalanced dataset)
- Count plots for all categorical features split by churn
- Churn rate % bar charts per feature category

### Bivariate Analysis
- KDE plots for MonthlyCharges and TotalCharges by churn
- Boxplots for tenure, MonthlyCharges, TotalCharges by churn
- Correlation bar chart of all features with Churn
- Cross-feature analysis for churned customers (gender breakdown)

### Multivariate Analysis
- Top 10 features correlation heatmap

---

## 🔑 Key Insights

- **Contract type** is the strongest churn driver — month-to-month customers churn at ~43% vs 3% for 2-year contracts
- **New customers are most vulnerable** — ~48% of churners leave within the first year
- **Fiber optic users** churn at 42% despite being premium customers
- **No security/support add-ons** = ~42% churn rate vs 14% with these services
- **Electronic check** users churn 3x more than auto-pay customers
- **Gender, phone service, streaming** have near-zero impact on churn

---

## 🚨 High-Risk Customer Profile

Customers matching **all four** of these conditions churn at an extremely high rate:

| Condition | Value |
|---|---|
| Contract | Month-to-month |
| Internet Service | Fiber optic |
| Payment Method | Electronic check |
| Tenure Group | 1–12 months |

> This segment makes up a small % of total customers but accounts for a disproportionate share of churn — the primary target for any retention campaign.

---

## ⚠️ Class Imbalance Note

The dataset is imbalanced at **73:27 (No Churn : Churn)**. Accuracy alone is a misleading metric here. The modeling phase will use techniques like **SMOTEENN** and evaluate using **Recall / F1-score** instead.

