# 📊 Telco Customer Churn Prediction & Analysis

End-to-end churn prediction project on the IBM Telco dataset: exploratory analysis to understand *why* customers leave, followed by model building to predict *who* will leave next with explainability and deployment.

---

## 📁 Project Structure
```
├── customer_churn_eda.ipynb          # EDA notebook
├── model_building.ipynb              # Model training, tuning, SHAP
├── RawData_Telco-Customer-Churn.csv  # Raw dataset (original)
├── EDA_telco_churn.csv               # Cleaned & encoded dataset (post EDA)
├── churn_model_lightgbm.pkl          # Final trained model
└── README.md
```

---

## 📦 Dataset
- **Source:** IBM Sample Telco Customer Churn Dataset
- **Rows:** 7,043 customers (7,032 after cleaning)
- **Columns:** 21 raw features
- **Target Variable:** `Churn` (Yes / No)

| Feature Type | Columns |
|---|---|
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Account Info | tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges |
| Services | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |

---

## 🔧 Tech Stack
```
numpy, pandas, matplotlib, seaborn
scikit-learn, lightgbm, imbalanced-learn (SMOTEENN)
optuna (hyperparameter tuning)
shap (explainability)
joblib (model persistence)
```

---

## Part 1: Exploratory Data Analysis

### 🧹 Data Cleaning
- `TotalCharges` was stored as string converted to float using `pd.to_numeric`
- 11 rows with missing `TotalCharges` dropped (< 0.2% of data)
- Outlier check (IQR method) on `MonthlyCharges` and `TotalCharges` none found, consistent with fixed telecom pricing tiers
- Working copy created with `df.copy()` to preserve the original

### 🛠️ Feature Engineering
- **Tenure Grouping:** customers grouped into 12-month bins (`1-12`, `13-24`, ... `61-72`) for easier pattern analysis
- `customerID` and raw `tenure` dropped post-encoding (ID has no predictive value; `tenure_group` replaces raw `tenure` to avoid redundancy)

### 📊 Analysis Performed
- **Univariate:** churn distribution (73% No / 27% Yes — imbalanced), count plots for all categorical features split by churn
- **Bivariate:** KDE plots for MonthlyCharges/TotalCharges by churn, correlation bar chart of all features vs. Churn, gender breakdown within churned customers
- **Multivariate:** correlation heatmap, Contract × InternetService churn-rate heatmap

### 🔑 Key Insights
- **Contract type** is the strongest churn driver month-to-month customers churn at ~43% vs. ~3% for two-year contracts
- **New customers are most vulnerable** ~48% of churners leave within the first year; churn drops to ~6.6% by year 5–6
- **Fiber optic users** churn at ~42% despite being premium customers
- **No security/support add-ons** → ~42% churn vs. ~14% with these services
- **Electronic check** users churn ~3x more than auto-pay customers
- **Gender, phone service, streaming** have near-zero impact on churn

### 🚨 High-Risk Customer Profile & Business Impact
Customers matching **all four** of these conditions churn at a disproportionately high rate:

| Condition | Value |
|---|---|
| Contract | Month-to-month |
| Internet Service | Fiber optic |
| Payment Method | Electronic check |
| Tenure Group | 1–12 months |

- Monthly revenue lost to churn: **~$139K**
- Revenue at risk from the high-risk segment alone: **~$52K/month**
- Retaining just 20% of churners would save **~$28K/month**

This segment is a small share of the customer base but accounts for a disproportionate share of churn the primary target for any retention campaign.

---

## Part 2: Model Building

### ⚠️ Handling Class Imbalance (73:27)
Two strategies were compared:
- **SMOTEENN** — resamples the training set only (synthetic minority oversampling + noisy-sample cleanup); test set left untouched to avoid leakage
- **scale_pos_weight** (LightGBM) leaves data unchanged, reweights the loss function instead

### 🤖 Models Compared

| Model | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|---|---|---|---|
| Decision Tree (baseline) | 0.48 | 0.49 | 0.49 |
| Decision Tree + SMOTEENN | 0.50 | 0.70 | 0.58 |
| Random Forest (class_weight=balanced) | 0.60 | 0.49 | 0.54 |
| Random Forest + SMOTEENN | 0.54 | 0.74 | 0.62 |
| LightGBM (scale_pos_weight) | 0.52 | 0.76 | 0.62 |
| **LightGBM (Optuna-tuned)** | **0.56** | **0.77** | **0.65** |

**Final model:** Optuna-tuned LightGBM (threshold = 0.5) best overall balance of precision, recall, and F1 for the minority (churn) class, with the added benefit of native imbalance handling (no synthetic data needed).

### 🎯 Hyperparameter Tuning
Tuned via **Optuna** (50 trials, TPE sampler) optimizing for F1-score on the churn class — `n_estimators`, `max_depth`, `learning_rate`, `num_leaves`, `min_child_samples`.

### 🔍 Explainability — SHAP
- **Global importance:** `Contract_Two year`, `Contract_One year`, `InternetService_Fiber optic`, `TotalCharges`, and `MonthlyCharges` are the most influential features
- **Direction of effect:** longer contracts reduce churn probability; fiber optic internet and higher monthly charges increase it
- **Local explanation:** waterfall plots break down individual predictions (e.g., a customer with 0.83 predicted churn probability driven by low tenure/TotalCharges, no long-term contract, and fiber optic + electronic check)

### 💾 Model Persistence
Final model saved with `joblib` and reload-verified to produce identical predictions ready for serving.

---

## 📌 Final Conclusion
Contract type, internet service type, payment method, and customer tenure are the dominant drivers of churn — both statistically (correlation, SHAP) and in raw churn-rate terms. The tuned LightGBM model catches ~77% of churners at a workable false-positive cost, and SHAP makes every prediction explainable enough to hand to a non-technical retention team.

**Next steps:** cross-validated hyperparameter tuning, a logistic regression baseline for comparison, and monitoring model drift post-deployment.
