"""
Telco Customer Churn Prediction API
------------------------------------
Serves the Optuna-tuned LightGBM model (threshold = 0.5) trained on the
IBM Telco Customer Churn dataset, with SHAP-based per-prediction explanations.
"""

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "churn_model_lightgbm.pkl"
FRONTEND_DIR = BASE_DIR.parent / "frontend"

THRESHOLD = 0.5  # chosen as final in the notebook comparison

app = FastAPI(title="Telco Churn Prediction API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load(MODEL_PATH)
FEATURE_ORDER = list(model.feature_name_)
EXPLAINER = shap.TreeExplainer(model)
_expected_value = np.ravel(EXPLAINER.expected_value)[0]
BASE_PROBABILITY = float(1 / (1 + np.exp(-_expected_value)))  # avg model output, sigmoid of log-odds base

# Each one-hot column gets a (label when =1, label when =0) pair, so a SHAP
# contribution always describes what's actually true about this customer,
# not just the raw column name.
FEATURE_LABELS = {
    "Partner_Yes": ("Has a partner", "No partner"),
    "Dependents_Yes": ("Has dependents", "No dependents"),
    "MultipleLines_Yes": ("Multiple phone lines", "Single phone line"),
    "InternetService_Fiber_optic": ("Fiber optic internet", "Not on fiber internet"),
    "InternetService_No": ("No internet service", "Has internet service"),
    "OnlineSecurity_No_internet_service": ("No internet service", "Has internet service"),
    "OnlineSecurity_Yes": ("Has online security", "No online security"),
    "OnlineBackup_No_internet_service": ("No internet service", "Has internet service"),
    "OnlineBackup_Yes": ("Has online backup", "No online backup"),
    "DeviceProtection_No_internet_service": ("No internet service", "Has internet service"),
    "DeviceProtection_Yes": ("Has device protection", "No device protection"),
    "TechSupport_No_internet_service": ("No internet service", "Has internet service"),
    "TechSupport_Yes": ("Has tech support", "No tech support"),
    "StreamingTV_No_internet_service": ("No internet service", "Has internet service"),
    "StreamingTV_Yes": ("Streams TV", "Doesn't stream TV"),
    "StreamingMovies_No_internet_service": ("No internet service", "Has internet service"),
    "StreamingMovies_Yes": ("Streams movies", "Doesn't stream movies"),
    "Contract_One_year": ("One-year contract", "Not on a one-year contract"),
    "Contract_Two_year": ("Two-year contract", "Not on a two-year contract"),
    "PaperlessBilling_Yes": ("Paperless billing", "Paper billing"),
    "PaymentMethod_Credit_card_(automatic)": ("Pays by credit card (auto)", "Doesn't pay by credit card"),
    "PaymentMethod_Electronic_check": ("Pays by electronic check", "Doesn't pay by electronic check"),
    "PaymentMethod_Mailed_check": ("Pays by mailed check", "Doesn't pay by mailed check"),
    "tenure_group_13_-_24": ("Tenure: 13-24 months", "Tenure not 13-24 months"),
    "tenure_group_25_-_36": ("Tenure: 25-36 months", "Tenure not 25-36 months"),
    "tenure_group_37_-_48": ("Tenure: 37-48 months", "Tenure not 37-48 months"),
    "tenure_group_49_-_60": ("Tenure: 49-60 months", "Tenure not 49-60 months"),
    "tenure_group_61_-_72": ("Tenure: 61-72 months", "Tenure not 61-72 months"),
}


def _describe_feature(name: str, value: float) -> str:
    if name == "SeniorCitizen":
        return "Senior citizen" if value == 1 else "Not a senior citizen"
    if name == "MonthlyCharges":
        return f"Monthly charges (${value:.2f})"
    if name == "TotalCharges":
        return f"Total charges (${value:.2f})"
    if name in FEATURE_LABELS:
        pos, neg = FEATURE_LABELS[name]
        return pos if value == 1 else neg
    return name


# ---------------------------------------------------------------------------
# Request schema — exactly the 19 raw fields a real customer record has.
# Everything else (one-hot columns, tenure_group, dropped columns) is
# derived server-side so the frontend never needs to know the model's
# internal feature representation.
# ---------------------------------------------------------------------------
class CustomerInput(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 5,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.4,
                "TotalCharges": 420.5,
            }
        }


class FeatureContribution(BaseModel):
    feature: str
    contribution: float  # signed SHAP value (log-odds); positive pushes toward churn
    direction: Literal["increases", "decreases"]


class PredictionResponse(BaseModel):
    churn_prediction: Literal["Churn", "No Churn"]
    churn_probability: float
    base_probability: float  # the model's average prediction across all customers, for context
    risk_level: Literal["Low", "Medium", "High"]
    threshold_used: float
    top_factors: list[FeatureContribution]


def _tenure_group(tenure: int) -> str | None:
    """Reproduces pd.cut(bins=range(1, 80, 12), right=False) from the EDA notebook."""
    if tenure < 1:
        return None  # falls in the 1-12 baseline bucket (dropped by drop_first)
    edges = list(range(1, 80, 12))  # [1, 13, 25, 37, 49, 61, 73]
    labels = [f"{i}_-_{i+11}" for i in edges[:-1]]
    for i in range(len(edges) - 1):
        if edges[i] <= tenure < edges[i + 1]:
            return labels[i] if i > 0 else None  # first bucket (1-12) is baseline
    return labels[-1] if tenure >= edges[-1] else None


def build_feature_row(c: CustomerInput) -> pd.DataFrame:
    """Recreates the exact one-hot-encoded, drop_first=True feature matrix
    the model was trained on (see New_Model__Building.ipynb / EDA_Analysis.ipynb)."""

    row = {name: 0 for name in FEATURE_ORDER}

    row["SeniorCitizen"] = c.SeniorCitizen
    row["MonthlyCharges"] = c.MonthlyCharges
    row["TotalCharges"] = c.TotalCharges

    def set_if_present(col: str):
        if col in row:
            row[col] = 1

    if c.Partner == "Yes":
        set_if_present("Partner_Yes")
    if c.Dependents == "Yes":
        set_if_present("Dependents_Yes")
    if c.MultipleLines == "Yes":
        set_if_present("MultipleLines_Yes")
    # NOTE: gender_Male, PhoneService_Yes, MultipleLines_No_phone_service were
    # dropped from the final model (near-zero correlation with churn).

    if c.InternetService == "Fiber optic":
        set_if_present("InternetService_Fiber_optic")
    elif c.InternetService == "No":
        set_if_present("InternetService_No")

    for base, val in [
        ("OnlineSecurity", c.OnlineSecurity),
        ("OnlineBackup", c.OnlineBackup),
        ("DeviceProtection", c.DeviceProtection),
        ("TechSupport", c.TechSupport),
        ("StreamingTV", c.StreamingTV),
        ("StreamingMovies", c.StreamingMovies),
    ]:
        if val == "Yes":
            set_if_present(f"{base}_Yes")
        elif val == "No internet service":
            set_if_present(f"{base}_No_internet_service")

    if c.Contract == "One year":
        set_if_present("Contract_One_year")
    elif c.Contract == "Two year":
        set_if_present("Contract_Two_year")

    if c.PaperlessBilling == "Yes":
        set_if_present("PaperlessBilling_Yes")

    if c.PaymentMethod == "Credit card (automatic)":
        set_if_present("PaymentMethod_Credit_card_(automatic)")
    elif c.PaymentMethod == "Electronic check":
        set_if_present("PaymentMethod_Electronic_check")
    elif c.PaymentMethod == "Mailed check":
        set_if_present("PaymentMethod_Mailed_check")
    # Bank transfer (automatic) is the drop_first baseline -> all zeros

    tg = _tenure_group(c.tenure)
    if tg:
        set_if_present(f"tenure_group_{tg}")

    return pd.DataFrame([row], columns=FEATURE_ORDER)


@app.get("/api/health")
def health():
    return {"status": "ok", "model_features": len(FEATURE_ORDER)}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    try:
        X = build_feature_row(customer)
        proba = float(model.predict_proba(X)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    pred = "Churn" if proba >= THRESHOLD else "No Churn"
    risk = "High" if proba >= 0.6 else "Medium" if proba >= 0.35 else "Low"

    shap_values = EXPLAINER.shap_values(X)[0]  # log-odds contribution per feature, this row
    row_values = X.iloc[0]
    ranked = sorted(
        zip(FEATURE_ORDER, shap_values), key=lambda t: abs(t[1]), reverse=True
    )[:8]
    top_factors = [
        FeatureContribution(
            feature=_describe_feature(name, row_values[name]),
            contribution=round(float(val), 4),
            direction="increases" if val > 0 else "decreases",
        )
        for name, val in ranked
    ]

    return PredictionResponse(
        churn_prediction=pred,
        churn_probability=round(proba, 4),
        base_probability=round(BASE_PROBABILITY, 4),
        risk_level=risk,
        threshold_used=THRESHOLD,
        top_factors=top_factors,
    )


# ---------------------------------------------------------------------------
# Serve the frontend (single-page app) as static files, so the whole project
# can be deployed as one service.
# ---------------------------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(FRONTEND_DIR / "index.html")
