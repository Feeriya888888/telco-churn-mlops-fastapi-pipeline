import os
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from train import add_domain_features, train_pipeline

MODEL_PATH = "model_bundle.joblib"
model_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        train_pipeline()
    model_state["artifact"] = joblib.load(MODEL_PATH)
    yield
    model_state.clear()


app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Микросервис онлайн-скоринга вероятности оттока клиентов на базе LightGBM",
    version="1.0.0",
    lifespan=lifespan,
)


class CustomerFeatures(BaseModel):
    tenure_months: int = Field(..., ge=0, le=120, description="Срок жизни клиента в месяцах")
    monthly_charges: float = Field(..., gt=0, le=500, description="Ежемесячный платеж ($)")
    total_charges: float = Field(..., ge=0, description="Общая сумма платежей за все время ($)")
    contract_type: Literal["Month-to-month", "One year", "Two year"]
    internet_service: Literal["Fiber optic", "DSL", "No"]
    tech_support: Literal["Yes", "No", "No internet service"]
    payment_method: Literal["Electronic check", "Mailed check", "Bank transfer", "Credit card"]


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_segment: str
    threshold_used: float
    recommended_action: str


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": "artifact" in model_state,
        "validation_metrics": model_state.get("artifact", {}).get("metrics"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerFeatures):
    artifact = model_state.get("artifact")
    if not artifact:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    raw_df = pd.DataFrame([customer.model_dump()])
    featured_df = add_domain_features(raw_df)

    features = artifact["num_features"] + artifact["cat_features"]
    proba = float(artifact["pipeline"].predict_proba(featured_df[features])[0, 1])
    threshold = artifact["threshold"]
    is_churn = proba >= threshold

    if proba >= 0.70:
        risk = "HIGH_RISK"
        action = "Персональный звонок менеджера + скидка 20% на 3 месяца"
    elif is_churn:
        risk = "MEDIUM_RISK"
        action = "Автоматическое SMS/Push-предложение бесплатной техподдержки"
    else:
        risk = "LOW_RISK"
        action = "Стандартное обслуживание, удержание не требуется"

    return PredictionResponse(
        churn_probability=round(proba, 4),
        churn_prediction=is_churn,
        risk_segment=risk,
        threshold_used=threshold,
        recommended_action=action,
    )
