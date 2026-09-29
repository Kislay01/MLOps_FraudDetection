"""
FastAPI service for real-time fraud detection inference.
Loads the current 'champion' model from the MLflow Model Registry.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.xgboost
from mlflow import MlflowClient
import pandas as pd
from typing import Dict, Any
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "fraud-detection-model"
MODEL_ALIAS = "champion"

app = FastAPI(title="Fraudar Fraud Detection API", version="1.0")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

model = None
model_version = None
feature_names = None


class Transaction(BaseModel):
    data: Dict[str, Any]


@app.on_event("startup")
def load_champion_model():
    global model, model_version, feature_names
    model_info = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    model_version = model_info.version
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.xgboost.load_model(model_uri)
    feature_names = model.get_booster().feature_names
    logger.info(f"Loaded champion model version {model_version} with {len(feature_names)} features.")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_version": model_version,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.post("/predict")
def predict(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = {col: transaction.data.get(col, 0) for col in feature_names}
    df = pd.DataFrame([row])

    proba = float(model.predict_proba(df)[0, 1])
    prediction = int(proba >= 0.5)

    logger.info(f"Prediction made: proba={proba:.4f}, prediction={prediction}, model_version={model_version}")

    return {
        "fraud_probability": proba,
        "is_fraud": prediction,
        "model_version": model_version,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }