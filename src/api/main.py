"""
FastAPI service for real-time fraud detection inference.
Supports blue-green deployment: routes traffic between two model versions
loaded from the MLflow Model Registry, with configurable traffic split.
Exposes Prometheus metrics for monitoring.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import mlflow
import mlflow.xgboost
from mlflow import MlflowClient
import pandas as pd
from typing import Dict, Any
import logging
import random
import json
import os
import time
from datetime import datetime, timezone
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

FRAUD_THRESHOLD = 0.6210  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "fraud-detection-model"

AUDIT_LOG_PATH = "logs/predictions_audit.jsonl"
os.makedirs("logs", exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# Blue-green state: two slots, each optionally loaded with a model + its alias/version
deployment_state = {
    "blue": {"alias": "champion", "model": None, "version": None, "feature_names": None},
    "green": {"alias": None, "model": None, "version": None, "feature_names": None},
}

# Traffic split: fraction of requests routed to "green" (0.0 = all blue, 1.0 = all green)
traffic_split = {"green_weight": 0.0}

# Prometheus metrics
PREDICTION_COUNT = Counter(
    "fraud_predictions_total", "Total number of predictions made",
    ["served_by", "model_version", "predicted_class"]
)
PREDICTION_LATENCY = Histogram(
    "fraud_prediction_latency_seconds", "Time taken to make a prediction"
)
PREDICTION_CONFIDENCE = Histogram(
    "fraud_prediction_confidence", "Distribution of predicted fraud probabilities",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
GREEN_TRAFFIC_WEIGHT = Gauge(
    "fraud_green_traffic_weight", "Current traffic weight routed to green slot"
)


def _load_slot(slot_name: str, alias: str):
    model_info = client.get_model_version_by_alias(MODEL_NAME, alias)
    model_uri = f"models:/{MODEL_NAME}@{alias}"
    model = mlflow.xgboost.load_model(model_uri)
    deployment_state[slot_name] = {
        "alias": alias,
        "model": model,
        "version": model_info.version,
        "feature_names": model.get_booster().feature_names,
    }
    logger.info(f"Loaded '{alias}' into slot '{slot_name}': version {model_info.version}")


def log_prediction_audit(record: Dict[str, Any], proba: float, prediction: int, slot_name: str, version: str):
    audit_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "transaction_id": record.get("TransactionID", "unknown"),
        "fraud_probability": proba,
        "is_fraud": prediction,
        "served_by": slot_name,
        "model_version": version,
    }
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(audit_entry) + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_slot("blue", "champion")
    yield


app = FastAPI(title="Fraudar Fraud Detection API", version="2.0", lifespan=lifespan)


class Transaction(BaseModel):
    data: Dict[str, Any]


class DeploymentConfig(BaseModel):
    green_alias: str
    green_weight: float


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "blue_version": deployment_state["blue"]["version"],
        "green_version": deployment_state["green"]["version"],
        "green_weight": traffic_split["green_weight"],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/admin/deploy_green")
def deploy_green(config: DeploymentConfig):
    """Load a model into the 'green' slot and set traffic split, without touching 'blue'."""
    if not (0.0 <= config.green_weight <= 1.0):
        raise HTTPException(status_code=400, detail="green_weight must be between 0.0 and 1.0")

    _load_slot("green", config.green_alias)
    traffic_split["green_weight"] = config.green_weight

    return {
        "status": "deployed",
        "green_version": deployment_state["green"]["version"],
        "green_weight": traffic_split["green_weight"]
    }


@app.post("/admin/promote_green")
def promote_green():
    """Promote green to blue (full cutover), reset green slot."""
    if deployment_state["green"]["model"] is None:
        raise HTTPException(status_code=400, detail="No model deployed in green slot")

    deployment_state["blue"] = deployment_state["green"]
    deployment_state["green"] = {"alias": None, "model": None, "version": None, "feature_names": None}
    traffic_split["green_weight"] = 0.0

    return {"status": "promoted", "blue_version": deployment_state["blue"]["version"]}


def _predict_with_slot(slot, record: Dict[str, Any]):
    row = {col: record.get(col, 0) for col in slot["feature_names"]}
    df = pd.DataFrame([row])
    proba = float(slot["model"].predict_proba(df)[0, 1])
    prediction = int(proba >= FRAUD_THRESHOLD)
    return proba, prediction


@app.post("/predict")
def predict(transaction: Transaction):
    if deployment_state["blue"]["model"] is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    use_green = (
        deployment_state["green"]["model"] is not None
        and random.random() < traffic_split["green_weight"]
    )
    slot = deployment_state["green"] if use_green else deployment_state["blue"]
    slot_name = "green" if use_green else "blue"

    start_time = time.time()
    proba, prediction = _predict_with_slot(slot, transaction.data)
    latency = time.time() - start_time

    PREDICTION_COUNT.labels(served_by=slot_name, model_version=str(slot["version"]), predicted_class=str(prediction)).inc()
    PREDICTION_LATENCY.observe(latency)
    PREDICTION_CONFIDENCE.observe(proba)
    GREEN_TRAFFIC_WEIGHT.set(traffic_split["green_weight"])

    logger.info(
        f"Prediction: proba={proba:.4f}, prediction={prediction}, "
        f"slot={slot_name}, model_version={slot['version']}"
    )

    log_prediction_audit(transaction.data, proba, prediction, slot_name, slot["version"])

    return {
        "fraud_probability": proba,
        "is_fraud": prediction,
        "model_version": slot["version"],
        "served_by": slot_name,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }