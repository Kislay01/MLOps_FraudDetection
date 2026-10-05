"""
FastAPI service for real-time fraud detection with multi-model routing.

Holds several named model slots loaded from the MLflow Model Registry
(normal-traffic champion, three drift specialists, one universal model).
Exactly one slot is active; an optional canary slot can take a fraction of traffic.
Routing is changed live via /admin/route and /admin/rollback.
Exposes Prometheus metrics, including per-model labeled confusion counts.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import json
import logging
import os
import random
import time

import mlflow
import mlflow.xgboost
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from mlflow import MlflowClient
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel

FRAUD_THRESHOLD = 0.6210  # fallback if a model version has no threshold tag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "http://localhost:5000"
DEFAULT_MODEL = "fraud-detection-model"
SLOT_ALIASES = {
    "fraud-detection-model": "champion",
    "fraud-spec-A": "production",
    "fraud-spec-B": "production",
    "fraud-spec-C": "production",
    "fraud-universal": "production",
}
AUDIT_FEATURES = [
    "TransactionAmt", "txn_1h_count", "time_since_last_txn",
    "P_emaildomain_freq", "DeviceType_freq", "ProductCD_C", "C1", "C2",
]

AUDIT_LOG_PATH = "logs/predictions_audit.jsonl"
os.makedirs("logs", exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

slots: Dict[str, Dict[str, Any]] = {}
state: Dict[str, Any] = {
    "active": DEFAULT_MODEL,
    "previous": None,
    "canary": None,
    "canary_weight": 0.0,
}
history: list = []

# Prometheus metrics
PREDICTION_COUNT = Counter(
    "fraud_predictions_total", "Total number of predictions made",
    ["served_by", "model_version", "predicted_class"],
)
PREDICTION_LATENCY = Histogram("fraud_prediction_latency_seconds", "Time taken to make a prediction")
PREDICTION_CONFIDENCE = Histogram(
    "fraud_prediction_confidence", "Distribution of predicted fraud probabilities",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
LABELED_OUTCOMES = Counter(
    "fraud_labeled_outcomes_total", "Predictions with a known label, by confusion-matrix outcome",
    ["served_by", "outcome"],
)
ACTIVE_MODEL = Gauge("fraud_active_model", "1 if the model is the active route, else 0", ["model"])
CANARY_WEIGHT = Gauge("fraud_canary_weight", "Fraction of traffic sent to the canary model")
ROUTE_SWITCHES = Counter("fraud_route_switches_total", "Active-model switches", ["from_model", "to_model"])


def _now():
    return datetime.now(timezone.utc).isoformat()


def _load_slot(name: str):
    alias = SLOT_ALIASES[name]
    mv = client.get_model_version_by_alias(name, alias)
    model = mlflow.xgboost.load_model(f"models:/{name}@{alias}")
    tags = mv.tags or {}
    slots[name] = {
        "model": model,
        "version": str(mv.version),
        "alias": alias,
        "threshold": float(tags.get("threshold", FRAUD_THRESHOLD)),
        "kind": tags.get("drift_type", tags.get("role", "normal")),
        "feature_names": model.get_booster().feature_names,
    }
    logger.info(f"Loaded {name}@{alias}: v{mv.version}, threshold {slots[name]['threshold']}")


def _publish_state():
    for name in SLOT_ALIASES:
        ACTIVE_MODEL.labels(model=name).set(1 if name == state["active"] else 0)
    CANARY_WEIGHT.set(state["canary_weight"] if state["canary"] else 0.0)


def _record(event: str, model: str, reason: str, weight: float = 1.0):
    history.append({"timestamp": _now(), "event": event, "model": model, "weight": weight, "reason": reason})
    del history[:-200]


def _cutover(model: str, reason: str):
    old = state["active"]
    if model != old:
        state["previous"] = old
        state["active"] = model
        ROUTE_SWITCHES.labels(from_model=old, to_model=model).inc()
    state["canary"], state["canary_weight"] = None, 0.0
    _record("cutover", model, reason)
    _publish_state()


def log_prediction_audit(record, proba, prediction, slot_name, slot, label):
    entry = {
        "timestamp": _now(),
        "transaction_id": record.get("TransactionID", "unknown"),
        "fraud_probability": proba,
        "is_fraud": prediction,
        "threshold": slot["threshold"],
        "served_by": slot_name,
        "model_version": slot["version"],
        "label": label,
        "features": {k: record.get(k) for k in AUDIT_FEATURES},
    }
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    for name in SLOT_ALIASES:
        try:
            _load_slot(name)
        except Exception as e:
            logger.warning(f"Could not load slot {name}: {e}")
    if DEFAULT_MODEL not in slots:
        raise RuntimeError(f"Default model {DEFAULT_MODEL} could not be loaded")
    _publish_state()
    _record("startup", state["active"], "service start")
    yield


app = FastAPI(title="Fraudar Fraud Detection API", version="3.0", lifespan=lifespan)


class Transaction(BaseModel):
    data: Dict[str, Any]
    label: Optional[int] = None


class RouteRequest(BaseModel):
    model: str
    weight: float = 1.0
    reason: str = "manual"


class ReloadRequest(BaseModel):
    model: str


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "active": state["active"],
        "canary": state["canary"],
        "canary_weight": state["canary_weight"],
        "loaded_models": sorted(slots),
        "timestamp": _now(),
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/admin/slots")
def get_slots():
    return {
        "active": state["active"],
        "previous": state["previous"],
        "canary": state["canary"],
        "canary_weight": state["canary_weight"],
        "slots": {n: {k: s[k] for k in ("version", "alias", "threshold", "kind")} for n, s in slots.items()},
        "history": history[-20:],
    }


@app.post("/admin/route")
def route(req: RouteRequest):
    """weight>=1: full cutover. 0<weight<1: canary share. weight<=0: clear canary."""
    if req.model not in slots:
        raise HTTPException(status_code=400, detail=f"Model {req.model} not loaded")
    if req.weight >= 1.0:
        _cutover(req.model, req.reason)
    elif req.weight > 0.0:
        if req.model == state["active"]:
            raise HTTPException(status_code=400, detail="Canary must differ from the active model")
        state["canary"], state["canary_weight"] = req.model, req.weight
        _record("canary", req.model, req.reason, req.weight)
        _publish_state()
    else:
        state["canary"], state["canary_weight"] = None, 0.0
        _record("canary_cleared", state["active"], req.reason)
        _publish_state()
    return {"active": state["active"], "canary": state["canary"], "canary_weight": state["canary_weight"]}


@app.post("/admin/rollback")
def rollback():
    if not state["previous"]:
        raise HTTPException(status_code=400, detail="No previous model to roll back to")
    _cutover(state["previous"], "rollback")
    return {"active": state["active"], "previous": state["previous"]}


@app.post("/admin/reload")
def reload_slot(req: ReloadRequest):
    """Reload a slot from the registry (e.g. after a new version gets the alias)."""
    if req.model not in SLOT_ALIASES:
        raise HTTPException(status_code=400, detail=f"Unknown model {req.model}")
    _load_slot(req.model)
    _record("reload", req.model, "registry reload")
    return {"model": req.model, "version": slots[req.model]["version"]}


def _predict_with_slot(slot, record: Dict[str, Any]):
    row = {col: record.get(col, 0) for col in slot["feature_names"]}
    df = pd.DataFrame([row])
    proba = float(slot["model"].predict_proba(df)[0, 1])
    return proba, int(proba >= slot["threshold"])


@app.post("/predict")
def predict(transaction: Transaction):
    name = state["active"]
    if state["canary"] and random.random() < state["canary_weight"]:
        name = state["canary"]
    slot = slots[name]

    start = time.time()
    proba, prediction = _predict_with_slot(slot, transaction.data)
    latency = time.time() - start

    PREDICTION_COUNT.labels(served_by=name, model_version=slot["version"], predicted_class=str(prediction)).inc()
    PREDICTION_LATENCY.observe(latency)
    PREDICTION_CONFIDENCE.observe(proba)

    if transaction.label is not None:
        outcome = {(1, 1): "tp", (1, 0): "fp", (0, 0): "tn", (0, 1): "fn"}[(prediction, int(transaction.label))]
        LABELED_OUTCOMES.labels(served_by=name, outcome=outcome).inc()

    log_prediction_audit(transaction.data, proba, prediction, name, slot, transaction.label)

    return {
        "fraud_probability": proba,
        "is_fraud": prediction,
        "threshold": slot["threshold"],
        "model_version": slot["version"],
        "served_by": name,
        "timestamp": _now(),
    }