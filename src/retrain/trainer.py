"""Fresh-window retraining: train a candidate on the latest labeled drifted rows, compare it with
the serving model on a held-out tail of the same window, and register it in MLflow."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import fbeta_score, precision_recall_curve

from src.drift.drift_generators import feature_cols

REGISTERED_NAME = "fraud-retrained"
ALIAS = "candidate"


def f2_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f2 = (5 * prec * rec) / np.maximum(4 * prec + rec, 1e-12)
    return float(thr[int(np.nanargmax(f2[:-1]))])


def f2_at(model, thr, df, features):
    p = model.predict_proba(df.reindex(columns=features, fill_value=0))[:, 1]
    return float(fbeta_score(df["isFraud"], (p >= thr).astype(int), beta=2, zero_division=0))


def split(fresh: pd.DataFrame, holdout=0.2):
    """Time-ordered: the newest `holdout` share is kept for the comparison, never trained on."""
    cut = int(len(fresh) * (1 - holdout))
    return fresh.iloc[:cut], fresh.iloc[cut:]


def train_candidate(train_fresh: pd.DataFrame, clean: pd.DataFrame, features=None,
                    n_estimators=200, seed=42):
    """Fit on the fresh drifted rows plus clean rows (so normal traffic is not forgotten).
    The F2-optimal threshold is chosen on the last 20% of the training part."""
    df = pd.concat([clean, train_fresh]) if len(clean) else train_fresh
    features = features or feature_cols(df)
    cut = int(len(train_fresh) * 0.8)
    fit_df = pd.concat([clean, train_fresh.iloc[:cut]]) if len(clean) else train_fresh.iloc[:cut]
    val_df = train_fresh.iloc[cut:]
    y = fit_df["isFraud"].astype(int)
    if y.sum() < 10 or val_df["isFraud"].sum() < 3:
        raise ValueError("not enough fraud rows in the fresh window to retrain")
    spw = (y == 0).sum() / max(y.sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=6, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=spw, tree_method="hist",
        eval_metric="aucpr", n_jobs=2, random_state=seed,
    )
    model.fit(fit_df[features].astype("float32"), y)
    p = model.predict_proba(val_df[features].astype("float32"))[:, 1]
    return model, f2_threshold(val_df["isFraud"].to_numpy(), p), features


def register(model, threshold, tags, tracking_uri):
    """Log the candidate as a new version of fraud-retrained and point alias 'candidate' at it."""
    import mlflow
    import mlflow.xgboost
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fraudar-retraining")
    with mlflow.start_run(run_name="fresh-window-retrain"):
        mlflow.log_params({k: v for k, v in tags.items() if k != "threshold"})
        mlflow.log_metric("threshold", threshold)
        info = mlflow.xgboost.log_model(model, name="model", registered_model_name=REGISTERED_NAME)
    client = MlflowClient(tracking_uri)
    version = str(info.registered_model_version)
    client.set_registered_model_alias(REGISTERED_NAME, ALIAS, version)
    client.set_model_version_tag(REGISTERED_NAME, version, "threshold", str(round(threshold, 4)))
    client.set_model_version_tag(REGISTERED_NAME, version, "role", "retrained")
    for k, v in tags.items():
        client.set_model_version_tag(REGISTERED_NAME, version, k, str(v))
    return version


def load_registered(name, alias, tracking_uri):
    import mlflow
    import mlflow.xgboost
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    mv = MlflowClient(tracking_uri).get_model_version_by_alias(name, alias)
    model = mlflow.xgboost.load_model(f"models:/{name}@{alias}")
    thr = float((mv.tags or {}).get("threshold", 0.5))
    return model, thr, list(model.get_booster().feature_names)
