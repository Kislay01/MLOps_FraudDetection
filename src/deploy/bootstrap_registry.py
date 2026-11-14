"""Register v1 and the model zoo from models/*.pkl into an empty MLflow (e.g. a fresh cloud VM),
with the same names, aliases and threshold tags the API expects. Safe to re-run: a model that
already has its alias is left alone."""
import mlflow
import mlflow.xgboost
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.drift.eval_models import load_model
from src.settings import MLFLOW_URI

V1 = ("fraud-detection-model", "champion", "models/xgb_fraud_model_v1.pkl", 0.6210,
      {"role": "normal"})
ZOO = [
    ("fraud-spec-A", "production", "models/zoo_spec_A.pkl", {"drift_type": "A", "description": "card_testing_amount"}),
    ("fraud-spec-B", "production", "models/zoo_spec_B.pkl", {"drift_type": "B", "description": "account_takeover_device"}),
    ("fraud-spec-C", "production", "models/zoo_spec_C.pkl", {"drift_type": "C", "description": "fraud_ring_velocity"}),
    ("fraud-universal", "production", "models/zoo_universal.pkl", {"drift_type": "U", "description": "universal_covariate_shift"}),
]


def has_alias(client, name, alias):
    try:
        client.get_model_version_by_alias(name, alias)
        return True
    except MlflowException:
        return False


def register(client, name, alias, model, threshold, tags):
    if has_alias(client, name, alias):
        print(f"{name}@{alias} already registered; skipped")
        return None
    with mlflow.start_run(run_name=f"bootstrap-{name}"):
        mlflow.log_params({"threshold": round(threshold, 4), **tags})
        info = mlflow.xgboost.log_model(model, name="model", registered_model_name=name)
    v = str(info.registered_model_version)
    client.set_registered_model_alias(name, alias, v)
    client.set_model_version_tag(name, v, "threshold", str(round(threshold, 4)))
    for k, val in tags.items():
        client.set_model_version_tag(name, v, k, str(val))
    print(f"{name} v{v} -> @{alias}, threshold {round(threshold, 4)}")
    return v


def main(tracking_uri=MLFLOW_URI, root="."):
    import joblib
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fraudar-bootstrap")
    client = MlflowClient(tracking_uri)
    name, alias, path, thr, tags = V1
    register(client, name, alias, load_model(f"{root}/{path}"), thr, tags)
    for name, alias, path, tags in ZOO:
        b = joblib.load(f"{root}/{path}")
        register(client, name, alias, b["model"], float(b["threshold"]), tags)


if __name__ == "__main__":
    main()
