"""Register zoo models in MLflow with alias 'production' and threshold/drift tags."""
import joblib
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("fraudar-model-zoo")
client = MlflowClient()

ZOO = {
    "spec_A": ("fraud-spec-A", "A", "card_testing_amount"),
    "spec_B": ("fraud-spec-B", "B", "account_takeover_device"),
    "spec_C": ("fraud-spec-C", "C", "fraud_ring_velocity"),
    "universal": ("fraud-universal", "U", "universal_covariate_shift"),
}

for key, (reg, kind, desc) in ZOO.items():
    b = joblib.load(f"models/zoo_{key}.pkl")
    thr = round(b["threshold"], 4)
    with mlflow.start_run(run_name=reg):
        mlflow.log_params({"drift_type": kind, "threshold": thr, "n_features": len(b["features"])})
        mlflow.xgboost.log_model(b["model"], name="model", registered_model_name=reg)
    ver = max(int(v.version) for v in client.search_model_versions(f"name='{reg}'"))
    client.set_registered_model_alias(reg, "production", ver)
    client.set_model_version_tag(reg, str(ver), "threshold", str(thr))
    client.set_model_version_tag(reg, str(ver), "drift_type", kind)
    client.set_model_version_tag(reg, str(ver), "description", desc)
    print(f"{reg} v{ver} -> alias production, threshold {thr}")

# v1 stays the normal-traffic champion; record its threshold/role for the router
client.set_model_version_tag("fraud-detection-model", "1", "threshold", "0.6210")
client.set_model_version_tag("fraud-detection-model", "1", "role", "normal")
print("fraud-detection-model v1 tagged (role=normal, threshold 0.6210)")