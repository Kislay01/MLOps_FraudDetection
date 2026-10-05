"""Evaluate registered models on normal and drifted test sets."""
import joblib
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score, roc_auc_score

MODELS = {
    "v1": ("models/xgb_fraud_model_v1.pkl", 0.6210),
    "v2": ("models/xgb_fraud_model_v2_retrained.pkl", 0.6801),
}
SETS = ["normal", "A", "B", "C", "U"]


def load_model(path):
    m = joblib.load(path)
    return m["model"] if isinstance(m, dict) else m


def feature_names(m):
    names = getattr(m, "feature_names_in_", None)
    if names is None:
        names = m.get_booster().feature_names
    return list(names)


def evaluate(model, thr, df):
    X = df.reindex(columns=feature_names(model), fill_value=0)
    y = df["isFraud"].to_numpy()
    p = model.predict_proba(X)[:, 1]
    pred = (p >= thr).astype(int)
    out = {
        "auc": roc_auc_score(y, p),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f2": fbeta_score(y, pred, beta=2, zero_division=0),
    }
    if "injected" in df.columns:
        out["recall_injected"] = float(pred[df["injected"].to_numpy() == 1].mean())
    return out


if __name__ == "__main__":
    rows = []
    for mname, (path, thr) in MODELS.items():
        model = load_model(path)
        for s in SETS:
            f = "data/drift_scenarios/normal_test.pkl" if s == "normal" else f"data/drift_scenarios/{s}_test.pkl"
            df = joblib.load(f)
            rows.append({"model": mname, "set": s, **evaluate(model, thr, df)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))