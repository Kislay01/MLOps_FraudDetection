"""Train the model zoo (3 specialists + 1 universal) and print the model x drift matrix."""
import json
import pathlib

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_recall_curve

from src.drift.drift_generators import feature_cols
from src.drift.eval_models import evaluate, load_model

D = pathlib.Path("data/drift_scenarios")
M = pathlib.Path("models")
KINDS = ["A", "B", "C", "U"]


def f2_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f2 = (5 * prec * rec) / np.maximum(4 * prec + rec, 1e-12)
    return float(thr[int(np.nanargmax(f2[:-1]))])


def fit(df, features):
    cut = int(len(df) * 0.8)
    tr, va = df.iloc[:cut], df.iloc[cut:]
    spw = (tr.isFraud == 0).sum() / max((tr.isFraud == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=spw, tree_method="hist",
        eval_metric="aucpr", n_jobs=4, random_state=42,
    )
    model.fit(tr[features], tr.isFraud)
    thr = f2_threshold(va.isFraud.to_numpy(), model.predict_proba(va[features])[:, 1])
    return model, thr


if __name__ == "__main__":
    train = {k: joblib.load(D / f"{k}_train.pkl") for k in KINDS}
    test = {"normal": joblib.load(D / "normal_test.pkl")}
    test.update({k: joblib.load(D / f"{k}_test.pkl") for k in KINDS})
    features = feature_cols(train["A"])

    zoo, meta = {}, {}
    for k in ["A", "B", "C"]:
        zoo[f"spec_{k}"] = fit(train[k], features)
    mixed = pd.concat([train[k].sample(10000, random_state=7) for k in KINDS])
    mixed = mixed.sample(frac=1, random_state=1).reset_index(drop=True)
    zoo["universal"] = fit(mixed, features)

    for name, (model, thr) in zoo.items():
        joblib.dump({"model": model, "features": features, "threshold": thr}, M / f"zoo_{name}.pkl")
        meta[name] = {"threshold": thr}
    json.dump(meta, open(M / "zoo_meta.json", "w"), indent=2)

    v1 = load_model("models/xgb_fraud_model_v1.pkl")
    candidates = {"v1": (v1, 0.6210), **zoo}

    rows = []
    for name, (model, thr) in candidates.items():
        for s, df in test.items():
            rows.append({"model": name, "set": s, **evaluate(model, thr, df)})
    R = pd.DataFrame(rows)
    order = list(candidates)
    for metric in ["f2", "auc", "recall_injected"]:
        print(f"\n== {metric} ==")
        print(R.pivot(index="model", columns="set", values=metric).loc[order].round(3).to_string())
    print("\nthresholds:", {k: round(v["threshold"], 4) for k, v in meta.items()})