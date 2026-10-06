"""Offline F2 of every zoo model on every scenario test set."""
import joblib
import pandas as pd

MODELS = ["spec_A", "spec_B", "spec_C", "universal"]
SETS = ["normal", "A", "B", "C", "U"]


def f2(y, p):
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    d = 5 * tp + 4 * fn + fp
    return round(5 * tp / d, 3) if d else float("nan")


data = {s: joblib.load(f"data/drift_scenarios/{s}_test.pkl") for s in SETS}
rows = {}
for m in MODELS:
    d = joblib.load(f"models/zoo_{m}.pkl")
    model, feats, thr = d["model"], d["features"], d["threshold"]
    rows[m] = {}
    for s, df in data.items():
        p = (model.predict_proba(df[feats])[:, 1] >= thr).astype(int)
        rows[m][s] = f2(df["isFraud"].to_numpy(), p)
res = pd.DataFrame(rows).T[SETS]
print(res.to_string())
res.to_csv("reports/specialist_cross_matrix.csv")
print("\nsaved reports/specialist_cross_matrix.csv")
