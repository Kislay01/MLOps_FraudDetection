"""Recall-first decision thresholds.

The registered thresholds maximise F2 on validation data. This script finds, for every model,
the highest threshold that still reaches a target recall (default 0.90) on data the model was
not trained on, and reports what that costs in precision on the drift test sets.

Tuning data (never the test sets):
  specialists A/B/C : last 20% of data/drift_scenarios/<k>_train.pkl (the same holdout train_zoo uses)
  universal         : last 20% of each of A/B/C/U train sets
  v1 (normal)       : data/processed/val_processed_v2.pkl rows 30000+ (not used by the stream or tests)

Output: models/thresholds_recall.json and reports/threshold_tuning.csv.
Serve with it: FRAUDAR_THRESHOLDS=models/thresholds_recall.json, or push live with --apply.
"""
import argparse
import json
import pathlib

import joblib
import numpy as np
import pandas as pd
import requests

from src.drift.eval_models import evaluate, feature_names, load_model
from src.settings import API_URL, admin_headers

REGISTRY_NAME = {"v1": "fraud-detection-model", "spec_A": "fraud-spec-A", "spec_B": "fraud-spec-B",
                 "spec_C": "fraud-spec-C", "universal": "fraud-universal"}
OWN_SET = {"v1": "normal", "spec_A": "A", "spec_B": "B", "spec_C": "C", "universal": "U"}


def holdout(df, frac=0.2):
    return df.iloc[int(len(df) * (1 - frac)):]


def recall_threshold(y, p, target):
    """Highest threshold whose recall on (y, p) is >= target."""
    pos = np.sort(p[y == 1])[::-1]
    if len(pos) == 0:
        raise ValueError("no positive rows to tune on")
    k = int(np.ceil(target * len(pos)))
    return float(pos[k - 1])


def load_models(models_dir, v1_thr):
    out = {"v1": (load_model(models_dir / "xgb_fraud_model_v1.pkl"), v1_thr)}
    for k in ["spec_A", "spec_B", "spec_C", "universal"]:
        b = joblib.load(models_dir / f"zoo_{k}.pkl")
        out[k] = (b["model"], float(b["threshold"]))
    return out


def tuning_sets(data_dir, val):
    tr = {k: joblib.load(data_dir / f"{k}_train.pkl") for k in ["A", "B", "C", "U"]}
    return {
        "v1": val.iloc[30000:],
        "spec_A": holdout(tr["A"]), "spec_B": holdout(tr["B"]), "spec_C": holdout(tr["C"]),
        "universal": pd.concat([holdout(tr[k]) for k in ["A", "B", "C", "U"]]),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-recall", type=float, default=0.90)
    ap.add_argument("--data-dir", default="data/drift_scenarios")
    ap.add_argument("--val", default="data/processed/val_processed_v2.pkl")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--v1-threshold", type=float, default=0.6210)
    ap.add_argument("--apply", action="store_true", help="also set the thresholds live on the API")
    ap.add_argument("--api", default=API_URL)
    a = ap.parse_args(argv)

    data_dir, models_dir = pathlib.Path(a.data_dir), pathlib.Path(a.models_dir)
    models = load_models(models_dir, a.v1_threshold)
    tune = tuning_sets(data_dir, joblib.load(a.val))
    tests = {"normal": joblib.load(data_dir / "normal_test.pkl")}
    tests.update({k: joblib.load(data_dir / f"{k}_test.pkl") for k in ["A", "B", "C", "U"]})

    new, rows = {}, []
    for name, (model, old_thr) in models.items():
        df = tune[name]
        p = model.predict_proba(df.reindex(columns=feature_names(model), fill_value=0))[:, 1]
        thr = min(old_thr, recall_threshold(df["isFraud"].to_numpy(), p, a.target_recall))
        new[REGISTRY_NAME[name]] = round(thr, 6)
        for label, t in (("f2_optimal", old_thr), ("recall_first", thr)):
            for s in ("normal", OWN_SET[name]) if OWN_SET[name] != "normal" else ("normal",):
                m = evaluate(model, t, tests[s])
                rows.append({"model": name, "threshold_kind": label, "threshold": round(t, 4), "test_set": s,
                             **{k: round(v, 4) for k, v in m.items() if k in ("precision", "recall", "f2")}})

    pathlib.Path(a.reports_dir).mkdir(exist_ok=True)
    out = models_dir / "thresholds_recall.json"
    out.write_text(json.dumps({"target_recall": a.target_recall, "thresholds": new}, indent=2))
    report = pd.DataFrame(rows)
    report.to_csv(pathlib.Path(a.reports_dir) / "threshold_tuning.csv", index=False)
    print(report.to_string(index=False))
    print(f"\nwrote {out}")

    if a.apply:
        for m, t in new.items():
            r = requests.post(f"{a.api}/admin/threshold", json={"model": m, "threshold": t,
                              "reason": f"recall-first {a.target_recall}"}, headers=admin_headers(), timeout=5)
            print(m, r.status_code, r.json() if r.ok else r.text)
    return new, report


if __name__ == "__main__":
    main()
