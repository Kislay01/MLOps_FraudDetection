"""Check that a deployed API (e.g. on a cloud VM) scores exactly like the local model files.

For each model, the same rows are scored by the API (POST /admin/score, which does not touch routing
or metrics) and by the local .pkl; the script reports the largest probability difference and F2 at
the model's threshold both ways. Differences above --tol mean the deployment does not reproduce the
local numbers (usually a different xgboost version or feature handling).

  python -m src.analysis.parity_check --api http://<vm>:8000 --rows 2000
"""
import argparse
import json
import sys

import joblib
import numpy as np
import requests
from sklearn.metrics import fbeta_score

from src.drift.eval_models import feature_names, load_model
from src.settings import API_URL, admin_headers

CHECKS = {   # model -> (local file, test set)
    "fraud-detection-model": ("xgb_fraud_model_v1.pkl", "normal"),
    "fraud-spec-A": ("zoo_spec_A.pkl", "A"),
    "fraud-spec-B": ("zoo_spec_B.pkl", "B"),
    "fraud-spec-C": ("zoo_spec_C.pkl", "C"),
    "fraud-universal": ("zoo_universal.pkl", "U"),
}


def remote_scores(api, model, records, batch=500):
    out, thr = [], None
    for i in range(0, len(records), batch):
        r = requests.post(f"{api}/admin/score", json={"model": model, "rows": records[i:i + batch]},
                          headers=admin_headers(), timeout=120)
        r.raise_for_status()
        j = r.json()
        out += j["probabilities"]
        thr = j["threshold"]
    return np.array(out), thr


def check(api, model, local_path, df, rows):
    m = load_model(local_path)
    df = df.iloc[:rows]
    feats = feature_names(m)
    local = m.predict_proba(df.reindex(columns=feats, fill_value=0))[:, 1]
    records = json.loads(df[feats + ["isFraud"]].to_json(orient="records"))
    remote, thr = remote_scores(api, model, records)
    y = df["isFraud"].to_numpy()
    return {"model": model, "rows": len(df), "threshold": thr,
            "max_abs_diff": float(np.max(np.abs(local - remote))),
            "f2_local": round(float(fbeta_score(y, local >= thr, beta=2, zero_division=0)), 4),
            "f2_api": round(float(fbeta_score(y, remote >= thr, beta=2, zero_division=0)), 4)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=API_URL)
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--data-dir", default="data/drift_scenarios")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--models", nargs="*", default=list(CHECKS))
    a = ap.parse_args(argv)
    results = []
    for model in a.models:
        path, s = CHECKS[model]
        df = joblib.load(f"{a.data_dir}/{s}_test.pkl")
        res = check(a.api, model, f"{a.models_dir}/{path}", df, a.rows)
        res["ok"] = res["max_abs_diff"] <= a.tol and res["f2_local"] == res["f2_api"]
        results.append(res)
        print(json.dumps(res))
    ok = all(r["ok"] for r in results)
    print("PARITY OK" if ok else "PARITY FAILED")
    return ok, results


if __name__ == "__main__":
    sys.exit(0 if main()[0] else 1)
