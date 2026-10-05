"""Smoke test: route to each model in turn, predict, then restore the default."""
import joblib
import requests

BASE = "http://localhost:8000"
df = joblib.load("data/drift_scenarios/normal_test.pkl")
row = df.iloc[0]
label = int(row["isFraud"])
rec = row.drop(["TransactionID", "isFraud", "TransactionDT", "card_entity"], errors="ignore").astype(float).to_dict()

print("health:", requests.get(f"{BASE}/health").json())
slots = requests.get(f"{BASE}/admin/slots").json()["slots"]
for name in slots:
    requests.post(f"{BASE}/admin/route", json={"model": name, "reason": "smoke_test"})
    p = requests.post(f"{BASE}/predict", json={"data": rec, "label": label}).json()
    print(f"{name:24s} served_by={p['served_by']:24s} proba={p['fraud_probability']:.4f} "
          f"thr={p['threshold']} pred={p['is_fraud']}")
requests.post(f"{BASE}/admin/route", json={"model": "fraud-detection-model", "reason": "smoke_test_end"})
r = requests.post(f"{BASE}/admin/rollback")
print("rollback:", r.status_code, r.json())
print("final:", requests.get(f"{BASE}/health").json()["active"])