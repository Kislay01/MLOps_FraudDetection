"""Offline end-to-end replay: normal -> A -> B -> C -> U drifts with fingerprinting + routing policy."""
import joblib
import pandas as pd

from src.drift.drift_generators import apply_drift
from src.router.fingerprint import classify
from src.router.policy import RoutingPolicy

W, R, STEP = 1000, 5000, 500
SCHEDULE = [
    (5000, 8000, "normal"), (8000, 11000, "A"), (11000, 13000, "normal"),
    (13000, 16000, "B"), (16000, 18000, "normal"), (18000, 21000, "C"),
    (21000, 23000, "normal"), (23000, 27000, "U"), (27000, 30000, "normal"),
]

train = joblib.load("data/processed/train_processed_v2.pkl")
val = joblib.load("data/processed/val_processed_v2.pkl")
base = val.iloc[:30000].reset_index(drop=True)
fp = joblib.load("models/fingerprinter.pkl")

drifted = {k: apply_drift(base, k, train, frac=0.30, seed=5) for k in ["A", "B", "C", "U"]}
clean = base.assign(injected=0)
parts = [clean.iloc[:R]]
for s, e, k in SCHEDULE:
    parts.append((clean if k == "normal" else drifted[k]).iloc[s:e])
stream = pd.concat(parts).reset_index(drop=True)


def truth(i):
    for s, e, k in SCHEDULE:
        if s <= i < e:
            return k
    return "normal"


policy = RoutingPolicy(confirm=2, recover=3)
buffer = clean.iloc[:R]
fp.set_reference(buffer)

rows = []
for end in range(R + W, len(stream) + 1, STEP):
    s = fp.score(stream.iloc[end - W:end])
    label = classify(s)
    d = policy.update(label)
    if label == "normal" and policy.state == "normal":
        buffer = pd.concat([buffer, stream.iloc[end - STEP:end]]).iloc[-R:]
        fp.set_reference(buffer)
    rows.append({
        "end_row": end, "truth": truth(end - 1), "label": label, "event": d["event"],
        "route": d["route"], "A": s["A"], "B": s["B"], "C": s["C"], "breadth": s["breadth"],
    })

R_df = pd.DataFrame(rows)
print(R_df.round(2).to_string(index=False))
print("\nSwitch/recover events:")
print(R_df[R_df.event.isin(["switch", "recover"])][["end_row", "truth", "event", "route"]].to_string(index=False))