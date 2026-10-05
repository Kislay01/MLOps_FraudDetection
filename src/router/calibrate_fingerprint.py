"""Calibrate the fingerprinter (rolling reference) and test detection/classification."""
import joblib
import pandas as pd

from src.drift.drift_generators import apply_drift
from src.router.fingerprint import Fingerprinter, classify

W, R = 1000, 5000
THRS = (1.0, 1.25, 1.5)
KINDS, FRACS = ["A", "B", "C", "U"], [0.10, 0.20, 0.30]

train = joblib.load("data/processed/train_processed_v2.pkl")
val = joblib.load("data/processed/val_processed_v2.pkl")

fp = Fingerprinter(train.iloc[-R:])

clean = val.iloc[:30000].reset_index(drop=True)
pos = list(range(R, len(clean) - W + 1, 500))
cal_pos, test_pos = pos[0::2], pos[1::2]
fp.calibrate_rolling(clean, cal_pos, R, W)

fresh_scores = []
for i in test_pos:
    fp.set_reference(clean.iloc[i - R:i])
    fresh_scores.append(fp.score(clean.iloc[i:i + W]))

print(f"clean: {len(cal_pos)} calibration windows, {len(test_pos)} held-out windows")
for thr in THRS:
    fa = [classify(s, group_thr=thr) for s in fresh_scores]
    print(f"  group_thr={thr}: false alarms {sum(l != 'normal' for l in fa)}/{len(fa)}")
print("  max clean scores:", {k: round(max(s[k] for s in fresh_scores), 2) for k in ["A", "B", "C", "breadth"]})

# Drift tests: reference = clean rows just before each window
base = val.iloc[:30000].reset_index(drop=True)
drifted = {(k, f): apply_drift(base, k, train, frac=f, seed=5) for k in KINDS for f in FRACS}
scores = {key: [] for key in drifted}
for i in range(R, len(base) - W + 1, W):
    fp.set_reference(base.iloc[i - R:i])
    for key, d in drifted.items():
        scores[key].append(fp.score(d.iloc[i:i + W]))

rows = []
for (kind, frac), sc in scores.items():
    row = {"kind": kind, "inject": frac}
    for thr in THRS:
        row[f"acc@{thr}"] = sum(classify(s, group_thr=thr) == kind for s in sc) / len(sc)
    for g in ["A", "B", "C", "breadth"]:
        row[g] = pd.Series([s[g] for s in sc]).mean()
    rows.append(row)
print(pd.DataFrame(rows).round(2).to_string(index=False))

fp.set_reference(train.iloc[-R:])
joblib.dump(fp, "models/fingerprinter.pkl", compress=3)