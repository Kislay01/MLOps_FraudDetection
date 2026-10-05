"""Build drift scenario datasets (A, B, C, U): train slices for specialists, test slices for evaluation/streaming."""
import pathlib

import joblib
import numpy as np

from src.drift.drift_generators import DRIFT_NAMES, apply_drift

OUT = pathlib.Path("data/drift_scenarios")
OUT.mkdir(parents=True, exist_ok=True)


def save(df, path):
    """float64 -> float32 and compress, to keep files small for DVC/R2."""
    f64 = df.select_dtypes(include="float64").columns
    df = df.astype({c: np.float32 for c in f64})
    joblib.dump(df, path, compress=3)


train = joblib.load("data/processed/train_processed_v2.pkl")
val = joblib.load("data/processed/val_processed_v2.pkl")

# Clean reference for quantiles/std; time-ordered slices (latest train rows, earliest val rows)
ref = train
train_slice = train.iloc[-40000:]
test_slice = val.iloc[:30000]

for kind, name in DRIFT_NAMES.items():
    tr = apply_drift(train_slice, kind, ref, frac=0.10, seed=11)
    te = apply_drift(test_slice, kind, ref, frac=0.10, seed=22)
    save(tr, OUT / f"{kind}_train.pkl")
    save(te, OUT / f"{kind}_test.pkl")
    print(
        f"{kind} ({name}): train {tr.shape} fraud={tr.isFraud.mean():.3f} "
        f"| test {te.shape} fraud={te.isFraud.mean():.3f} injected={int(te.injected.sum())}"
    )

save(test_slice.reset_index(drop=True), OUT / "normal_test.pkl")
print("normal_test:", test_slice.shape, f"fraud={test_slice.isFraud.mean():.3f}")