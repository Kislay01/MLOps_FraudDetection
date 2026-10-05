"""Per-feature KS fingerprint of each drift vs the clean test slice."""
import pathlib

import joblib
import pandas as pd
from scipy.stats import ks_2samp

from src.drift.drift_generators import feature_cols

D = "data/drift_scenarios"
normal = joblib.load(f"{D}/normal_test.pkl")
cols = feature_cols(normal)

res = {}
for k in ["A", "B", "C", "U"]:
    df = joblib.load(f"{D}/{k}_test.pkl")
    res[k] = {c: ks_2samp(normal[c].astype(float), df[c].astype(float)).statistic for c in cols}

M = pd.DataFrame(res)
pathlib.Path("reports").mkdir(exist_ok=True)
M.to_csv("reports/fingerprint_matrix.csv")

for k in M:
    print(f"\n{k}: drifted features (KS>0.05): {int((M[k] > 0.05).sum())} of {len(M)}")
    print(M[k].sort_values(ascending=False).head(6).round(3).to_string())