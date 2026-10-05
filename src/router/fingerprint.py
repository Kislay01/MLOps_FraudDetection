"""Drift fingerprinting: per-feature KS vs a rolling clean reference, summarised per drift group."""
import numpy as np
import pandas as pd

from src.drift.drift_generators import FREQ_COLS, PRODUCT_COLS, VELOCITY_COLS, feature_cols

GROUPS = {
    "A": ["TransactionAmt"],
    "B": FREQ_COLS + PRODUCT_COLS,
    "C": VELOCITY_COLS + ["C1", "C2"],
}


def _ks(ref_sorted, x):
    x = np.sort(x)
    allv = np.concatenate([ref_sorted, x])
    cdf_ref = np.searchsorted(ref_sorted, allv, side="right") / len(ref_sorted)
    cdf_x = np.searchsorted(x, allv, side="right") / len(x)
    return float(np.abs(cdf_ref - cdf_x).max())


class Fingerprinter:
    def __init__(self, reference: pd.DataFrame):
        self.features = feature_cols(reference)
        grouped = {c for cols in GROUPS.values() for c in cols}
        self.other = [c for c in self.features if c not in grouped]
        self.tau = {c: 0.05 for c in self.features}
        self.set_reference(reference)

    def set_reference(self, reference: pd.DataFrame):
        self.ref = {c: np.sort(reference[c].to_numpy(dtype=np.float64)) for c in self.features}

    def ks(self, window: pd.DataFrame):
        return {c: _ks(self.ref[c], window[c].to_numpy(dtype=np.float64)) for c in self.features}

    def calibrate_rolling(self, clean: pd.DataFrame, positions, ref_size=5000, w=1000,
                          margin=1.15, floor=0.03):
        """Noise threshold per feature = max KS of (window vs the preceding ref_size clean rows) * margin."""
        mx = {c: 0.0 for c in self.features}
        for i in positions:
            self.set_reference(clean.iloc[i - ref_size:i])
            for c, v in self.ks(clean.iloc[i:i + w]).items():
                mx[c] = max(mx[c], v)
        self.tau = {c: max(floor, mx[c] * margin) for c in self.features}

    def score(self, window: pd.DataFrame):
        ks = self.ks(window)
        ratio = {c: ks[c] / self.tau[c] for c in self.features}
        out = {g: float(np.mean(sorted((ratio[c] for c in cols), reverse=True)[:3])) for g, cols in GROUPS.items()}
        out["breadth"] = int(sum(ratio[c] > 1 for c in self.other))
        return out


def classify(s, group_thr=1.25, breadth_thr=20):
    if s["breadth"] >= breadth_thr:
        return "U"
    active = {g: s[g] for g in GROUPS if s[g] >= group_thr}
    if not active:
        return "normal"
    if len(active) >= 2:
        return "U"
    return next(iter(active))