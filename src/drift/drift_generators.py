"""Synthetic drift generators for Fraudar multi-model routing.

Each drift takes a clean dataframe, injects fraud rows with a distinct feature
signature, and returns a copy with an `injected` flag (1 = injected fraud row).
"""
import re

import numpy as np
import pandas as pd

META_COLS = ["TransactionID", "isFraud", "TransactionDT", "card_entity", "injected"]

DRIFT_NAMES = {
    "A": "card_testing_amount",
    "B": "account_takeover_device",
    "C": "fraud_ring_velocity",
    "U": "universal_covariate_shift",
}

PRODUCT_COLS = ["ProductCD_C", "ProductCD_H", "ProductCD_R", "ProductCD_S", "ProductCD_W"]
FREQ_COLS = ["P_emaildomain_freq", "R_emaildomain_freq", "DeviceType_freq", "DeviceInfo_freq"]
VELOCITY_COLS = [
    "entity_txn_count_so_far", "time_since_last_txn",
    "txn_1h_count", "txn_1h_avg_amt", "txn_24h_count", "txn_24h_avg_amt",
]

# Fixed seed so train/test share the same shifted feature set and factors for U.
_U_SEED = 1234


def feature_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in META_COLS]


def _set(df, pos, col, values):
    """Assign values at integer positions, preserving the column dtype."""
    arr = df[col].to_numpy(copy=True)
    arr[pos] = values
    df[col] = arr.astype(df[col].dtype)


def _pick(df, rng, frac):
    pool = np.flatnonzero(df["isFraud"].to_numpy() == 0)
    n = int(len(df) * frac)
    return rng.choice(pool, size=n, replace=False)


def u_shifted_columns(df):
    cand = [c for c in df.columns if re.fullmatch(r"(V|C|D)\d+", c)]
    rng = np.random.default_rng(_U_SEED)
    cols = list(rng.choice(cand, size=min(80, len(cand)), replace=False))
    factors = rng.uniform(1.4, 2.2, size=len(cols))
    return cols, factors


def apply_drift(df: pd.DataFrame, kind: str, ref: pd.DataFrame, frac: float = 0.10, seed: int = 0):
    """Return a drifted copy of df. `ref` = clean reference data (for quantiles/std)."""
    assert kind in DRIFT_NAMES, f"unknown drift {kind}"
    rng = np.random.default_rng(seed)
    out = df.reset_index(drop=True).copy()
    out["injected"] = 0
    pos = _pick(out, rng, frac)
    n = len(pos)

    if kind == "A":  # card testing: micro-amount probes
        _set(out, pos, "TransactionAmt", rng.uniform(0.5, 3.0, n).round(2))

    elif kind == "B":  # account takeover: new product + rare email/device
        for c in PRODUCT_COLS:
            _set(out, pos, c, 1 if c == "ProductCD_C" else 0)
        for c in FREQ_COLS:
            rare = float(ref[c].quantile(0.02))
            _set(out, pos, c, np.full(n, rare))

    elif kind == "C":  # fraud ring: burst velocity
        _set(out, pos, "txn_1h_count", rng.integers(8, 25, n))
        _set(out, pos, "txn_24h_count", rng.integers(15, 55, n))
        _set(out, pos, "time_since_last_txn", rng.uniform(1, 30, n))
        _set(out, pos, "txn_1h_avg_amt", rng.uniform(20, 80, n))
        _set(out, pos, "C1", rng.integers(10, 40, n))
        _set(out, pos, "C2", rng.integers(10, 40, n))

    elif kind == "U":  # universal: broad covariate shift on all rows
        cols, factors = u_shifted_columns(out)
        for c, f in zip(cols, factors):
            out[c] = out[c].astype(float) * f
            std = float(ref[c].std()) * f
            arr = out[c].to_numpy(copy=True)
            arr[pos] = arr[pos] + 2.0 * std
            out[c] = arr

    _set(out, pos, "isFraud", 1)
    out.loc[out.index[pos], "injected"] = 1
    return out