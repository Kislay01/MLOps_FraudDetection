"""Small synthetic stand-in for the processed IEEE-CIS data, so tests run without DVC/R2.
Same column names the pipeline relies on; fraud depends on a few features so models learn something."""
import numpy as np
import pandas as pd
import xgboost as xgb

from src.drift.drift_generators import FREQ_COLS, PRODUCT_COLS, VELOCITY_COLS, feature_cols


def make_frame(n=6000, seed=0, fraud_rate=0.04):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "TransactionID": np.arange(n) + seed * 10_000_000,
        "TransactionDT": np.arange(n) * 30,
        "card_entity": rng.integers(0, 800, n),
        "TransactionAmt": rng.lognormal(4, 1, n).round(2),
    })
    prod = rng.integers(0, len(PRODUCT_COLS), n)
    for i, c in enumerate(PRODUCT_COLS):
        df[c] = (prod == i).astype(int)
    for c in FREQ_COLS:
        df[c] = rng.uniform(0.001, 0.4, n)
    df["entity_txn_count_so_far"] = rng.integers(1, 60, n)
    df["time_since_last_txn"] = rng.exponential(3000, n)
    df["txn_1h_count"] = rng.poisson(1, n)
    df["txn_1h_avg_amt"] = rng.lognormal(4, 0.8, n)
    df["txn_24h_count"] = rng.poisson(3, n)
    df["txn_24h_avg_amt"] = rng.lognormal(4, 0.8, n)
    for k in range(1, 15):
        df[f"C{k}"] = rng.poisson(2, n).astype(float)
    for k in range(1, 16):
        df[f"D{k}"] = rng.exponential(50, n)
    for k in range(1, 61):
        df[f"V{k}"] = rng.normal(1, 0.5, n)
    z = 2.2 * df["V1"] - 1.5 * df["V2"] + 0.8 * np.log1p(df["C1"]) - 4.6 + rng.normal(0, 0.6, n)
    thr = np.quantile(z, 1 - fraud_rate)
    df["isFraud"] = (z >= thr).astype(int)
    return df


def fit(df, n_estimators=60):
    feats = feature_cols(df)
    m = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=4, learning_rate=0.2,
                          tree_method="hist", n_jobs=2, random_state=0)
    m.fit(df[feats], df["isFraud"])
    return m, feats
