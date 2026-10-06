"""Rows the orchestrator has seen minus rows the scoring consumer has scored, over the latest run."""
import argparse

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--audit", default="logs/predictions_audit.jsonl")
ap.add_argument("--decisions", default="logs/decisions.jsonl")
ap.add_argument("--tps", type=float, default=50.0)
a = ap.parse_args()

aud = pd.read_json(a.audit, lines=True)
aud["timestamp"] = pd.to_datetime(aud.timestamp, utc=True)
aud = aud.sort_values("timestamp")
gap = aud.timestamp.diff().dt.total_seconds() > 20
start = aud.timestamp[gap].iloc[-1] if gap.any() else aud.timestamp.iloc[0]
aud = aud[aud.timestamp >= start]

dec = pd.read_json(a.decisions, lines=True)
dec["timestamp"] = pd.to_datetime(dec.timestamp, utc=True)
dec = dec.sort_values("timestamp").reset_index(drop=True)
drops = dec.index[dec.rows_seen.diff() < 0]
if len(drops):
    dec = dec.loc[drops[-1]:]
dec = dec[dec.timestamp >= start].reset_index(drop=True)

dec["scored"] = aud.timestamp.values.searchsorted(dec.timestamp.values, side="right")
dec["lag_rows"] = dec.rows_seen - dec.scored
dec["lag_s"] = (dec.lag_rows / a.tps).round(1)
dec["time"] = dec.timestamp.dt.strftime("%H:%M:%S")

show = dec[dec.event.isin(["switch", "recover"]) | (dec.index % 10 == 0)]
print(show[["time", "rows_seen", "scored", "lag_rows", "lag_s", "event", "route"]].to_string(index=False))
print(f"\nmax lag: {int(dec.lag_rows.max())} rows (~{dec.lag_s.max()} s of stream time)")
