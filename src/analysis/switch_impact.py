"""F2 of the rows served just before and just after every model switch, from the API audit log."""
import argparse

import pandas as pd


def f2(df):
    tp = int(((df.is_fraud == 1) & (df.label == 1)).sum())
    fp = int(((df.is_fraud == 1) & (df.label == 0)).sum())
    fn = int(((df.is_fraud == 0) & (df.label == 1)).sum())
    d = 5 * tp + 4 * fn + fp
    return (round(5 * tp / d, 3) if d else float("nan")), tp + fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="logs/predictions_audit.jsonl")
    ap.add_argument("--since", default=None, help="UTC time, e.g. 2026-10-06T11:00")
    ap.add_argument("--rows", type=int, default=300, help="rows compared on each side of a switch")
    ap.add_argument("--min-seg", type=int, default=150)
    ap.add_argument("--out", default="reports/switch_impact.csv")
    a = ap.parse_args()

    df = pd.read_json(a.audit, lines=True)
    df = df[df.label.notna()].copy()
    df["timestamp"] = pd.to_datetime(df.timestamp)
    if a.since:
        df = df[df.timestamp >= pd.Timestamp(a.since, tz="UTC")]
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["seg"] = (df.served_by != df.served_by.shift()).cumsum()
    groups = [g for _, g in df.groupby("seg", sort=True)]

    out = []
    for prev, nxt in zip(groups, groups[1:]):
        if len(prev) < a.min_seg or len(nxt) < a.min_seg:
            continue
        before, after = prev.tail(a.rows), nxt.head(a.rows)
        fb, pb = f2(before)
        fa, pa = f2(after)
        out.append({
            "switch_utc": nxt.timestamp.iloc[0].strftime("%H:%M:%S"),
            "from": prev.served_by.iloc[0], "to": nxt.served_by.iloc[0],
            "f2_before": fb, "fraud_before": pb, "f2_after": fa, "fraud_after": pa,
        })
    res = pd.DataFrame(out)
    res.to_csv(a.out, index=False)
    print(res.to_string(index=False))
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
