"""Consume the transaction stream and score every record through the routing API."""
import argparse
import json
import time
from collections import Counter

import requests
from kafka import KafkaConsumer

KAFKA = "localhost:9092"
TOPIC = "transactions"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--group", default=f"scoring-{int(time.time())}")
    a = ap.parse_args()

    consumer = KafkaConsumer(
        TOPIC, bootstrap_servers=KAFKA, auto_offset_reset="latest", group_id=a.group,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )
    sess = requests.Session()
    served, oc, n = Counter(), Counter(), 0
    print(f"Scoring consumer ready (group {a.group}). Start the producer now.", flush=True)

    for msg in consumer:
        rec = msg.value
        label = int(rec.get("isFraud", 0))
        r = sess.post(f"{a.api}/predict", json={"data": rec, "label": label}, timeout=10)
        r.raise_for_status()
        p = r.json()
        served[p["served_by"]] += 1
        oc[{(1, 1): "tp", (1, 0): "fp", (0, 0): "tn", (0, 1): "fn"}[(p["is_fraud"], label)]] += 1
        n += 1
        if n % 200 == 0:
            tp, fp, fn = oc["tp"], oc["fp"], oc["fn"]
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec_ = tp / (tp + fn) if tp + fn else 0.0
            f2 = 5 * prec * rec_ / (4 * prec + rec_) if prec + rec_ else 0.0
            print(f"[{n}] last 200: precision={prec:.2f} recall={rec_:.2f} F2={f2:.2f} | served_by={dict(served)}", flush=True)
            served.clear()
            oc.clear()


if __name__ == "__main__":
    main()