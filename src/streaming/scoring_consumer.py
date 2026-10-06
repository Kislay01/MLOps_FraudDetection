"""Consume the transaction stream and score every record through the routing API.
Requests are sent concurrently so the scorer keeps pace with the producer."""
import argparse
import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests
from kafka import KafkaConsumer

KAFKA = "localhost:9092"
TOPIC = "transactions"
OUTCOME = {(1, 1): "tp", (1, 0): "fp", (0, 0): "tn", (0, 1): "fn"}
_local = threading.local()


def session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def score(api, rec):
    label = int(rec.get("isFraud", 0))
    try:
        r = session().post(f"{api}/predict", json={"data": rec, "label": label}, timeout=10)
        r.raise_for_status()
        p = r.json()
    except Exception:
        return None
    return p["served_by"], OUTCOME[(p["is_fraud"], label)]


def kafka_lag(consumer):
    parts = consumer.assignment()
    if not parts:
        return 0
    end = consumer.end_offsets(list(parts))
    return int(sum(end[p] - consumer.position(p) for p in parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--group", default=f"scoring-{int(time.time())}")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    consumer = KafkaConsumer(
        TOPIC, bootstrap_servers=KAFKA, auto_offset_reset="latest", group_id=a.group,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")), max_poll_records=200,
    )
    pool = ThreadPoolExecutor(max_workers=a.workers)
    served, oc, n, errors = Counter(), Counter(), 0, 0
    print(f"Scoring consumer ready (group {a.group}, {a.workers} workers). Start the producer now.", flush=True)

    while True:
        batch = consumer.poll(timeout_ms=500)
        recs = [m.value for msgs in batch.values() for m in msgs]
        if not recs:
            continue
        for res in pool.map(lambda r: score(a.api, r), recs):
            if res is None:
                errors += 1
                continue
            served[res[0]] += 1
            oc[res[1]] += 1
            n += 1
            if n % 200 == 0:
                tp, fp, fn = oc["tp"], oc["fp"], oc["fn"]
                prec = tp / (tp + fp) if tp + fp else 0.0
                rec_ = tp / (tp + fn) if tp + fn else 0.0
                f2 = 5 * prec * rec_ / (4 * prec + rec_) if prec + rec_ else 0.0
                print(f"[{n}] last 200: precision={prec:.2f} recall={rec_:.2f} F2={f2:.2f} | "
                      f"lag={kafka_lag(consumer)} rows errors={errors} | served_by={dict(served)}", flush=True)
                served.clear()
                oc.clear()


if __name__ == "__main__":
    main()
