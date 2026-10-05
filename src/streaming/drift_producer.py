"""Live drift-injecting producer: replays the stable validation segment at a constant rate
and applies the drift mode currently set in control/drift_mode.json."""
import argparse
import json
import pathlib
import time

import joblib
from kafka import KafkaProducer
from prometheus_client import Counter, Gauge, start_http_server

from src.drift.drift_generators import DRIFT_NAMES, apply_drift

KAFKA = "localhost:9092"
TOPIC = "transactions"
CONTROL = pathlib.Path("control/drift_mode.json")
MODES = ["normal"] + list(DRIFT_NAMES)

ROWS_SENT = Counter("stream_rows_sent_total", "Rows published to Kafka")
INJECTED_MODE = Gauge("stream_injected_mode", "1 for the drift mode currently injected", ["mode"])


def read_control():
    try:
        c = json.loads(CONTROL.read_text())
        mode = c.get("mode", "normal")
        return (mode if mode in MODES else "normal"), float(c.get("frac", 0.3))
    except Exception:
        return "normal", 0.3


def publish_mode(mode):
    for m in MODES:
        INJECTED_MODE.labels(mode=m).set(1 if m == mode else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tps", type=float, default=50)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--rows", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--metrics-port", type=int, default=8002)
    a = ap.parse_args()

    train = joblib.load("data/processed/train_processed_v2.pkl")
    ref = train.iloc[-40000:]
    val = joblib.load("data/processed/val_processed_v2.pkl")
    df = val.iloc[:a.rows].reset_index(drop=True)

    CONTROL.parent.mkdir(exist_ok=True)
    CONTROL.write_text(json.dumps({"mode": "normal", "frac": 0.3}))
    start_http_server(a.metrics_port)
    publish_mode("normal")

    producer = KafkaProducer(
        bootstrap_servers=KAFKA,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8"),
    )

    interval = 1.0 / a.tps
    next_t = time.time()
    sent, last_mode = 0, None
    print(f"Streaming {len(df)} rows at {a.tps} tps; warm-up {a.warmup} rows (no injection).")

    for i in range(0, len(df), a.batch):
        chunk = df.iloc[i:i + a.batch]
        mode, frac = read_control()
        if sent < a.warmup:
            mode = "normal"
        if mode != "normal":
            chunk = apply_drift(chunk, mode, ref, frac=frac, seed=i)
        if mode != last_mode:
            print(f"[row {sent}] injection mode -> {mode}", flush=True)
            publish_mode(mode)
            last_mode = mode
        for rec in chunk.to_dict("records"):
            producer.send(TOPIC, key=rec["TransactionID"], value=rec)
            sent += 1
            ROWS_SENT.inc()
            next_t += interval
            delay = next_t - time.time()
            if delay > 0:
                time.sleep(delay)
        if sent == a.warmup:
            print(f"[row {sent}] warm-up complete; injection enabled", flush=True)

    producer.flush()
    publish_mode("normal")
    print(f"Done. Sent {sent} rows.")


if __name__ == "__main__":
    main()