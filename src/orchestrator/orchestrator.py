"""Orchestrator: consumes the stream, fingerprints rolling windows, applies the routing
policy, and drives the API router. Exports Prometheus metrics and a decisions log."""
import argparse
import json
import os
import time
from collections import deque
from datetime import datetime, timezone

import joblib
import pandas as pd
import requests
from kafka import KafkaConsumer
from prometheus_client import Counter, Gauge, start_http_server

from src.router.fingerprint import GROUPS, classify
from src.router.policy import ROUTES, RoutingPolicy

KAFKA = "localhost:9092"
TOPIC = "transactions"
DECISIONS_LOG = "logs/decisions.jsonl"
GROUP_THR, BREADTH_THR = 1.25, 20
LABELS = ["normal", "A", "B", "C", "U"]
STATES = ["warming_up"] + LABELS

DRIFT_SCORE = Gauge("orch_drift_score", "Group drift score (mean top-3 KS/tau)", ["group"])
BREADTH = Gauge("orch_drift_breadth", "Number of ungrouped features drifting at once")
DETECTED = Gauge("orch_detected_label", "1 for the label detected in the latest window", ["label"])
STATE = Gauge("orch_policy_state", "1 for the current policy state", ["state"])
GROUP_THRESHOLD = Gauge("orch_group_threshold", "Group score alarm threshold")
BREADTH_THRESHOLD = Gauge("orch_breadth_threshold", "Breadth alarm threshold")
WINDOWS = Counter("orch_windows_total", "Windows evaluated")
DECISIONS = Counter("orch_decisions_total", "Routing decisions", ["event"])
API_ERRORS = Counter("orch_api_errors_total", "Failed API admin calls")
GUARDRAIL = Counter("orch_guardrail_total", "Guardrail rollbacks", ["model"])
GUARD_F2 = Gauge("orch_guardrail_f2", "Live F2 of the active model since it was routed in (-1 = not enough rows)")
GUARD_FLOOR = Gauge("orch_guardrail_floor", "Guardrail F2 floor")
FALLBACK = {
    "fraud-spec-A": "fraud-universal",
    "fraud-spec-B": "fraud-universal",
    "fraud-spec-C": "fraud-universal",
    "fraud-universal": "fraud-detection-model",
}


def guard_status(api):
    try:
        r = requests.get(f"{api}/admin/performance", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        API_ERRORS.inc()
        print(f"  !! performance call failed: {e}", flush=True)
        return None


def df_from(records, cols):
    return pd.DataFrame(list(records))[cols].astype("float64")


def publish_state(state):
    for s in STATES:
        STATE.labels(state=s).set(1 if s == state else 0)


def set_route(api, model, reason):
    try:
        r = requests.post(f"{api}/admin/route", json={"model": model, "weight": 1.0, "reason": reason}, timeout=5)
        r.raise_for_status()
        return True
    except Exception as e:
        API_ERRORS.inc()
        print(f"  !! API route call failed: {e}", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--step", type=int, default=250)
    ap.add_argument("--ref", type=int, default=5000)
    ap.add_argument("--confirm", type=int, default=2)
    ap.add_argument("--recover", type=int, default=3)
    ap.add_argument("--group", default=f"orchestrator-{int(time.time())}")
    ap.add_argument("--metrics-port", type=int, default=8001)
    ap.add_argument("--floor", type=float, default=0.5, help="guardrail F2 floor")
    ap.add_argument("--min-rows", type=int, default=150)
    ap.add_argument("--min-pos", type=int, default=10)
    ap.add_argument("--guard-strikes", type=int, default=2)
    ap.add_argument("--recall-floor", type=float, default=0.5, help="guardrail also needs recall below this")
    a = ap.parse_args()

    os.makedirs("logs", exist_ok=True)
    fp = joblib.load("models/fingerprinter.pkl")
    cols = fp.features
    policy = RoutingPolicy(confirm=a.confirm, recover=a.recover)

    start_http_server(a.metrics_port)
    GROUP_THRESHOLD.set(GROUP_THR)
    BREADTH_THRESHOLD.set(BREADTH_THR)
    publish_state("warming_up")
    for lab in LABELS:
        DETECTED.labels(label=lab).set(1 if lab == "normal" else 0)

    set_route(a.api, ROUTES["normal"], "orchestrator start")
    GUARD_FLOOR.set(a.floor)
    GUARD_F2.set(-1)
    for m in FALLBACK:
        GUARDRAIL.labels(model=m).inc(0)
    for ev in ("switch", "recover", "guardrail"):
        DECISIONS.labels(event=ev).inc(0)
    strikes = 0

    consumer = KafkaConsumer(
        TOPIC, bootstrap_servers=KAFKA, auto_offset_reset="latest", group_id=a.group,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )
    print(f"Orchestrator ready (group {a.group}). Start the producer now; "
          f"first {a.ref} rows form the clean reference.", flush=True)

    clean_buf = deque(maxlen=a.ref)
    recent = deque(maxlen=a.window)
    since, ready, seen = [], False, 0

    for msg in consumer:
        rec = msg.value
        seen += 1
        recent.append(rec)

        if not ready:
            clean_buf.append(rec)
            if len(clean_buf) == a.ref:
                fp.set_reference(df_from(clean_buf, cols))
                ready = True
                publish_state(policy.state)
                print(f"[rows {seen}] warm-up complete; monitoring started", flush=True)
            continue

        since.append(rec)
        if len(since) < a.step:
            continue

        s = fp.score(df_from(recent, cols))
        label = classify(s, group_thr=GROUP_THR, breadth_thr=BREADTH_THR)
        d = policy.update(label)

        if d["event"] in ("switch", "recover"):
            reason = f"fingerprint={label}" if d["event"] == "switch" else "traffic normal again"
            set_route(a.api, d["route"], reason)
            DECISIONS.labels(event=d["event"]).inc()

        perf = guard_status(a.api)
        guard_event = None
        if perf:
            act = perf["active"]
            rec = perf["slots"].get(act) or {}
            enough = (rec.get("n", 0) >= a.min_rows and rec.get("positives", 0) >= a.min_pos
                      and rec.get("f2") is not None)
            GUARD_F2.set(rec["f2"] if enough else -1)
            pos = rec.get("tp", 0) + rec.get("fn", 0)
            recall = rec["tp"] / pos if pos else 1.0
            bad = (enough and act in FALLBACK and label != "normal"
                   and rec["f2"] < a.floor and recall < a.recall_floor)
            strikes = strikes + 1 if bad else 0
            if strikes >= a.guard_strikes:
                fb = FALLBACK[act]
                why = f"guardrail: F2 {rec['f2']:.2f} < {a.floor} on last {rec['n']} rows"
                if set_route(a.api, fb, why):
                    GUARDRAIL.labels(model=act).inc()
                    DECISIONS.labels(event="guardrail").inc()
                    guard_event = f"{act} -> {fb} (F2 {rec['f2']:.2f})"
                    print(f"[rows {seen}] GUARDRAIL rollback {guard_event}", flush=True)
                strikes = 0

        if label == "normal" and policy.state == "normal":
            clean_buf.extend(since)
            fp.set_reference(df_from(clean_buf, cols))
        since = []

        WINDOWS.inc()
        for g in GROUPS:
            DRIFT_SCORE.labels(group=g).set(s[g])
        BREADTH.set(s["breadth"])
        for lab in LABELS:
            DETECTED.labels(label=lab).set(1 if lab == label else 0)
        publish_state(policy.state)

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(), "rows_seen": seen, "label": label,
            "event": d["event"], "state": policy.state, "route": d["route"],
            "scores": {g: round(s[g], 3) for g in GROUPS}, "breadth": s["breadth"],
            "active": perf["active"] if perf else None, "guardrail": guard_event,
        }
        with open(DECISIONS_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[rows {seen}] label={label:6s} A={s['A']:.2f} B={s['B']:.2f} C={s['C']:.2f} "
              f"breadth={s['breadth']:3d} event={d['event']:8s} route={d['route']}", flush=True)


if __name__ == "__main__":
    main()